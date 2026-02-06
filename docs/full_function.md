# 项目功能全景与规格边界

给一份配置，`csxgb` 会跑完整研究流水线：拉数、构池、打标签、做特征、训练、评估、回测、落盘产物、输出持仓快照。

## 文档定位

为避免和其他文档重复且减少“文档漂移”，本文件定位为：

* 全景流程说明
* 关键参数入口索引
* 难点、边界、非目标与工时估算

完整参数字典和默认值请以 `docs/config.md` 为准。  
CLI 全量参数请以 `docs/cli.md` 和 `csxgb <cmd> --help` 为准。  
输出字段 Schema 请以 `docs/outputs.md` 为准。

## 端到端流程

1. 按 `market + universe` 确定股票池（静态或 PIT by-date）。
2. 通过 `data.provider` 拉取日线，按规则补齐字段并过滤样本。
3. 可选合并 `fundamentals`。
4. 按 `label` 生成收益标签。
5. 按 `features` 生成技术特征并做横截面变换。
6. 训练 `XGBRegressor`，做时间序列评估与稳定性检验。
7. 运行 Top-K 回测与持仓输出（可含 OOS/live）。
8. 写入 run 目录与 `summary.json` 供复现、汇总、持仓查询。

## 模块规格（关键参数索引）

### 1) Universe（股票池与样本过滤）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `universe.mode` | `str` | `auto` | `auto/pit/static` | 股票池模式 |
| `universe.require_by_date` | `bool` | `false` | - | 强制要求 PIT 文件 |
| `universe.by_date_file` | `path` | `null` | `pit` 模式必填 | PIT 股票池输入 |
| `universe.symbols/symbols_file` | `list/path` | 模板内置列表 | 静态池入口 | 静态股票池 |
| `universe.min_symbols_per_date` | `int` | `eval.n_quantiles` | 自动下限到 `n_quantiles` | 小截面日期过滤 |
| `universe.min_listed_days` | `int` | `0` | `>=0` | 上市天数过滤 |
| `universe.drop_st` | `bool` | `false` | CN 语义更强 | 名称含 ST 的股票过滤 |
| `universe.drop_suspended` | `bool` | `true` | - | 启用可交易性判定 |
| `universe.suspended_policy` | `str` | `mark` | `mark/filter` | 标记或删除停牌样本 |
| `universe.min_turnover` | `float` | `0` | `>=0` | `amount` 成交额下限 |

关键行为：

* `by_date_file` 存在时会进入 PIT 逻辑；`mode=static` 但给了 `by_date_file` 也会切到 PIT 并告警。
* `drop_suspended=true` 先计算 `is_tradable`。只有 `suspended_policy=filter` 才真正删行，不是默认就删。
* 如果 `eval.sample_on_rebalance_dates=true`，先按再平衡日采样，再做 `min_symbols_per_date` 日期过滤。

相关命令：

* `csxgb universe hk-connect`：构建港股通 PIT 股票池（流动性过滤）。
* `csxgb universe index-components`：拉指数成分，可同时产出 PIT by-date CSV。

### 2) Data（数据源、缓存、符号映射）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `market` | `str` | `cn` | `cn/hk/us` | 市场口径 |
| `data.provider` | `str` | `tushare` | `tushare/rqdata/eodhd` | 数据源 |
| `data.start_date/start_years` | `str/num` | `start_years=5` | `start_date` 优先 | 起始时间 |
| `data.end_date` | `str` | `today` | 支持 `today/t-1/...` | 截止时间 |
| `data.price_col` | `str` | `close` | 列必须存在 | 标签/回测用价 |
| `data.cache_*` | - | - | 见 `docs/config.md` | 缓存策略 |
| `data.retry.*` | - | 见模板 | - | 重试与 token 轮换 |
| `data.eodhd.hk_symbol_mode` | `str` | `null` | `strip_one/strip_all/pad4/pad5` | HK 符号转换 |

关键行为：

* provider 别名会标准化（如 `rqdatac -> rqdata`，`ts -> tushare`）。
* HK 内部代码标准化为 `00001.HK`；RQData 会转为 `00001.XHKG`。
* `TUSHARE_TOKEN/TUSHARE_TOKEN_2` 可轮换；RQData 支持配置和环境变量混合初始化。

### 3) Fundamentals（可选基本面合并）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `fundamentals.enabled` | `bool` | `false`(代码默认) | - | 是否合并基本面 |
| `fundamentals.source` | `str` | `provider` | `provider/file` | 数据来源 |
| `fundamentals.required` | `bool` | `false` | - | 缺失时是否直接报错 |
| `fundamentals.column_map` | `dict` | `{}` | - | 字段标准化 |
| `fundamentals.features` | `list` | `[]` | - | 要并入的特征列 |
| `fundamentals.allow_missing_features` | `bool` | `false` | - | 缺失列是否放行 |
| `fundamentals.ffill/ffill_limit` | `bool/int` | `true/null` | - | 按 `ts_code` 前向填充 |
| `fundamentals.log_market_cap*` | - | - | - | 市值对数特征 |

关键行为：

* `source=provider` 当前只支持 TuShare；其它 provider 会禁用并告警，`required=true` 则报错退出。
* `source=file` 支持 CSV/Parquet。

### 4) Label（标签）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `label.horizon_mode` | `str` | `fixed` | `fixed/next_rebalance` | 标签窗口定义 |
| `label.horizon_days` | `int` | `5` | 通常设为正数 | `fixed` 模式持有期 |
| `label.shift_days` | `int` | `0` | 通常设为非负数 | 入场价格位移 |
| `label.rebalance_frequency` | `str` | 继承 `eval` | - | `next_rebalance` 频率 |
| `label.target_col` | `str` | `future_return` | - | 标签列名 |
| `label.winsorize_pct` | `float/null` | `null` | `0 < x < 0.5` | 标签截尾 |

关键行为：

* `next_rebalance` 使用“交易日 -> 下一个 rebalance 日”的映射求 `exit_price`。
* 若再平衡点不足会回退为 `fixed` 并告警。

### 5) Features（特征与横截面变换）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `features.list` | `list[str]` | 内置技术指标集合 | 不能为空 | 参与训练列 |
| `features.params.sma_windows` | `list[int]` | `[]` | - | SMA 窗口 |
| `features.params.rsi` | `int/list` | `14` | - | RSI 窗口 |
| `features.params.macd` | `list[int]` | `[12,26,9]` | 长度 3 | MACD 参数 |
| `features.params.volume_sma_windows` | `list[int]` | `[]` | - | 量能窗口 |
| `features.cross_sectional.method` | `str` | `none` | `none/zscore/rank` | 横截面标准化 |
| `features.cross_sectional.winsorize_pct` | `float/null` | `null` | `0 < x < 0.5` | 横截面截尾 |

当前内置技术特征形态：

* `sma_{w}`、`sma_{w}_diff`
* `rsi_{n}`
* `macd_hist`
* `volume_sma{w}_ratio`
* `vol`

### 6) Model（XGBoost）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `model.type` | `str` | `xgb_regressor` | 当前仅回归流程 | 模型类型 |
| `model.params` | `dict` | 内置默认参数 | 传给 `XGBRegressor` | 模型超参 |
| `model.sample_weight_mode` | `str` | `none` | `none/date_equal` | 样本权重方案 |

### 7) Eval（评估与稳健性）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `eval.test_size` | `float` | `0.2` | 通常使用 0~1 比例 | 训练/测试切分 |
| `eval.n_splits` | `int` | `5` | 建议 `>=2` | 时间序列 CV 折数 |
| `eval.n_quantiles` | `int` | `5` | 建议 `>=2` | 分位数组数 |
| `eval.signal_direction` | `float` | `1` | 不能为 `0` | 信号方向 |
| `eval.signal_direction_mode` | `str` | `fixed` | `fixed/train_ic/cv_ic` | 自动翻转模式 |
| `eval.min_abs_ic_to_flip` | `float` | `0` | `>=0` | 翻转阈值 |
| `eval.embargo_days/purge_days` | `int/null` | `null` | - | 泄漏隔离 gap |
| `eval.sample_on_rebalance_dates` | `bool` | `false` | - | 仅在再平衡日采样 |
| `eval.save_artifacts` | `bool` | `true` | - | 是否落盘 |
| `eval.save_dataset` | `bool` | `false` | 需 `save_artifacts=true` | 导出数据集 |
| `eval.permutation_test.*` | - | - | - | 置换检验 |
| `eval.walk_forward.*` | - | - | - | 滚动窗口验证 |
| `eval.final_oos.*` | - | - | - | 最终留出期 |
| `eval.rolling.*` | - | - | - | 滚动 IC/Sharpe |
| `eval.bucket_ic.*` | - | - | - | 分桶 IC |

关键行为：

* 训练集和测试集评估均输出，且支持 Spearman/Pearson IC。
* `permutation_test` 是按 `trade_date` 分组打乱标签，不是全局洗牌。
* `walk_forward` 可配置是否同时跑回测（`backtest_enabled`）。

### 8) Backtest 与 Execution（回测）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `backtest.enabled` | `bool` | `true` | - | 是否跑回测 |
| `backtest.top_k` | `int` | 继承 `eval.top_k` | - | 持仓数量 |
| `backtest.rebalance_frequency` | `str` | 继承 `eval` | - | 调仓频率 |
| `backtest.long_only/short_k` | `bool/int` | `true/null` | - | 多空模式 |
| `backtest.transaction_cost_bps` | `float` | 继承 `eval` | - | 成本基线 |
| `backtest.buffer_exit/entry` | `int` | `0/0` | - | 缓冲区降换手 |
| `backtest.exit_mode` | `str` | `rebalance` | `rebalance/label_horizon` | 退出模式 |
| `backtest.exit_horizon_days` | `int/null` | `null` | `label_horizon` 下生效 | 固定持有期 |
| `backtest.exit_price_policy` | `str` | `strict` | `strict/ffill/delay` | 出场价口径 |
| `backtest.exit_fallback_policy` | `str` | `ffill` | `ffill/none` | 延迟失败回退 |
| `backtest.tradable_col` | `str/null` | `is_tradable` | - | 可交易列 |
| `backtest.execution.cost_model` | `dict` | `bps` | `bps/none` | 成本模型扩展 |
| `backtest.execution.exit_policy` | `dict` | 继承旧键 | `strict/ffill/delay` | 退出策略扩展 |

关键行为：

* `cost_model.name=none` 可关闭成本；`bps` 支持 `round_trip`。
* `exit_mode=label_horizon` 与调仓频率冲突时会跳过或报错，不是无条件可用。

### 9) Live（当前持仓快照）

| 参数键 | 类型 | 默认值 | 约束/取值 | 作用 |
| --- | --- | --- | --- | --- |
| `live.enabled` | `bool` | `false` | 需 `eval.save_artifacts=true` | 是否输出 live 持仓 |
| `live.as_of` | `str` | `t-1` | 支持日期 token | live 截止日 |
| `live.train_mode` | `str` | `full` | `full/train` | 训练样本策略 |

## CLI 总览（按命令组）

研究主流程：

* `csxgb run`
* `csxgb grid`
* `csxgb summarize`
* `csxgb holdings`
* `csxgb snapshot`
* `csxgb init-config`

数据源工具：

* `csxgb rqdata info`
* `csxgb rqdata quota`
* `csxgb tushare verify-token`

股票池工具：

* `csxgb universe index-components`
* `csxgb universe hk-connect`

说明：

* `universe index-components`、`universe hk-connect`、`tushare verify-token` 采用“CLI 转发参数到底层脚本”模式，参数以脚本 `--help` 为准。
* `grid` 当前机制是“先跑一次 base pipeline 产出 `eval_scored.parquet`，再在同一份 scored 数据上循环 Top-K/成本组合”，不是每个组合都重新训练。

## 输出产物（关键文件）

run 目录规则：

* `<eval.output_dir>/<run_name>_<timestamp>_<config_hash>/`

通用核心文件：

* `summary.json`
* `config.used.yml`
* `feature_importance.csv`
* `eval_scored.parquet`（若可用）
* `dataset.parquet`（`eval.save_dataset=true`）

评估文件：

* `ic_test.csv`
* `ic_pearson_test.csv`
* `ic_train.csv`、`ic_pearson_train.csv`（启用时）
* `quantile_returns.csv`
* `turnover_eval.csv`
* `permutation_test.csv`（启用时）
* `walk_forward_summary.csv`（启用时）

回测文件：

* `backtest_net.csv`
* `backtest_gross.csv`
* `backtest_turnover.csv`
* `backtest_periods.csv`
* 可选：`backtest_benchmark.csv`、`backtest_active.csv`

持仓文件：

* 回测：`positions_by_rebalance.csv`、`positions_current.csv`、`rebalance_diff.csv`
* OOS：`positions_by_rebalance_oos.csv`、`positions_current_oos.csv`、`rebalance_diff_oos.csv`
* Live：`positions_by_rebalance_live.csv`、`positions_current_live.csv`、`rebalance_diff_live.csv`

live 最新指针：

* `latest.json` 仅在 `live.enabled=true` 时写入，路径在 `eval.output_dir` 根目录下，不是固定写死到 `out/live_runs/`。

## 难点（工程视角）

1. 多 provider/多市场的数据一致性与符号标准化。
2. PIT 股票池与样本过滤顺序对结果影响很大。
3. 泄漏防控要贯穿标签、切分、评估、回测四层。
4. 评估体系不是单指标问题，需 CV + walk-forward + permutation 组合验证。
5. 回测细节（成本、buffer、退出策略、可交易性）决定策略可落地性。
6. 可复现依赖配置冻结、缓存策略、产物落盘和代码版本固定。

## 边界与非目标

本项目当前定位是“研究 + 持仓建议快照”，明确不覆盖以下能力：

* 券商/OMS 账户对接与自动下单执行。
* 成交回执、撤单重试、盘中执行控制。
* 涨跌停、盘口冲击、复杂成交模型等微观结构仿真。
* 账户级风控约束（行业/风格/敞口/现金管理）的一体化执行闭环。

补充边界：

* `holdings/snapshot` 输出的是目标持仓，不等同真实成交持仓。
* 交易日历 token 在部分场景可能退化为自然日逻辑（无交易日历时会给告警）。
* 数据供应商回补/修订会导致“同配置不同时间”结果变化。

## 工时估算（前提 + 三档）

前提假设：

* 单人开发，具备 Python + 量化研究工程经验。
* 数据源权限、token、配额已就绪。
* 至少包含基础测试与文档，不含 OMS/实盘执行系统改造。
* 市场和 provider 数量越多，工时按倍数上升。

三档估算：

| 档位 | 范围定义 | 预计工时 |
| --- | --- | --- |
| 基础可用版 | 单市场、单 provider、主流程可跑通、基础产物落盘 | 180-280 小时 |
| 可复现研究版 | 多市场/多 provider、PIT 股票池、稳健评估、回归测试与文档闭环 | 280-450 小时 |
| 可实盘运维版 | 在研究版基础上补执行接入、监控、审计、失败恢复与运维工具 | 450-800+ 小时 |

结论：

* 仅做“研究可用”通常很难低于 200 小时。
* 若目标是“长期可复现 + 可运维”，应按 300 小时以上预算。
