# 项目全量功能矩阵、难点分层与工时重估（人天）

给一份配置，`csml` 会跑完整研究流水线：拉数、构池、打标签、做特征、训练、评估、回测、落盘产物、输出持仓快照。

## 文档定位

本文件用于回答三个问题：

1. 这个项目有哪些功能。
2. 这些功能的关键工程难点分布在哪一层。
3. 以人天口径看，开发/维护预算应怎么估。

说明：

* 本文是全量功能矩阵 + 难点与工时判断。
* 参数字典与默认值仍以 `docs/config.md` 为准。
* CLI 全量参数仍以 `docs/cli.md` 和 `csml <cmd> --help` 为准。
* 输出字段 Schema 仍以 `docs/outputs.md` 为准。

## 一、全量功能矩阵

### 1) CLI 命令矩阵（用户可见入口）

| 命令 | 能力 | 关键输入 | 关键输出/副作用 | 备注 |
| --- | --- | --- | --- | --- |
| `csml run` | 主流程：训练/评估/回测/持仓产物 | `--config` | `out/runs/...` 全套产物 | 研究主入口 |
| `csml grid` | Top-K × 成本 × buffer（入/出）敏感性网格 | base config + grid 参数 | `grid_summary.csv` | 先跑一次 base，再复用 scored 数据；详见 [cli.md#2-csml-grid](cli.md#2-csml-grid) |
| `csml sweep-linear` | Ridge/ElasticNet 参数 sweep | `--sweep-config` 或 CLI 网格参数 | `out/sweeps/<tag>/` + 自动 summarize | 已覆盖（不是 `sweep`，是 `sweep-linear`） |
| `csml summarize` | 聚合历史 run 关键指标 | `--runs-dir`、筛选参数 | `runs_summary.csv` | 研究对比入口 |
| `csml holdings` | 输出当前持仓 | `--config/--run-dir`、`--as-of` | text/csv/json | 支持 `auto/backtest/live` |
| `csml snapshot` | 一键 run + holdings | `--config` 或 `--run-dir` | live/回测快照 | 适合 cron/CI |
| `csml alloc` | 从持仓做等权手数分配 | 持仓来源 + 资金参数 | 分配表（text/csv/json） | 依赖 RQData 价格和 round lot |
| `csml init-config` | 导出内置模板配置 | `--market`、`--out` | 本地 YAML 模板 | 支持 `--force` 覆盖 |
| `csml rqdata info` | 检查 RQData 初始化信息 | 可选 `--config` / 账号覆盖 | 登录/用户信息 | 运维排障辅助 |
| `csml rqdata quota` | 查询 RQData 配额 | 可选 `--pretty` | 配额信息 | 运维排障辅助 |
| `csml tushare verify-token` | 验证 TuShare token 可用性 | token 环境变量/透传参数 | 验证结果 | 运维排障辅助 |
| `csml universe index-components` | 拉指数成分并可生成 PIT 文件 | 脚本透传参数 | symbols/by-date 文件 | CLI 透传到底层脚本 |
| `csml universe hk-connect` | 构建港股通 PIT universe | `--config` + 脚本透传参数 | by-date + symbols + meta | 含流动性过滤 |

### 2) Pipeline 模块矩阵（`csml run` 内部能力）

| 模块 | 已实现能力 | 关键参数入口 | 典型风险点 |
| --- | --- | --- | --- |
| Universe | `auto/pit/static` 股票池，按日期过滤、停牌/上市天数/成交额过滤 | `universe.*` | 过滤顺序改变结果；PIT 数据缺失 |
| Data | `tushare/rqdata/eodhd`，market-aware symbol 规则，缓存与重试 | `market`、`data.*` | 多 provider 口径不一致 |
| Fundamentals | provider/file 两路并入，列映射与缺失策略 | `fundamentals.*` | provider 能力不对齐 |
| Label | `fixed/next_rebalance`，shift 与截尾 | `label.*` | 时序泄漏、标签口径偏差 |
| Features | 技术特征生成 + 横截面 `none/zscore/rank` | `features.*` | 窗口与样本可用性冲突 |
| Model | `xgb_regressor/xgb_ranker/ridge/elasticnet` | `model.*` | 模型与样本权重设定不当 |
| Eval | train/test + CV IC，direction flip，置换检验，walk-forward，final OOS，rolling/bucket 指标 | `eval.*` | 仅看单指标导致过拟合 |
| Backtest/Execution | Top-K、多空、成本模型、buffer、exit policy、tradable 约束 | `backtest.*` | 回测语义与真实交易偏差 |
| Live | 产出 live 持仓文件与 `latest.json` 指针 | `live.*` | 依赖 `eval.save_artifacts=true` |
| Reproducibility | 冻结配置、哈希 run 目录、核心产物落盘 | `eval.output_dir` 等 | 缓存/数据回补导致重跑差异 |

### 3) 输出产物矩阵（run 目录）

run 目录规则：

* `<eval.output_dir>/<run_name>_<timestamp>_<config_hash>/`

| 类别 | 文件 | 触发条件 | 用途 |
| --- | --- | --- | --- |
| 核心 | `summary.json` | 默认 | 汇总关键指标，供 summarize 消费 |
| 核心 | `config.used.yml` | 默认 | 复现实验所用配置 |
| 核心 | `feature_importance.csv` | 模型支持时 | 解释性与特征筛选 |
| 核心 | `eval_scored.parquet` | 可用时 | grid/summarize/二次分析 |
| 核心 | `dataset.parquet` | `eval.save_dataset=true` | 诊断/复查输入样本 |
| 评估 | `ic_test.csv`、`ic_pearson_test.csv` | 默认 | 核心 IC 评估 |
| 评估 | `ic_train.csv`、`ic_pearson_train.csv` | 启用时 | 训练期对照 |
| 评估 | `quantile_returns.csv`、`turnover_eval.csv` | 默认 | 分层收益与换手 |
| 评估 | `permutation_test.csv` | 启用置换检验 | 抗伪发现 |
| 评估 | `walk_forward_summary.csv` | 启用 walk-forward | 时变稳健性 |
| 回测 | `backtest_net.csv`、`backtest_gross.csv` | `backtest.enabled=true` | 净值与成本拖累 |
| 回测 | `backtest_turnover.csv`、`backtest_periods.csv` | `backtest.enabled=true` | 换手与周期收益 |
| 回测 | `backtest_benchmark.csv`、`backtest_active.csv` | 配 benchmark 时 | 主动收益分析 |
| 持仓 | `positions_by_rebalance.csv` 等 | 回测开启 | 历史调仓与当前持仓 |
| 持仓 | `*_oos.csv` | 启用 final OOS | 留出期持仓 |
| 持仓 | `*_live.csv` | `live.enabled=true` | live 目标持仓 |
| 指针 | `latest.json` | `live.enabled=true` | 最新 live run 定位 |

### 4) 明确边界（当前不覆盖）

* 券商/OMS 账户对接与自动下单执行。
* 成交回执、撤单重试、盘中执行控制。
* 涨跌停、盘口冲击、复杂成交模型等微观结构仿真。
* 账户级风控约束（行业/风格/敞口/现金管理）的一体化执行闭环。

补充边界：

* `holdings/snapshot` 输出的是目标持仓，不等同真实成交持仓。
* 交易日历 token 在部分场景可能退化为自然日逻辑（无交易日历时会给告警）。
* 数据供应商回补/修订会导致“同配置不同时间”结果变化。

## 二、难点分层（工程 + 研究）

| 层级 | 难点主题 | 为什么难 | 典型失败模式 | 降险措施 |
| --- | --- | --- | --- | --- |
| L1 数据接入层 | 多 provider、多市场、符号标准化 | 同名字段语义不一致，符号体系不同 | 某市场可跑、跨市场失真 | 统一 symbol 规范 + provider 适配测试 |
| L1 数据接入层 | API 配额、失败重试、token 轮换 | 第三方服务不稳定且有频率限制 | 间歇失败、批量任务中断 | 重试/退避/轮换 + 配额监控 |
| L2 研究正确性层 | PIT universe 与过滤顺序 | 顺序不同会改变样本分布 | “能跑通但结果漂移” | 固化顺序、日志化样本数变化 |
| L2 研究正确性层 | 标签/切分/评估泄漏防控 | 泄漏点跨多个模块 | 指标异常高，实盘失效 | purge/embargo（含默认推导与告警）+ 时间切分单测 |
| L2 研究正确性层 | 稳健性验证组合 | 单一指标不能代表可交易性 | 过拟合模型上线 | CV + walk-forward + permutation 联检 |
| L3 回测语义层 | 成本、buffer、退出策略、可交易性 | 每个细节都影响收益与换手 | 回测结果过于乐观 | 参数显式化 + 默认值审计 |
| L3 回测语义层 | `label_horizon` 与调仓频率协同 | 退出与再平衡可能冲突 | 逻辑跳过或行为不一致 | 冲突检测与报错策略 |
| L4 可复现运维层 | 可复现依赖“数据+配置+代码”三方冻结 | 任一变化都可造成结果偏差 | 同配置重跑不一致 | `config.used.yml` + run hash + 缓存标记 |
| L4 可复现运维层 | 研究工具链编排（grid/sweep/summarize） | 批处理链路长，局部失败常见 | 半途失败、汇总不完整 | `continue-on-error` + 状态文件落盘 |

判断标准：

* L1-L2 偏“能不能做对”。
* L3 偏“回测是否接近可执行现实”。
* L4 偏“是否可长期维护与复现”。

## 三、工时重估（按人天）

口径说明：

* 1 人天 = 8 小时。
* 估算针对“单人开发，具备 Python + 量化研究工程经验”。
* 默认数据权限/配额已就绪，不含 OMS/自动交易系统改造。

### 1) 自下而上拆分（研究平台范围）

| 工作包 | 主要内容 | 估算（人天） |
| --- | --- | --- |
| 基础工程骨架 | CLI、配置解析、日志、目录约定 | 4-8 |
| 数据与 provider | 多源适配、字段标准化、缓存、重试 | 10-18 |
| Universe 工具链 | PIT 处理、指数成分、港股通构建 | 8-14 |
| 标签与特征 | 标签口径、技术特征、横截面变换 | 8-14 |
| 建模与评估 | 多模型、CV/IC、稳健性检验 | 8-16 |
| 回测与执行语义 | 成本、退出、buffer、持仓输出 | 10-20 |
| 研究编排工具 | `grid/sweep-linear/summarize` | 8-16 |
| 结果消费工具 | `holdings/snapshot/alloc` | 6-12 |
| 测试与回归 | 单测、集成测试、修复迭代 | 10-20 |
| 文档与示例 | README + docs + cookbook | 4-8 |
| 小计 |  | 76-146 |
| 风险缓冲 | API 波动、数据修订、需求返工（15%-25%） | 12-36 |
| 合计 | 研究平台总量级 | 88-182 人天 |

### 2) 三档预算（更便于立项）

| 档位 | 范围定义 | 重估（人天） |
| --- | --- | --- |
| 基础可用版 | 单市场、单 provider、主流程可跑、核心产物齐全 | 30-50 |
| 可复现研究版 | 多市场/多 provider、PIT、稳健评估、较完整测试与文档 | 60-100 |
| 准生产运维版 | 在研究版上补监控、审计、失败恢复、稳定批处理 | 100-160 |

扩展说明（不在当前项目边界内）：

* 若要补齐“券商接入 + 下单 + 回执 + 执行风控”闭环，通常还需额外 60-120 人天，取决于券商/OMS 复杂度。

### 3) 工时倍率因子（为什么会“比预期更长”）

* 每新增 1 个市场：总工时通常增加 15%-25%。
* 每新增 1 个 provider：总工时通常增加 20%-35%。
* 若要求强可复现（固定数据快照、严格审计）：增加 20%-40%。
* 若要求准实盘稳定性（定时任务、失败恢复、告警闭环）：增加 25%-50%。

## 结论

* `docs/full_function.md` 现版本已按“全量功能矩阵”列出 CLI 与 pipeline 能力，`sweep-linear` 已明确纳入。
* “难点”不应只看算法本身，主要工时消耗常在数据一致性、泄漏防控、回测语义和可复现运维。
* 以人天口径，项目级预算通常应按 60 人天以上（可复现研究目标）准备；若要长期稳定运维，应按 100 人天以上准备。
