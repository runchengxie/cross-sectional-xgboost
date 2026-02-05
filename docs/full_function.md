# 这个项目整体在做哪些事

你命令行量化全流程：给它一份配置，它会产出训练结果 + 回测结果 + 持仓文件 + 可追溯的文档记录。

## 项目功能清单

### 1) 选市场、选股票池（Universe）

* 动态股票池（PIT by-date）与最小截面规模
  * 支持 `universe.mode`（静态/文件/按日期），并可要求 by-date 文件必须覆盖（`require_by_date`）。
  * 会按 `min_symbols_per_date` 把截面太小的交易日整天丢掉，并记录 dropped_dates（还会落盘 dropped_dates.csv）。
  * `min_symbols_per_date` 会自动下限到 `n_quantiles`（避免分组数大于当日股票数）。
  * dropped_dates 口径是“用于建模/评估的样本”；若启用 `sample_on_rebalance_dates`，会先抽样再做 date-count 过滤。
* 上市天数过滤（min_listed_days）
  * 通过 `list_date` 计算 `listed_days = (trade_date - list_date).days`，再按 `MIN_LISTED_DAYS` 过滤。
* ST 过滤（drop_st）
  * ST 概念是 CN 特有；代码不强制限制 market，仅提示 CN-specific。其他市场通常不会命中（除非股票名里刚好含 ST）。
  * 基于 `basic_df["name"]` 是否包含 “ST”（大小写不敏感）得到 `st_codes`，再把这些代码剔除。
* 停牌/不可交易处理（drop_suspended + suspended_policy）
  * 先定义 `is_tradable = (vol > 0) & (amount > 0)`；若无 `amount` 字段则退化为仅 `vol > 0`。
  * `suspended_policy: "mark"`：不删除，只是保留 `is_tradable` 标记列（给回测/信号用）。
  * `suspended_policy: "filter"`：把 `is_tradable=False` 的行直接删掉。
  * 另一个硬开关 `drop_suspended` 也会触发过滤逻辑（本质就是“删掉停牌行”）。
* 最小流动性（min_turnover）
  * 用 `amount >= MIN_TURNOVER` 做过滤（注意这里叫 turnover 但实际字段是成交额 `amount`）。
* 港股通股票池PIT HK Connect universe：
  * 项目自带 `build_hk_connect_universe.py`：按 `lookback_days / min_window_days / top_quantile / min_turnover` 做流动性筛选，输出 `universe_by_date.csv` 和 `hk_connect_symbols.txt`，并可写 meta。

### 2) 拉数据

* provider 选择（tushare / rqdata / eodhd）
  * provider 名称规范化/别名映射在 `resolve_provider` 做，市场字符串规范在 `normalize_market` 做。
* HK 代码内部标准格式
  * 内部标准是 5 位补零 + `.HK`，例如 `1.HK / 0001.HK / 00001.XHKG` 都会被规范成 `00001.HK`。
* RQData 代码转换（HK）
  * 内部 `00001.HK` → RQData `00001.XHKG`（港股用 XHKG 交易所标记）。
* EODHD 代码转换（HK）与 hk_symbol_mode ，EODHD 的 HK 代码格式可以配置，支持这些模式
  * `strip_one`：去掉前导 0（至少保留一个字符），再拼 suffix（默认 `.HK`）。
  * `strip_all`：更激进地转成 int 再转回字符串（本质也是去前导 0，但更偏“数字化”）。
  * `pad4`：补零到 4 位。
  * `pad5`：补零到 5 位（与内部一致）。

配置支持基本面数据

* 配置层面支持：
  * `column_map` 做字段标准化，`features` 选择要合并进主表的基本面列，`required/allow_missing_features` 控制缺失时是报错还是放过。
* 实现层面支持：
  * 合并后会按 `ts_code` 做 `ffill`（可配 `ffill_limit`）。
  * `log_market_cap: true` 时，会把 `market_cap_col` 做 `np.log` 生成 `log_market_cap_col`（默认 `log_mcap`）。
  * fundamentals 开了但没拿到数据：`required=true` 会直接报错；否则只是警告然后继续跑。
  * `source=provider` 目前仅支持 Tushare；其他 provider 会被禁用并给出警告（`required=true` 时直接报错）。

### 3) 打标签（label）

对数据进行按再平衡周期定义的未来收益这种更贴近组合交易的标签逻辑

* `label.horizon_mode` 有两种：`fixed` 或 `next_rebalance`。
* `next_rebalance` 的真实逻辑：

  * 先算 rebalance dates，再把每个交易日映射到下一个 rebalance 日。
  * 用“下一个 rebalance 日对应的（已 shift 的）price”当 `exit_price`；`entry_price` 也按 shift 后的 price 计算。
  * `future_return = exit_price / entry_price - 1`。

### 4) 做特征（features）

当前实现的特征（就是配置 `features.list` ）：

* `sma_{w}`：对 `close` 做 SMA（窗口来自 `sma_windows`）。
* `sma_{w}_diff`：对 SMA 做 `pct_change`（即 SMA 的日度变化率）。
* `rsi_{n}`：RSI（`rsi` 参数默认 14）。
* `macd_hist`：MACD histogram（参数来自 `macd: [fast, slow, signal]`）。
* `volume_sma{w}_ratio`：`vol / SMA(vol,w)`（默认 w=5）。
* 以及原始 `vol`（成交量列本身）。

参数入口：`sma_windows / rsi / macd / volume_sma_windows`。

横截面标准化，`cross_sectional.method` 支持：

* `none`
* `zscore`：按日截面减均值除标准差（std=0 会处理成 NaN 后再填 0）。
* `rank`：按截面做分位排名（pct rank - 0.5）。
* `winsorize_pct` 是可选预处理（在 `zscore/rank` 前按分位数裁剪）。
  并且 method 合法性校验写在 pipeline 配置解析里。

### 5) 训练模型（XGBoost 回归）

配置里就是 XGB regressor，自带常用参数（n_estimators、max_depth、subsample…）以及 sample_weight 方案。

XGBRegressor(params)模型参数包括：

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `reg_alpha`
* `reg_lambda`
* `random_state`
* `min_child_weight`
* `gamma`

* `sample_weight_mode` 有：
  * `none`
  * `date_equal`：每个日期的样本权重 = `1 / 当天样本数`（让每个交易日权重相等）。

### 6) 评估

* n_splits、分位数分组、IC/IR。
  * `eval.n_splits` 控制时间序列 CV 的折数（默认 5）。
  * `eval.n_quantiles` 控制分位数分组数（默认 5）。
  * 输出 IC/IR 到 `ic_<oos/live>.csv`，分位数收益到 `quantile_returns.csv`。
* permutation test（打乱标签检验是不是靠运气）。
* walk-forward（滚动窗口验证，还可以连带回测）。
* 时间序列 CV + embargo/purge（其实是一个 gap）
  * 用 `TimeSeriesSplit(n_splits)`，然后 `gap = max(embargo_days, purge_days)`，把训练集尾部切掉避免泄露。
* IC/IR 统计项
  * IC 支持 `spearman` / `pearson`，按日算 corr，再汇总 `mean/std/ir/t_stat/p_value`（如果 scipy 可用）。
* 分位数组（quantile returns）
  * 每天对 `pred` 做 rank，再 `qcut` 分箱，算每箱平均未来收益，得到 `q_ret`（按 trade_date）。
* signal_direction_mode + min_abs_ic_to_flip（方向自适应阈值）
* `signal_direction_mode` 支持 `fixed/train_ic/cv_ic`。
* `cv_ic` 模式下：只有当 `|mean(IC)| >= min_abs_ic_to_flip` 才会把信号方向设为 `sign(mean_ic)`，否则保持原方向。
* permutation test
  * 核心：按 trade_date 分组打乱训练集 label（不是全局洗牌），每次跑一遍 CV IC，记录 mean/std/scores/runs。
* walk-forward
  * 支持 `n_windows / test_size / step_size / anchor_end`，会切出一系列滚动窗口（含 train_start/end、test_start/end）。

### 7) 回测（Top-K 组合 + 成本 + 交易规则）

回测参数包括：Top-K、再平衡频率、单边成本 bps、buffer 规则、exit_mode、价格/缺失处理策略。

* backtest 会用 `backtest_topk(...)`，参数里包含 top_k、rebalance_dates、buffer_exit/entry、long_only/short_k、tradable_col、exit_price_policy、exit_fallback_policy、execution model。
  * `exit_price_policy` 三选一：`strict/ffill/delay`；`exit_fallback_policy` 二选一：`ffill/none`。
  * 交易成本模型可配：
    * `none`
    * `bps`（可选 `round_trip`，默认 true）。
* 配置解析层附带校验。

而且它会落盘持仓文件：`positions_by_rebalance.csv`、`positions_current.csv`，以及 OOS 和 live 版本。

输出产物清单：

* 产物目录：`out/runs/<run_name>_<timestamp>_<hash>/`
* 典型产物：`summary.json`、`config.used.yml`、`ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`、`feature_importance.csv`
* 持仓清单：`positions_by_rebalance.csv`、`positions_current.csv`、`signal_asof`、`next_entry_date`、`holding_window` 字段；`holding_window` 约定为 `entry_date -> next_entry_date`（next 为空表示最新持仓区间）。
* Live 持仓清单：`positions_by_rebalance_live.csv`、`positions_current_live.csv`
* 再平衡差异：`rebalance_diff.csv`、`rebalance_diff_live.csv`
* Live 最新指针：`out/live_runs/latest.json`（指向最新 live run）

### 8) 把这一切包装成 CLI 工具，可复现、可实盘化

完整 CLI：`run/grid/holdings/snapshot/rqdata quota/.../universe hk-connect`。
输出目录结构也规定了：每次 run 一个带 hash/timestamp 的目录，里面有 `summary.json`、`config.used.yml`、回测/IC/特征重要性/持仓等。
`holdings` 会去读最新 run 的 `summary.json` 来决定用 backtest 还是 live 的持仓文件，并支持 `t-1/today` 这种日期 token。

## 项目难点

把它难点拆开

1. 数据层永远不干净
   多数据源、多市场、不同 symbol 格式、字段命名不一致、缺失值、停牌处理。代码里专门有 column candidates 和 symbol 转换逻辑。

2. 股票池是按日期变的才像真的
   PIT universe、港股通过滤、流动性筛选，这些一做就会牵扯交易日历、上市天数、停牌策略等。

3. 避免无意中泄漏未来信息
   配置里放了很多避免未来信息的机制：按再平衡日期抽样、embargo/purge、next_rebalance horizon 等。

4. 评估不是一个指标能解决的
   IC/分位数、方向翻转阈值、walk-forward、permutation test 这些都是为了回答同一个问题：模型是不是在“撞运气”。  

5. 回测细节
   成本、buffer、退出机制、价格延迟/填充策略、再平衡间隔和标签周期的一致性检查。

6. 可复现和可操作性是工程难点，不是模型难点
   run 目录、summary/config.used、positions 输出、`holdings/snapshot` 读最新结果。
   数据回补/修订会导致同配置不同结果，复现依赖 cache + config.used + 代码版本固定。

7. 网格搜索是批量实验
   你的 grid 会为每个组合写临时 config，然后直接 `pipeline.run()`，再去磁盘找 `summary.json` 抄一行进 CSV。
   这也是你前面觉得“怎么这么慢”的根本原因之一。

## 没有 AI ，这大概要多少总工时？

先说结论：很难低于 200 小时，而且如果你还要边做边学、边调数据源、边修泄漏和回测细节，300–600 小时非常常见。

给你一个比较可信的拆分（按“有经验的 Python/量化工程师单人”粗估）：

| 模块                                | 典型工时（小时） | 为什么耗时                |
| --------------------------------- | -------: | -------------------- |
| 数据接入 + 缓存 + 字段清洗（多 provider、多市场）  |   40–120 | 最多坑，最难一次做对。          |
| 股票池（static + PIT/by-date + 港股通构建） |    20–60 | 交易日历/停牌/流动性筛选全在这。    |
| 标签 + 特征工程 + 标准化                   |    20–60 | 做出来不难，做得“不会泄漏/不会炸”难。 |
| 训练 + CV + 指标（IC/分位数等）             |    20–80 | 评估逻辑一复杂就容易出 bug。     |
| 回测（Top-K、成本、buffer、退出策略、持仓落盘）     |   40–120 | 细节一多，验证成本极高。         |
| walk-forward + permutation test   |    20–80 | “防自欺”越强，代码越复杂。       |
| CLI/配置系统/产物组织/文档                  |    20–80 | 让项目能被“重复使用”的工程部分。    |

把这些加起来，中位数大概 250–350 小时。
如果开发者对量化回测细节不熟，或者数据源/市场规则更复杂，乘个 1.5 倍很正常，也就是 350–600 小时。
