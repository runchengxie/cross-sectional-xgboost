# 输出产物与字段约定

本页说明 run 目录中的关键文件与字段约定，便于写自动化消费脚本（风控、报表、下游执行等）。

## 产物目录

默认每次运行会写到：

`out/runs/<run_name>_<timestamp>_<config_hash>/`

`live` 推荐单独目录，例如 `out/live_runs/...`。

## `summary.json` 顶层结构

`summary.json` 顶层字段（固定键）：

| 顶层键 | 说明 |
| --- | --- |
| `run` | 本次运行元数据（名称、时间戳、配置来源、模型类型、输出目录） |
| `data` | 市场、数据源、日期区间、样本规模 |
| `dataset` | `dataset.parquet` 的 schema/行数/索引信息 |
| `universe` | 股票池模式、PIT 文件与停牌处理策略 |
| `label` | 标签窗口、`shift_days`、标签模式 |
| `split` | 训练/测试日期与 purge/embargo 信息 |
| `eval` | IC、分位数、换手、错误指标、方向判定、滚动指标 |
| `backtest` | 回测参数、绩效统计、基准/主动收益与滚动 Sharpe |
| `final_oos` | 最终留出期（启用时）对应评估与回测摘要 |
| `positions` | 回测持仓文件路径与窗口字段声明 |
| `live` | live 模式状态、as_of 与 live 持仓文件路径 |
| `fundamentals` | 基本面数据源与字段配置摘要 |
| `walk_forward` | 滚动窗口验证参数与结果 |

说明：

1. 这些键固定存在，但部分值会是 `null`/空对象（例如未启用 `final_oos`、未启用 `live`）。
1. 消费脚本建议优先读 `summary.json` 里保存的文件路径，不要硬编码文件名。

## 稳定性契约（给下游脚本）

稳定 contract（版本演进时尽量保持不变）：

1. `summary.json` 顶层固定键集合（`run/data/dataset/universe/label/split/eval/backtest/final_oos/positions/live/fundamentals/walk_forward`）。
1. 持仓主键列语义：`trade_date`、`entry_date`、`ts_code`、`stock_ticker`、`weight`、`signal`、`rank`、`side`。
1. `summary.json` 内记录的文件路径优先级高于固定文件名推断。

best-effort（可能为空、缺失或未产出文件）：

1. `final_oos` / `live` / `walk_forward` 子结构（取决于对应功能是否启用）。
1. `dataset.parquet`、`eval_scored.parquet`、`backtest_*.csv` 等产物（取决于配置与数据可用性）。
1. 任何依赖外部数据源补数/修订得到的统计值（同配置在不同日期可能变化）。

## 持仓文件

### `positions_by_rebalance.csv` / `positions_by_rebalance_live.csv`

每个调仓期的目标持仓明细，核心列：

| 列名 | 说明 |
| --- | --- |
| `rebalance_date` | 信号计算日（`YYYYMMDD`） |
| `signal_asof` | 同 `rebalance_date`，用于快照展示 |
| `entry_date` | 实际入场日（考虑 `shift_days`） |
| `next_entry_date` | 下一次入场日（最后一期为空） |
| `holding_window` | `entry_date -> next_entry_date`（最后一期为 `entry_date`） |
| `ts_code` | 标的代码（内部标准格式） |
| `stock_ticker` | 标的代码（外部通用别名，等价于 `ts_code`） |
| `weight` | 目标权重（long-only 下通常等权） |
| `signal` | 该标的预测信号值 |
| `rank` | 当期截面排序名次 |
| `side` | `long` 或 `short` |

### `positions_current.csv` / `positions_current_live.csv`

只保留最新 `entry_date` 的那一组持仓，列结构与 `positions_by_rebalance` 一致。

兼容说明：

1. 项目内部仍以 `ts_code` 作为主字段。
1. 对外消费（CLI JSON/CSV）可使用 `stock_ticker`，其值与 `ts_code` 一致。

### `positions_by_rebalance_oos.csv` / `positions_current_oos.csv`

仅在启用 `eval.final_oos` 且成功评估时输出，字段与主文件一致。

## 调仓差异文件

`rebalance_diff.csv`（以及 `_live` / `_oos` 版本）展示“最新一期 vs 上一期”的变化：

| 列名 | 说明 |
| --- | --- |
| `entry_date` / `entry_date_prev` | 当前与上一期入场日 |
| `ts_code` / `side` | 标的与方向 |
| `stock_ticker` | 标的代码外部别名（等价于 `ts_code`） |
| `weight` / `weight_prev` | 当前与上一期权重（缺失补 0） |
| `signal` / `signal_prev` | 当前与上一期信号 |
| `rank` / `rank_prev` | 当前与上一期 rank |
| `weight_delta` | `weight - weight_prev` |
| `change` | `added` / `removed` / `changed` |

## 数据集文件

### `dataset.parquet`（可选）

仅在同时满足以下条件时输出：

1. `eval.save_artifacts=true`
1. `eval.save_dataset=true`

格式约定：

1. Parquet 以 `(trade_date, ts_code)` 为 MultiIndex。
1. 列顺序为：`price_col` + `features` + `label` + `is_tradable`（若存在）。
1. 对应 schema 会写入 `summary.json -> dataset.schema`。

## 研究工具输出契约

下面三类文件不在单个 run 目录内，但属于研究流程中的核心对比产物。

### `csml summarize`：`runs_summary.csv`

默认位置：

`<first-runs-dir>/runs_summary.csv`（可用 `--output` 覆盖）。

来源：

1. 递归扫描 `--runs-dir` 下的 `summary.json`。
1. 对应读取同目录 `config.used.yml`。
1. 生成 `flag_*`、`score`、`dsr` 列用于筛选/排序。

列契约（当前稳定列顺序）：

```text
source_runs_dir,run_dir,run_name,run_timestamp,config_hash,summary_path,config_path,market,data_provider,data_start_date,data_end_date,data_end_date_config,data_rows,data_rows_model,data_rows_model_in_sample,data_rows_model_oos,data_dropped_dates,universe_mode,label_horizon_days,label_shift_days,eval_top_k,backtest_top_k,transaction_cost_bps,eval_rebalance_frequency,backtest_rebalance_frequency,eval_buffer_exit,eval_buffer_entry,backtest_buffer_exit,backtest_buffer_entry,eval_ic_mean,eval_ic_ir,eval_long_short,eval_turnover_mean,backtest_periods,backtest_periods_per_year,backtest_total_return,backtest_ann_return,backtest_ann_vol,backtest_sharpe,backtest_skew,backtest_kurtosis_excess,backtest_max_drawdown,backtest_avg_turnover,backtest_avg_cost_drag,dsr,dsr_sr0,dsr_n_trials,dsr_var_trials,flag_short_sample,flag_negative_long_short,flag_high_turnover,flag_relative_end_date,score,status,error
```

`score` 计算规则：

```text
score = backtest_sharpe
      - score_drawdown_weight * abs(backtest_max_drawdown)
      - score_cost_weight * backtest_avg_cost_drag
```

默认权重（可由 CLI 覆盖）：

1. `score_drawdown_weight = 0.5`
1. `score_cost_weight = 10.0`

补充：

1. 若 `backtest_sharpe` 缺失，则 `score` 为空。
1. 若 `backtest_max_drawdown` 或 `backtest_avg_cost_drag` 缺失，会按 0 处理惩罚项。
1. `dsr` 为 Deflated Sharpe Ratio（0-1），在 summarize 阶段按可比策略分组计算；`dsr_sr0` 为组内多重比较修正后的 Sharpe 阈值（原频率）。
1. `dsr_n_trials` 使用分组内尝试次数（attempts count）；`dsr_var_trials` 为分组内原频率 Sharpe 的样本方差（`ddof=1`）。

### `csml grid`：`grid_summary.csv`

默认位置：

`out/runs/grid_summary.csv`（可用 `--output` 覆盖）。

来源：

1. 先执行一次 base pipeline（产出 `eval_scored.parquet`）。
1. 在同一份 scored 数据上循环 `top_k × cost_bps × buffer_exit × buffer_entry`。
1. 每行对应一个参数组合，不会为每个格点重训模型。

列契约（当前稳定列顺序）：

```text
run_name,top_k,cost_bps,buffer_exit,buffer_entry,summary_path,output_dir,label_horizon_days,eval_ic_mean,eval_ic_ir,eval_long_short,eval_turnover_mean,backtest_periods,backtest_total_return,backtest_ann_return,backtest_ann_vol,backtest_sharpe,backtest_max_drawdown,backtest_avg_turnover,backtest_avg_cost_drag,status,error
```

### `csml sweep-linear`：`out/sweeps/<tag>/`

目录结构：

```text
out/sweeps/<tag>/
  configs/
    ridge_*.yml
    elasticnet_*.yml
  jobs.csv
  run_results.csv
  runs_summary.csv   # 默认会自动 summarize，除非 --skip-summarize
```

其中：

1. `jobs.csv` 列契约：`order,model,alpha,l1_ratio,run_name,config_path`
1. `run_results.csv` 列契约：`order,run_name,config_path,status,error`
1. `runs_summary.csv` 列契约与 `csml summarize` 章节一致。

## 其他常用文件

1. `config.used.yml`：本次运行实际生效配置（复现实验首选）。
1. `eval_scored.parquet`：评估样本打分明细（启用 artifact 时）。
1. `ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`：指标时序数据。
1. `feature_importance.csv`：模型特征重要性。
1. `walk_forward_feature_importance.csv`：walk-forward 每个窗口的特征重要性明细。
1. `walk_forward_feature_stability.csv`：跨窗口稳定性统计（命中率/均值/方差等）。
