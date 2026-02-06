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
| `run` | 本次运行元数据（名称、时间戳、配置来源、输出目录） |
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
| `weight` | 目标权重（long-only 下通常等权） |
| `signal` | 该标的预测信号值 |
| `rank` | 当期截面排序名次 |
| `side` | `long` 或 `short` |

### `positions_current.csv` / `positions_current_live.csv`

只保留最新 `entry_date` 的那一组持仓，列结构与 `positions_by_rebalance` 一致。

### `positions_by_rebalance_oos.csv` / `positions_current_oos.csv`

仅在启用 `eval.final_oos` 且成功评估时输出，字段与主文件一致。

## 调仓差异文件

`rebalance_diff.csv`（以及 `_live` / `_oos` 版本）展示“最新一期 vs 上一期”的变化：

| 列名 | 说明 |
| --- | --- |
| `entry_date` / `entry_date_prev` | 当前与上一期入场日 |
| `ts_code` / `side` | 标的与方向 |
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

## 其他常用文件

1. `config.used.yml`：本次运行实际生效配置（复现实验首选）。
1. `eval_scored.parquet`：评估样本打分明细（启用 artifact 时）。
1. `ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`：指标时序数据。
1. `feature_importance.csv`：模型特征重要性。

