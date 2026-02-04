# 配置参考

内置模板位于 `src/csxgb/config/*.yml`，导出后的配置默认放在 `config/`。`--config` 支持内置别名（`default/cn/hk/us`）或文件路径。

模板导出示例：

```bash
csxgb init-config --market hk --out config/
```

## 关键参数

* `universe`：股票池、过滤条件、最小截面规模（支持 `by_date_file` 动态池；可用 `mode/require_by_date/suspended_policy` 明确 PIT 与停牌处理）
* `market`：`cn` / `hk` / `us`
* `data`：`provider`、`rqdata` / `eodhd` 或 `daily_endpoint` / `basic_endpoint` / `column_map`（字段映射为 `trade_date/ts_code/close/vol/amount`）、`cache_tag`、`retry`
* `fundamentals`：Level 0 基本面数据合并（`features`/`column_map`/`ffill`/`log_market_cap`/`required`）
* `label`：预测窗口、shift、winsorize（支持 `horizon_mode=next_rebalance`）
* `features`：特征清单与窗口
* `model`：XGBoost 参数，`sample_weight_mode`（`none`/`date_equal`）
* `eval`：切分、分位数、换手成本、embargo/purge、`signal_direction_mode`、`min_abs_ic_to_flip`、`sample_on_rebalance_dates`，以及可选的 `report_train_ic`、`save_artifacts`、`permutation_test` 与 `walk_forward`
* `backtest`：再平衡频率、Top-K、成本、`long_only/short_k`、基准、`exit_mode`、`exit_price_policy` 与 `buffer_exit/buffer_entry`

## 基本面数据

* 默认 `fundamentals.enabled=true`（CN/Default 走 TuShare `daily_basic`，HK/US 默认走本地文件）；如无数据可先设为 `false`。
* `fundamentals.source=provider` 走数据源接口（目前仅支持 TuShare）；`source=file` 则读取本地 CSV/Parquet。缺文件会警告并跳过（可用 `fundamentals.required=true` 强制报错）。
* 使用 `fundamentals.column_map` 映射字段，再通过 `ffill` 做按股票时间向前填充。
