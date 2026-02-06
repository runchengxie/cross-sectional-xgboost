# 配置参考

内置模板位于 `src/csxgb/config/*.yml`，导出后的配置默认放在 `config/`。`--config` 支持内置别名（`default/cn/hk/us`）或文件路径。

模板导出示例：

```bash
csxgb init-config --market hk --out config/
```

补充文档：

* 输出字段与产物说明：`docs/outputs.md`
* 数据源差异与缓存行为：`docs/providers.md`

## 关键参数

* `universe`：股票池、过滤条件、最小截面规模（支持 `by_date_file` 动态池；可用 `mode/require_by_date/suspended_policy` 明确 PIT 与停牌处理）
* `market`：`cn` / `hk` / `us`
* `data`：`provider`、`rqdata` / `eodhd` 或 `daily_endpoint` / `basic_endpoint` / `column_map`（字段映射为 `trade_date/ts_code/close/vol/amount`）、`cache_tag`、`retry`
* `fundamentals`：Level 0 基本面数据合并（`features`/`column_map`/`ffill`/`log_market_cap`/`required`）
* `label`：预测窗口、shift、winsorize（支持 `horizon_mode=next_rebalance`）
* `features`：特征清单与窗口
* `model`：XGBoost 参数，`sample_weight_mode`（`none`/`date_equal`）
* `eval`：切分、分位数、换手成本、embargo/purge、`signal_direction_mode`、`min_abs_ic_to_flip`、`sample_on_rebalance_dates`，以及可选的 `report_train_ic`、`save_artifacts`、`save_dataset`、`permutation_test`、`walk_forward`、`final_oos`，还可配置 `rolling`（滚动 IC/Sharpe 窗口）与 `bucket_ic`（分桶 IC）
* `backtest`：再平衡频率、Top-K、成本、`long_only/short_k`、基准、`exit_mode`、`exit_price_policy` 与 `buffer_exit/buffer_entry`，可选 `execution`（cost_model / exit_policy）
* `live`：可选“当下持仓快照”，用于在固定回测之外输出当前组合

## 数据与缓存（data）

常用键：

* `start_date` / `start_years`：若同时配置，`start_date` 优先生效；`start_years` 会从 `end_date` 往前回推。
* `end_date`：支持 `today` / `t-1` / `last_trading_day` / `last_completed_trading_day` / `YYYYMMDD`。
* `price_col`：价格列名（用于标签与回测）。
* `cache_dir`：缓存目录。
* `cache_tag` / `cache_version`：缓存命名空间（同一数据源下隔离不同版本）。
* `cache_mode` / `daily_cache_mode`：`symbol`（每个 symbol 一个缓存）或 `range/window`（按时间区间缓存）。
* `cache_refresh_days`：命中缓存时刷新最近 N 天的范围（仅 `symbol` 模式有意义）。
* `cache_refresh_on_hit`：命中缓存时是否也触发刷新。

TuShare 相关可覆盖项（按需配置）：

* `daily_endpoint` / `basic_endpoint`：覆盖默认接口。
* `daily_fields` / `basic_fields`：覆盖字段列表。
* `daily_params` / `basic_params`：额外参数（与上面字段合并）。
* `daily_symbol_param` / `daily_start_param` / `daily_end_param`：覆盖接口参数名。

## Walk-forward 细节

* `eval.walk_forward.backtest_enabled`：为 walk-forward 的每个窗口是否跑回测（对 live 配置常设为 false）。
* `eval.walk_forward.permutation_test`：可在每个窗口单独做置换检验。

## 基本面数据

* 默认 `fundamentals.enabled=true`（CN/Default 走 TuShare `daily_basic`，HK/US 默认走本地文件）；如无数据可先设为 `false`。
* `fundamentals.source=provider` 走数据源接口（目前仅支持 TuShare）；`source=file` 则读取本地 CSV/Parquet。缺文件会警告并跳过（可用 `fundamentals.required=true` 强制报错）。
* 使用 `fundamentals.column_map` 映射字段，再通过 `ffill` 做按股票时间向前填充。

## Live 模式

`live` 用于在同一套配置下生成“当前持仓快照”。建议搭配单独的 live 配置文件与输出目录，避免和回测产物混在一起。

```yaml
data:
  end_date: "t-1"   # 支持 today / t-1 / YYYYMMDD / last_trading_day

eval:
  output_dir: "out/live_runs"
  save_artifacts: true

backtest:
  enabled: false

live:
  enabled: true
  as_of: "t-1"
  train_mode: "full"   # full=用全部可用标签训练; train=复用回测训练集模型
```

说明：

* `last_trading_day` / `last_completed_trading_day` 需要交易日历支持（`provider=rqdata`），否则会退回到自然日并给出警告。
* Live 产物固定写入 `positions_by_rebalance_live.csv` 与 `positions_current_live.csv`（live-only 不再生成普通文件）；持仓文件会包含 `signal_asof/next_entry_date/holding_window` 辅助字段。
* `csxgb holdings --source live` 会优先读取 summary 中的 live 文件路径。
* 一键快照：`csxgb snapshot --config config/hk_live.yml`（内部先 run 再输出 holdings），可用 `--skip-run` / `--run-dir` 只读已有结果。

## 最终 OOS 留出期

当需要在 walk-forward/CV 之外保留一段“最终验收期”，可使用 `eval.final_oos`。该留出期不会参与任何训练/调参，仅用于最后评估。

```yaml
eval:
  final_oos:
    enabled: true
    size: 0.1   # 支持比例(0-1)或绝对日期数量
```

## Dataset 输出（可选）

设置 `eval.save_dataset=true`（需同时 `eval.save_artifacts=true`）会在 run_dir 额外写出 `dataset.parquet`。Schema 固定为 `(trade_date, ts_code)` 索引，对应列为 `price_col` + `features` + `label` + `is_tradable`（如存在），便于后续接 Qlib 或其他框架。

## 执行假设模块（可选）

`backtest.execution` 可覆盖成本与退出规则，未配置时仍使用 `transaction_cost_bps` 与 `exit_*` 旧键：

```yaml
backtest:
  execution:
    cost_model:
      name: bps
      bps: 15
      round_trip: true
    exit_policy:
      price: delay
      fallback: ffill
```

## 补充指标配置（可选）

```yaml
eval:
  rolling:
    enabled: true
    windows_months: [6, 12]   # 6M/12M 滚动 IC 与 Sharpe
  bucket_ic:
    enabled: true
    method: spearman          # spearman / pearson
    min_count: 0              # 分桶样本不足时可跳过
    schemes:
      - name: industry
        column: industry_code
        type: category
      - name: market_cap
        column: log_mcap
        type: quantile
        n_bins: 3
      - name: liquidity
        column: amount
        type: quantile
        n_bins: 3
```
