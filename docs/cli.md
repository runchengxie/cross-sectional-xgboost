# CLI 参数大全

本页汇总 `csml` 所有子命令及可传参数，便于直接查阅。

## 查看帮助

```bash
csml --help
csml <subcommand> --help
```

## 1) `csml run`

用途：运行主流程（训练/评估/回测）。

参数：

* `--config <path_or_alias>`：配置路径或内置别名（`default/cn/hk/us`）。

示例：

```bash
csml run --config config/hk.yml
csml run --config hk
```

## 2) `csml grid`

用途：Top-K × 交易成本 × buffer（`buffer_exit/buffer_entry`）敏感性网格，并输出 `grid_summary.csv`。该命令先跑一次 base run，再复用其 `eval_scored.parquet` 循环参数组合，不会为每个网格点重训模型。

参数：

* `--config <path_or_alias>`：基础配置（默认 `config/hk.yml`）。
* `--top-k <values>`：可重复传，支持逗号分隔（默认 `5,10,20`）。
* `--cost-bps <values>`：可重复传，支持逗号分隔（默认 `15,25,40`）。
* `--buffer-exit <values>`：可重复传，支持逗号分隔（默认取配置里的 `buffer_exit`）。
* `--buffer-entry <values>`：可重复传，支持逗号分隔（默认取配置里的 `buffer_entry`）。
* `--output <csv_path>`：输出 CSV（默认 `out/runs/grid_summary.csv`）。
* `--run-name-prefix <prefix>`：run_name 前缀。
* `--log-level <level>`：日志级别（`CRITICAL/ERROR/WARNING/INFO/DEBUG`）。

示例：

```bash
csml grid --config config/hk.yml --top-k 5,10 --cost-bps 15,25

# 同时扫 buffer（交易层三件套：Top-K × 成本 × buffer）
csml grid \
  --config config/hk_selected__baseline.yml \
  --top-k 10,20 \
  --cost-bps 15,25,40 \
  --buffer-exit 8,10 \
  --buffer-entry 4,5
```

## 3) `csml sweep-linear`

用途：生成 Ridge/ElasticNet 网格配置，批量执行 `run`，并自动 `summarize` 导出对比表。

参数：

* `--sweep-config <path>`：sweep 参数 YAML（CLI 参数会覆盖 YAML）。
* `--config <path_or_alias>`：基础配置（默认 `config/hk_selected__baseline.yml`）。
* 兼容迁移：若仍配置 `config/hk_selected.yml` 且文件不存在，会自动回退到 `config/hk_selected__baseline.yml` 并提示 warning。
* `--run-name-prefix <prefix>`：批量 run_name 前缀（默认 `hk_sel_`）。
* `--sweeps-dir <dir>`：sweep 产物根目录（默认 `out/sweeps`）。
* `--tag <name>`：本次实验标签（默认当前时间戳）。
* `--runs-dir <dir>`：覆盖生成 config 的 `eval.output_dir`。
* `--ridge-alpha <values>`：可重复传，支持逗号分隔（默认 `0.01,0.1,1,10,100`）。
* `--elasticnet-alpha <values>`：可重复传，支持逗号分隔（默认 `0.01,0.1,1`）。
* `--elasticnet-l1-ratio <values>`：可重复传，支持逗号分隔（默认 `0.1,0.5,0.9`）。
* `--skip-ridge` / `--skip-elasticnet`：跳过某一类模型。
* `--dry-run`：仅生成 configs/jobs 清单，不执行 run。
* `--continue-on-error`：单个组合失败后继续跑后续组合。
* `--skip-summarize`：跳过自动汇总。
* `--summary-output <csv_path>`：汇总 CSV 输出路径（默认 `<sweep-dir>/runs_summary.csv`）。
* `--log-level <level>`：日志级别（`CRITICAL/ERROR/WARNING/INFO/DEBUG`）。

示例：

```bash
csml sweep-linear --sweep-config config/sweeps/hk_selected__linear_a.yml

# 临时覆盖 tag 和 dry-run
csml sweep-linear \
  --sweep-config config/sweeps/hk_selected__linear_a.yml \
  --tag hk_linear_a_debug \
  --dry-run
```

## 4) `csml summarize`

用途：跨历史 run 聚合 `summary.json` + `config.used.yml`，输出总表。

参数：

* `--runs-dir <dir>`：扫描目录，可重复传（默认 `out/runs`）。
* `--output <csv_path>`：输出路径（默认 `<first-runs-dir>/runs_summary.csv`）。
* `--run-name-prefix <prefix>`：仅保留 run_name 以该前缀开头的 run，可重复传，支持逗号分隔。
* `--since <datetime>`：仅保留该时点及之后的 run。
  * 支持：`YYYYMMDD`、`YYYY-MM-DD`、`YYYYMMDD_HHMMSS`、`YYYY-MM-DDTHH:MM:SS`。
  * 也支持：`today`、`now`、`yesterday`、`t-1`。
* `--latest-n <int>`：过滤后仅保留最新 N 条（必须 > 0）。
* `--short-sample-periods <int>`：`flag_short_sample` 阈值（默认 `24`）。
* `--high-turnover-threshold <float>`：`flag_high_turnover` 阈值（默认 `0.7`）。
* `--score-drawdown-weight <float>`：`score` 中回撤惩罚权重（默认 `0.5`）。
* `--score-cost-weight <float>`：`score` 中成本惩罚权重（默认 `10.0`）。
* `--exclude-flag-short-sample`：过滤掉 `flag_short_sample=true`。
* `--exclude-flag-high-turnover`：过滤掉 `flag_high_turnover=true`。
* `--exclude-flag-negative-long-short`：过滤掉 `flag_negative_long_short=true`。
* `--exclude-flag-relative-end-date`：过滤掉配置里 `data.end_date` 仍是相对日期 token（如 `today/t-1`）的 run。
* `--sort-by <timestamp|score|dsr>`：按时间、`score` 或 `dsr` 排序（默认 `timestamp`）。
* `--log-level <level>`：日志级别（`CRITICAL/ERROR/WARNING/INFO/DEBUG`）。

`score` 计算方式（用于 `--sort-by score`）：

```text
score = backtest_sharpe
      - score_drawdown_weight * abs(backtest_max_drawdown)
      - score_cost_weight * backtest_avg_cost_drag
```

默认权重：

* `score_drawdown_weight = 0.5`
* `score_cost_weight = 10.0`

补充：

* 若 `backtest_sharpe` 缺失，`score` 为空。
* 若 `backtest_max_drawdown` 或 `backtest_avg_cost_drag` 缺失，惩罚项按 0 处理。
* `dsr` 在 summarize 阶段按可比策略分组计算（`market` + `label_horizon_days` + `backtest_rebalance_frequency` + `transaction_cost_bps` + `backtest_top_k`），`N` 使用该组尝试次数。
* `dsr` 的输入 Sharpe 会从年化值换回原频率（`sr = sr_ann / sqrt(periods_per_year)`），组内方差也在原频率下计算。

示例：

```bash
# 全量汇总
csml summarize --runs-dir out/runs --output out/runs/runs_summary.csv

# 仅看最近一次 grid 相关结果
csml summarize --runs-dir out/runs --run-name-prefix hk_grid --latest-n 1

# 仅看 2026-02-01 之后的数据
csml summarize --runs-dir out/runs --since 2026-02-01

# 先过滤短样本/高换手，再按 score 看候选
csml summarize \
  --runs-dir out/runs \
  --exclude-flag-short-sample \
  --exclude-flag-high-turnover \
  --exclude-flag-relative-end-date \
  --sort-by score
```

## 5) `csml holdings`

用途：输出最近一次 run 的当前持仓。

参数：

* `--config <path_or_alias>`：用于定位 run。
* `--run-dir <dir>`：直接指定 run 目录（优先于 `--config`）。
* `--top-k <int>`：可选 Top-K 过滤。
* `--as-of <date_or_token>`：持仓时点（默认 `t-1`）。
  * 支持：`YYYYMMDD`、`YYYY-MM-DD`、`today`、`t-1`、`last_trading_day`、`last_completed_trading_day`。
  * 说明：当能识别到 `provider=rqdata` 且有 `market` 上下文（来自 run `summary.json` 或 `--config`）时，`last_trading_day` 两个 token 会按交易日解析；否则回退自然日并输出 warning。
* `--source <mode>`：`auto/backtest/live`（默认 `auto`）。
* `--format <fmt>`：`text/csv/json`（默认 `text`）。
* `--out <path>`：输出到文件；不传则 stdout。

字段兼容：

1. 读取到的持仓文件中，标的列支持 `ts_code` 或 `stock_ticker`（推荐 `stock_ticker`）。

示例：

```bash
csml holdings --config config/hk.yml --as-of t-1
csml holdings --run-dir out/runs/<run_dir> --format csv --out out/positions/latest.csv
```

## 6) `csml snapshot`

> snapshot 在效果上等价于先后运行 run 和 holdings 两个命令，价值是流程封装 + 降低出错率，虽然看起来多余，但是设计思路包括：
>
> * 一条命令保证先产出再读取：不会出现忘了先跑 run，或者跑了别的 config，然后 holdings 读到旧结果的意外。
> * 更适合脚本/定时任务：对 crontab、CI、Airflow 这种自动化逻辑更友好，一次命令完成一个完整动作。
> * 支持 --skip-run / --run-dir：有利于指定对某个特定 run 目录出快照，而不依赖项目的 auto 默认最近一次run。

用途：live 快照（默认先 run，再输出 live holdings）。

参数：

* `--config <path_or_alias>`：配置路径。
* `--run-dir <dir>`：直接使用已有 run。
* `--as-of <date_or_token>`：默认 `t-1`。
  * 支持：`YYYYMMDD`、`YYYY-MM-DD`、`today`、`t-1`、`last_trading_day`、`last_completed_trading_day`（行为同 `holdings`）。
* `--skip-run`：跳过 run，只读取已有结果。
* `--top-k <int>`：可选 Top-K 过滤。
* `--format <fmt>`：`text/csv/json`（默认 `text`）。
* `--out <path>`：输出文件路径。

示例：

```bash
csml snapshot --config config/hk_live.yml
csml snapshot --config config/hk_live.yml --skip-run --format json
```

## 7) `csml alloc`

用途：按最新持仓做 Top-N 等权资金分配，自动换算为每只股票买多少手/股（价格和 round lot 来自 RQData）。

参数：

* `--config <path_or_alias>`：用于定位最新 run（可选）。
* `--run-dir <dir>`：直接指定 run 目录（优先于 `--config`）。
* `--positions-file <csv>`：直接指定持仓文件（优先于 run 定位）。
* `--top-k <int>`：可选 Top-K 过滤（用于定位 run）。
* `--as-of <date_or_token>`：持仓时点（默认 `t-1`）。
  * 支持：`YYYYMMDD`、`YYYY-MM-DD`、`today`、`t-1`、`last_trading_day`、`last_completed_trading_day`（行为同 `holdings`）。
* `--source <mode>`：`auto/backtest/live`（默认 `auto`）。
* `--side <mode>`：`long/short/all`（默认 `long`）。
* `--top-n <int>`：从排序后的持仓中取前 N 名做等权（默认 `20`）。
* `--cash <float>`：总资金（默认 `1000000`）。
* `--buffer-bps <float>`：预留现金（bps，默认 `0`）。
* `--price-field <name>`：RQData 价格字段（默认 `close`）。
* `--price-lookback-days <int>`：回看价格窗口天数（默认 `20`）。
* `--username/--password`：可选覆盖 RQData 账号。
* `--format <fmt>`：`text/csv/json`（默认 `text`）。
* `--out <path>`：输出到文件；不传则 stdout。

字段兼容：

1. `--positions-file` 中标的列支持 `ts_code` 或 `stock_ticker`（推荐 `stock_ticker`）。

示例：

```bash
# 最新 run 的前20等权，资金100万
csml alloc --config config/hk_live.yml --source live --top-n 20 --cash 1000000

# 明确指定 run，做前10等权并导出 JSON
csml alloc --run-dir out/runs/<run_dir> --source live --top-n 10 --format json --out out/alloc/top10.json

# 直接指定 positions 文件，做前5等权
csml alloc --positions-file out/runs/<run_dir>/positions_by_rebalance_live.csv --top-n 5
```

## 8) `csml rqdata info`

用途：初始化并显示 RQData 登录信息。

参数：

* `--config <path_or_alias>`：可选配置（读取 `data.rqdata.init`）。
* `--username <name>`：覆盖用户名。
* `--password <password>`：覆盖密码。

## 9) `csml rqdata quota`

用途：查询 RQData 配额。

参数：

* `--config <path_or_alias>`：可选配置。
* `--username <name>`：覆盖用户名。
* `--password <password>`：覆盖密码。
* `--pretty`：人类可读格式输出。

## 10) `csml tushare verify-token`

用途：验证 TuShare token。

参数：

* CLI 会将后续参数原样转发到底层脚本。
* 推荐用 `csml tushare verify-token` 直接执行。

## 11) `csml universe index-components`

用途：拉取指数成分并输出 symbols 文件。

参数：

* CLI 会将后续参数原样转发到底层脚本（`fetch_index_components.py`）。
* 使用 `--by-date-out` 时，PIT CSV 会输出 `trade_date`、`ts_code`、`stock_ticker` 三列。

常见示例：

```bash
csml universe index-components --index-code 000300.SH --month 202501
```

## 12) `csml universe hk-connect`

用途：构建港股通 PIT universe。

参数：

* `--config <path_or_alias>`：可选配置路径。
* 其余参数原样转发到底层脚本（`build_hk_connect_universe.py`）。
* 产出的 by-date CSV 会同时包含 `ts_code` 与 `stock_ticker`。

常见示例：

```bash
csml universe hk-connect --config config/universe.hk_connect.yml --mode daily
```

## 13) `csml init-config`

用途：导出内置配置模板。

参数：

* `--market <name>`：模板名（`default/cn/hk/us`，默认 `default`）。
* `--out <path_or_dir>`：输出路径或目录。
* `--force`：允许覆盖已有文件。

示例：

```bash
csml init-config --market hk --out config/
```
