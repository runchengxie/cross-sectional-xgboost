# CLI 参数大全

本页汇总 `csxgb` 所有子命令及可传参数，便于直接查阅。

## 查看帮助

```bash
csxgb --help
csxgb <subcommand> --help
```

## 1) `csxgb run`

用途：运行主流程（训练/评估/回测）。

参数：

* `--config <path_or_alias>`：配置路径或内置别名（`default/cn/hk/us`）。

示例：

```bash
csxgb run --config config/hk.yml
csxgb run --config hk
```

## 2) `csxgb grid`

用途：Top-K × 交易成本敏感性网格，并输出 `grid_summary.csv`。

参数：

* `--config <path_or_alias>`：基础配置（默认 `config/hk.yml`）。
* `--top-k <values>`：可重复传，支持逗号分隔（默认 `5,10,20`）。
* `--cost-bps <values>`：可重复传，支持逗号分隔（默认 `15,25,40`）。
* `--output <csv_path>`：输出 CSV（默认 `out/runs/grid_summary.csv`）。
* `--run-name-prefix <prefix>`：run_name 前缀。
* `--log-level <level>`：日志级别（`CRITICAL/ERROR/WARNING/INFO/DEBUG`）。

示例：

```bash
csxgb grid --config config/hk.yml --top-k 5,10 --cost-bps 15,25
```

## 3) `csxgb summarize`

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
* `--log-level <level>`：日志级别（`CRITICAL/ERROR/WARNING/INFO/DEBUG`）。

示例：

```bash
# 全量汇总
csxgb summarize --runs-dir out/runs --output out/runs/runs_summary.csv

# 仅看最近一次 grid 相关结果
csxgb summarize --runs-dir out/runs --run-name-prefix hk_grid --latest-n 1

# 仅看 2026-02-01 之后的数据
csxgb summarize --runs-dir out/runs --since 2026-02-01
```

## 4) `csxgb holdings`

用途：输出最近一次 run 的当前持仓。

参数：

* `--config <path_or_alias>`：用于定位 run。
* `--run-dir <dir>`：直接指定 run 目录（优先于 `--config`）。
* `--top-k <int>`：可选 Top-K 过滤。
* `--as-of <date_or_token>`：持仓时点（默认 `t-1`）。
  * 支持：`YYYYMMDD`、`YYYY-MM-DD`、`today`、`t-1`。
* `--source <mode>`：`auto/backtest/live`（默认 `auto`）。
* `--format <fmt>`：`text/csv/json`（默认 `text`）。
* `--out <path>`：输出到文件；不传则 stdout。

示例：

```bash
csxgb holdings --config config/hk.yml --as-of t-1
csxgb holdings --run-dir out/runs/<run_dir> --format csv --out out/positions/latest.csv
```

## 5) `csxgb snapshot`

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
* `--skip-run`：跳过 run，只读取已有结果。
* `--top-k <int>`：可选 Top-K 过滤。
* `--format <fmt>`：`text/csv/json`（默认 `text`）。
* `--out <path>`：输出文件路径。

示例：

```bash
csxgb snapshot --config config/hk_live.yml
csxgb snapshot --config config/hk_live.yml --skip-run --format json
```

## 6) `csxgb alloc`

用途：按最新持仓做 Top-N 等权资金分配，自动换算为每只股票买多少手/股（价格和 round lot 来自 RQData）。

参数：

* `--config <path_or_alias>`：用于定位最新 run（可选）。
* `--run-dir <dir>`：直接指定 run 目录（优先于 `--config`）。
* `--positions-file <csv>`：直接指定持仓文件（优先于 run 定位）。
* `--top-k <int>`：可选 Top-K 过滤（用于定位 run）。
* `--as-of <date_or_token>`：持仓时点（默认 `t-1`）。
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

示例：

```bash
# 最新 run 的前20等权，资金100万
csxgb alloc --config config/hk_live.yml --source live --top-n 20 --cash 1000000

# 明确指定 run，做前10等权并导出 JSON
csxgb alloc --run-dir out/runs/<run_dir> --source live --top-n 10 --format json --out out/alloc/top10.json

# 直接指定 positions 文件，做前5等权
csxgb alloc --positions-file out/runs/<run_dir>/positions_by_rebalance_live.csv --top-n 5
```

## 7) `csxgb rqdata info`

用途：初始化并显示 RQData 登录信息。

参数：

* `--config <path_or_alias>`：可选配置（读取 `data.rqdata.init`）。
* `--username <name>`：覆盖用户名。
* `--password <password>`：覆盖密码。

## 8) `csxgb rqdata quota`

用途：查询 RQData 配额。

参数：

* `--config <path_or_alias>`：可选配置。
* `--username <name>`：覆盖用户名。
* `--password <password>`：覆盖密码。
* `--pretty`：人类可读格式输出。

## 9) `csxgb tushare verify-token`

用途：验证 TuShare token。

参数：

* CLI 会将后续参数原样转发到底层脚本。
* 推荐用 `csxgb tushare verify-token` 直接执行。

## 10) `csxgb universe index-components`

用途：拉取指数成分并输出 symbols 文件。

参数：

* CLI 会将后续参数原样转发到底层脚本（`fetch_index_components.py`）。

常见示例：

```bash
csxgb universe index-components --index-code 000300.SH --month 202501
```

## 11) `csxgb universe hk-connect`

用途：构建港股通 PIT universe。

参数：

* `--config <path_or_alias>`：可选配置路径。
* 其余参数原样转发到底层脚本（`build_hk_connect_universe.py`）。

常见示例：

```bash
csxgb universe hk-connect --config config/universe.hk_connect.yml --mode daily
```

## 12) `csxgb init-config`

用途：导出内置配置模板。

参数：

* `--market <name>`：模板名（`default/cn/hk/us`，默认 `default`）。
* `--out <path_or_dir>`：输出路径或目录。
* `--force`：允许覆盖已有文件。

示例：

```bash
csxgb init-config --market hk --out config/
```
