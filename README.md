# cross-sectional-xgboost

使用 TuShare / RQData / EODHD 日线数据与 XGBoost 回归进行截面因子研究与评估（支持 A/HK/US 多市场配置切换）。流程包含特征工程、时间序列切分、IC 评估、分位数组合收益、换手率估计与特征重要性输出。

项目输出以研究复现为主，核心产物包括 IC、分位数组合收益、换手成本估计与特征重要性。默认产物落盘在 `out/runs/<run_name>_<timestamp>_<hash>/`。

## 项目逻辑

为简化项目复杂度，项目默认按月再平衡，假设交易执行以手动下单（或外部执行系统）为主。本项目以“研究 + 信号/建议持仓快照”为核心，暂不覆盖交易执行链路，也就是说，该项目暂不考虑的功能包括但不限于：

1. 账户/持仓对账

* 从券商/OMS 拉当前持仓、现金、可用额度
* 处理分红、拆并股、停牌、无法交易等异常

1. 目标持仓 → 交易指令

* 用“目标持仓 vs 当前持仓”生成 trade list（买/卖数量、限价/市价、最小成交股数、滑点预估）
* 交易成本、冲击成本、换手上限、单票上限、行业/风格约束

1. 执行与回执

* 下单、撤单、回执、部分成交、失败重试
* 盘后生成实际成交版绩效归因

1. 审计与可追溯

* 每次 live run 固化：用的配置、数据截止日、产物文件、最终下单指令、执行回执

## 研究范围

* 研究定位：低频、long-only、面向个人研究的因子挖掘与回测工具。
* 不覆盖：涨跌停/盘口滑点、复杂成交模型、交易系统级别的执行与风控。
* 建议：把 README 当入口，深入细节放在配置与脚本中。

## 数据来源和可复现性

* 数据源可用性取决于供应商 API 与账号权限（TuShare/RQData/EODHD）；README 不保证实时可用性。
* 数据可能回补/修订，导致相同配置在不同时间得到不同结果。
* 想要可复现：固定 `data.start_date/end_date`，避免 `today/now`；保留 `cache/` Parquet；设置 `data.cache_tag`；归档 `out/runs/` 与 `config.used.yml`；记录代码版本（git commit hash）。
* 多数据源差异、symbol 规则与缓存行为见 `docs/providers.md`。

## 快速指南

Python >= 3.12，依赖见 `pyproject.toml`。可选使用 `uv` 与 `direnv`（仓库内提供 `.envrc.example`）。

1. 安装依赖（推荐 `uv`）：

```bash
uv venv --seed
uv sync
```

使用 `venv + pip`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需 RQData：

```bash
pip install -e .[rqdata]
```

准备环境变量（见下文“Credentials / 环境变量”）：

```bash
cp .env.example .env
```

## Credentials / 环境变量

Token/账号需要你在对应数据供应商侧申请。最小必需项取决于 `data.provider`：

* `tushare`：至少 `TUSHARE_TOKEN`
* `rqdata`：`RQDATA_USERNAME` + `RQDATA_PASSWORD`
* `eodhd`：`EODHD_API_TOKEN`

环境变量清单（推荐写入 `.env`）：

* `TUSHARE_TOKEN`
* `TUSHARE_TOKEN_2`（可选，用于轮换）
* `EODHD_API_TOKEN`
* `RQDATA_USERNAME`
* `RQDATA_PASSWORD`

TuShare Token 验证（CLI 入口）：

```bash
csxgb tushare verify-token
```

RQData 初始化支持配置与环境变量混用（配置优先）。示例：

```yaml
data:
  provider: rqdata
  rqdata:
    init:
      username: "your-user"
      password: "your-pass"
```

## CLI 命令一览

### 1) `csxgb run`

* 作用：跑主流程 pipeline（训练/评估/回测一条龙），配置用 `--config` 指定（YAML 路径或内置模板名 `default/cn/hk/us`）。
* 输出：会落到 `out/runs/<run_name>_<timestamp>_<hash>/`，典型产物包括 `summary.json`、`config.used.yml`、IC/回测/特征重要性、以及持仓 CSV 等。
* 注意：数据源可能需要环境变量鉴权（例如 TuShare 的 token）。

### 2) `csxgb grid`

* 作用：做 Top-K × 交易成本(bps) 的敏感性网格，逐个组合跑 pipeline，然后把关键指标汇总到一个 CSV。
* 常用参数（脚本侧定义的）：`--top-k`（可多次传、逗号分隔）、`--cost-bps`、`--output`（默认 `out/runs/grid_summary.csv`）、`--run-name-prefix`、`--log-level`。

### 3) `csxgb holdings`

* 作用：从最近一次 run 的产物里读“当前持仓清单”，并按 `--as-of`（支持 `today/t-1/日期`）输出；支持 backtest/live 两类持仓源。
* 关键参数：

  * `--config`（用于定位“哪个配置的最近 run”）或 `--run-dir`（直接指定 run 目录）
  * `--source auto|backtest|live`（默认 auto）
  * `--format text|csv|json`、`--out`（写文件或 stdout）
* 它读的典型文件：`positions_current.csv`（回测）/ `positions_current_live.csv`（实盘）。

### 4) `csxgb snapshot`

* 作用：给“实盘/准实盘”用的快捷命令：

  * 默认会先跑一次 pipeline（除非你 `--skip-run` 或直接给 `--run-dir`），然后用 live 源吐出 holdings。
* 必须满足：要么 `--config` 要么 `--run-dir`，否则直接报错退出。
* 关键参数：`--as-of`、`--skip-run`、`--top-k`、`--format`、`--out`。
* 隐藏但很重要的约束：如果你用 live 配置，配置解析时要求 `live.enabled=true` 时必须 `eval.save_artifacts=true`（否则无法形成 snapshot）。

### 5) `csxgb rqdata info`

* 作用：初始化 `rqdatac` 并打印登录/用户信息。
* 账号来源优先级：CLI 显式 `--username/--password` > 配置 `rqdata.init` > 环境变量 `RQDATA_USERNAME/RQDATA_PASSWORD`。

### 6) `csxgb rqdata quota`

* 作用：同样初始化 `rqdatac`，然后查 quota 使用情况；`--pretty` 会输出人类可读信息 + 图形化显示剩余流量。
* 依赖：RQData 相关依赖在 optional-deps 里（`rqdata` 这组）。

### 7) `csxgb tushare verify-token`

* 作用：验证 TuShare token 是否可用（实际做法：拿 token 调 TuShare 的接口看能不能返回配额/积分信息），并逐个打印结果。
* 读取的环境变量：`TUSHARE_TOKEN`、`TUSHARE_TOKEN_2`，以及兼容用的 `TUSHARE_API_KEY`。

### 8) `csxgb universe index-components`

* 作用：从 TuShare 拉“指数成分”，写成一个 symbols 文本文件（每行一个）。
* 鉴权：必须先设 `TUSHARE_TOKEN`（或 `TUSHARE_TOKEN_2` / legacy `TUSHARE_API_KEY`），否则直接退出。
* 实现方式：CLI 这层用 `argparse.REMAINDER` 把剩余参数原样转发给脚本（所以脚本支持什么参数，以脚本为准）。

### 9) `csxgb universe hk-connect`

* 作用：构建“港股通股票池（PIT）+ 流动性过滤”的 universe：用 RQData 拉可买标的，再按一段窗口的成交额等指标筛选，输出按日期的 universe 表和“最新一期 symbols”。
* 默认输出：

  * `out/universe/universe_by_date.csv`
  * `out/universe/hk_connect_symbols.txt`
  * meta：`out/universe/universe_by_date.meta.yml`
* 参数入口：`csxgb universe hk-connect --config <yaml> ...`，其余参数同样是转发给脚本。

### 10) `csxgb init-config`

* 作用：把包内置的配置模板导出到你本地文件系统（默认写到 `./config/<template>.yml`）。
* 参数：`--market default/cn/hk/us`、`--out`（文件或目录）、`--force`（允许覆盖）。
* 覆盖保护：目标存在且没 `--force` 就拒绝覆盖。

### 11) `csxgb summarize`

* 作用：跨多个历史 run 目录聚合关键指标（读取每个 run 的 `summary.json` + `config.used.yml`），输出总表 CSV。
* 默认扫描：`out/runs`（递归）。
* 常用参数：`--runs-dir`（可重复传多个目录）、`--output`（默认 `<runs-dir>/runs_summary.csv`）。
* 筛选参数：`--run-name-prefix`（按 run_name 前缀过滤，可重复传）、`--since`（只保留该时点之后的 run）、`--latest-n`（过滤后只保留最新 N 条）。
* 额外筛选列：会自动生成 `flag_short_sample`、`flag_negative_long_short`、`flag_high_turnover` 和 `score`，便于快速筛选异常/候选 run。
* 完整参数清单见 `docs/cli.md`。

常用指令：

```bash
# 主流程
csxgb run --config hk

# 或指定配置文件
csxgb run --config config/hk_selected.yml

# Top-K × 成本敏感性网格
csxgb grid --config config/hk.yml

# 跨 run 汇总总表（研究对比）
csxgb summarize --runs-dir out/runs --output out/runs/runs_summary.csv

# 只看最近一次 grid 相关汇总（示例）
csxgb summarize --runs-dir out/runs --run-name-prefix hk_grid --latest-n 1

# 当期持仓清单（从最近一次 run 读取）
csxgb holdings --config config/hk.yml --as-of t-1
csxgb holdings --config config/hk.yml --as-of 20260131 --format csv --out out/positions/20260131.csv

# 实盘快照（推荐 live 配置）
csxgb run --config config/hk_live.yml
csxgb holdings --config config/hk_live.yml --source live
csxgb snapshot --config config/hk_live.yml
csxgb snapshot --config config/hk_live.yml --skip-run
csxgb snapshot --run-dir out/live_runs/<run_dir>

# RQData 信息 / 配额
csxgb rqdata info
csxgb rqdata quota           # JSON，含百分比与剩余量
csxgb rqdata quota --pretty  # 人类可读 + 进度条

# 指数成分与港股通股票池
csxgb universe index-components --index-code 000300.SH --month 202501
csxgb universe hk-connect --mode daily
```

## 配置模板

配置参考与模板说明 `docs/config.md`。

## 文档导航

建议阅读顺序：

1. 快速上手（本 README）
1. 常见研究流程：`docs/cookbook.md`
1. 配置参数：`docs/config.md`
1. 指标口径：`docs/metrics.md`
1. 输出字段/schema：`docs/outputs.md`
1. 数据源差异：`docs/providers.md`
1. 常见故障：`docs/troubleshooting.md`
1. 开发与测试：`docs/dev.md`
1. 功能全景与规格边界（关键参数入口）：`docs/full_function.md`
1. 贡献说明：`CONTRIBUTING.md`

`docs/index.md` 提供集中导航与起步参数清单。
`docs/full_function.md` 聚焦流程全景、关键参数入口、边界与工时估算，不替代 `docs/config.md` 与 `docs/cli.md` 的完整参数清单。

## 输出产物

* 产物目录：`out/runs/<run_name>_<timestamp>_<hash>/`
* 典型产物：`summary.json`、`config.used.yml`、`ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`、`feature_importance.csv`
* 持仓清单：`positions_by_rebalance.csv`、`positions_current.csv`、`signal_asof`、`next_entry_date`、`holding_window` 字段；`holding_window` 约定为 `entry_date -> next_entry_date`（next 为空表示最新持仓区间）。
* Live 持仓清单：`positions_by_rebalance_live.csv`、`positions_current_live.csv`
* 再平衡差异：`rebalance_diff.csv`、`rebalance_diff_live.csv`
* Live 最新指针：`<eval.output_dir>/latest.json`（仅 `live.enabled=true` 时写入）
* 详细字段说明见 `docs/outputs.md`。

## 模型假设

* 回测为 long-only Top-K 等权组合，按再平衡周期持有。
* 成交价使用 `price_col`（默认 close）并在 `rebalance_date + shift_days` 入场、下一次再平衡/持有期结束出场；近似 EOD 策略。
* 持仓快照输出的是 target holdings，`entry_date = rebalance_date + shift_days`。当 `shift_days=1` 时，月末信号对应次月首个交易日入场，“当月持仓”可能仍是上月组合。
* 成本模型：`transaction_cost_bps` 为单边成本；首期建仓只计单边成本，后续按换手率计算双边成本。
* 换手率考虑权重漂移后的再平衡需求；支持 Top-K 缓冲区（`buffer_exit/buffer_entry`）降低换手；停牌/缺失通过 `is_tradable` + `backtest.exit_price_policy` 近似处理（strict/ffill/delay），仍未建模涨跌停/盘口滑点等。
* `exit_mode=label_horizon` 不支持与再平衡频率重叠（若持有期 > 再平衡间隔会直接跳过/报错）；需保持间隔≈持有期，或改用 `exit_mode=rebalance`。

## 注意事项

* 静态 `symbols`/`symbols_file` 会在历史回测中产生前视偏差；严谨回测应使用 `by_date_file`（PIT），并将 `universe.mode` 设为 `pit` 或开启 `universe.require_by_date`。
* `fetch_index_components.py` 默认导出静态成分列表，适合研究/当期池；历史回测请使用 `--by-date-out` 生成 PIT 成分并接入 `by_date_file`。
* `drop_st` 基于名称匹配；`drop_suspended` 默认改为生成 `is_tradable` 标记（可用 `universe.suspended_policy=filter` 继续硬过滤），仍非严格 PIT。

## 常见研究流程

常见研究流程已移至 `docs/cookbook.md`（见 `docs/cookbook.md`）。
