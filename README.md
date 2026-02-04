# cross-sectional-xgboost

使用 TuShare / RQData / EODHD 日线数据与 XGBoost 回归进行截面因子研究与评估（支持 A/HK/US 多市场配置切换）。流程包含特征工程、时间序列切分、IC 评估、分位数组合收益、换手率估计与特征重要性输出。

项目输出以研究复现为主，核心产物包括 IC、分位数组合收益、换手成本估计与特征重要性。默认产物落盘在 `out/runs/<run_name>_<timestamp>_<hash>/`。

## 项目逻辑

为简化项目复杂度，该项目交易频率按月计，交易频率地，假设交易执行以手动下单为主（或者其他的辅助项目），该项目以“研究 + 信号/建议持仓快照”为主，暂不考虑交易执行，也就是说，该项目暂时不考虑的功能包括但不限于：

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

1. 准备环境变量（见 `Credentials`）：

```bash
cp .env.example .env
```

1. 导出内置配置模板（`default/cn/hk/us`）：

```bash
csxgb init-config --market hk --out config/
```

1. 运行一次（`--config` 支持内置别名或路径）：

```bash
csxgb run --config hk
csxgb run --config config/hk.yml
```

1. 查看产物：

```bash
ls -lh out/runs/
```

## 数据库变量

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

## CLI 指令参考

命令一览：

* `csxgb run`
* `csxgb grid`
* `csxgb holdings`
* `csxgb snapshot`
* `csxgb rqdata info`
* `csxgb rqdata quota`
* `csxgb tushare verify-token`
* `csxgb universe index-components`
* `csxgb universe hk-connect`
* `csxgb init-config`

常用例子：

```bash
# 主流程
csxgb run --config hk

# Top-K × 成本敏感性网格
csxgb grid --config config/hk.yml

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

## 输出产物

* 产物目录：`out/runs/<run_name>_<timestamp>_<hash>/`
* 典型产物：`summary.json`、`config.used.yml`、`ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`、`feature_importance.csv`
* 持仓清单：`positions_by_rebalance.csv`、`positions_current.csv`
* Live 持仓清单：`positions_by_rebalance_live.csv`、`positions_current_live.csv`
* 再平衡差异：`rebalance_diff.csv`、`rebalance_diff_live.csv`
* Live 最新指针：`out/live_runs/latest.json`（指向最新 live run）

## 模型假设

* 回测为 long-only Top-K 等权组合，按再平衡周期持有。
* 成交价使用 `price_col`（默认 close）并在 `rebalance_date + shift_days` 入场、下一次再平衡/持有期结束出场；近似 EOD 策略。
* 持仓快照输出的是 target holdings，`entry_date = rebalance_date + shift_days`。当 `shift_days=1` 时，月末信号对应次月首个交易日入场，“当月持仓”可能仍是上月组合。
* 持仓文件新增 `signal_asof`、`next_entry_date`、`holding_window` 字段；`holding_window` 约定为 `entry_date -> next_entry_date`（next 为空表示最新持仓区间）。
* 成本模型：`transaction_cost_bps` 为单边成本；首期建仓只计单边成本，后续按换手率计算双边成本。
* 换手率已考虑权重漂移后的再平衡需求；支持 Top-K 缓冲区（`buffer_exit/buffer_entry`）降低换手；停牌/缺失通过 `is_tradable` + `backtest.exit_price_policy` 近似处理（strict/ffill/delay），仍未建模涨跌停/盘口滑点等。
* `exit_mode=label_horizon` 不支持与再平衡频率重叠（若持有期 > 再平衡间隔会直接跳过/报错）；需保持间隔≈持有期，或改用 `exit_mode=rebalance`。

## 注意事项

* 静态 `symbols`/`symbols_file` 会在历史回测中产生前视偏差；严谨回测应使用 `by_date_file`（PIT），并将 `universe.mode` 设为 `pit` 或开启 `universe.require_by_date`。
* `fetch_index_components.py` 默认导出静态成分列表，适合研究/当期池；历史回测请使用 `--by-date-out` 生成 PIT 成分并接入 `by_date_file`。
* `drop_st` 基于名称匹配；`drop_suspended` 默认改为生成 `is_tradable` 标记（可用 `universe.suspended_policy=filter` 继续硬过滤），仍非严格 PIT。

## 常见研究流程

常见研究流程已移至 `docs/cookbook.md`（见 `docs/cookbook.md`）。
