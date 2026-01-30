# cross-sectional-xgboost

使用 TuShare / RQData / EODHD 日线数据与 XGBoost 回归进行截面因子挖掘和评估（支持 A/HK/US 多市场配置切换）。流程包含特征工程、时间序列切分、IC 评估、分位数组合收益、换手率估计与特征重要性输出。

项目是基于一个散户的视角，因此：

* 低频策略
* 无做空
* 不涉及过大的股票池，避免带来巨大的滑点和交易成本

## 功能概览

* 拉取 TuShare / RQData / EODHD 日线数据（按 `data.provider` 选择数据源）并缓存到 `cache/`（Parquet）
* 计算 SMA、RSI、MACD、成交量等技术指标
* 训练 XGBoost 回归模型并评估截面 IC
* 输出分位数组合收益、长短组合收益、换手率估计

## 环境与依赖

* Python >= 3.12
* 依赖见 `pyproject.toml`
* 可选：`uv` + `direnv`（仓库内已提供 `.envrc.example`）

## 安装方式

使用 `uv`（推荐）：

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

若使用 RQData（含 rqdatac/rqsdk），请安装可选依赖：

```bash
pip install -e .[rqdata]
```

## 配置 TuShare Token（仅当 data.provider=tushare）

主程序与工具脚本优先读取 `TUSHARE_TOKEN`，其次 `TUSHARE_TOKEN_2`，仅在兼容旧配置时才读取 `TUSHARE_API_KEY`。
如果你已从 `.env.example` 复制到 `.env`，请确保补充 `TUSHARE_TOKEN`（可选 `TUSHARE_TOKEN_2` 作为备用）。
默认会在重试时按顺序轮换 Token（可通过 `data.retry.rotate_tokens` 关闭）。

示例 `.env`：

```bash
TUSHARE_TOKEN="replace-with-your-tushare-pro-token"
TUSHARE_TOKEN_2="replace-with-your-second-tushare-pro-token"
# Legacy alias (avoid using unless required by old setups)
# TUSHARE_API_KEY="replace-with-your-tushare-pro-token"
EODHD_API_TOKEN="replace-with-your-eodhd-token"
RQDATA_USERNAME="your-user"
RQDATA_PASSWORD="your-pass"
```

若使用 `direnv`：

```bash
cp .envrc.example .envrc
direnv allow
```

## 配置 RQData（仅当 data.provider=rqdata）

需要安装 `rqdatac`（建议 `pip install -e .[rqdata]`）。项目仅用到日线行情接口，不要求 `rqdatac_hk`（除非你要用港股通成分股等扩展功能）。

如需传入初始化参数，可在配置中设置 `data.rqdata.init`，例如：

```yaml
data:
  provider: rqdata
  rqdata:
    init:
      username: "your-user"
      password: "your-pass"
```

也可使用环境变量 `RQDATA_USERNAME`（或 `RQDATA_USER`）/ `RQDATA_PASSWORD`（配置文件优先级更高）。

## 配置 EODHD（仅当 data.provider=eodhd）

使用环境变量 `EODHD_API_TOKEN`（或配置 `data.eodhd.api_token`）。可选字段：

* `data.eodhd.exchange`：交易所代码（如 `HK`）。
* `data.eodhd.hk_symbol_mode`：港股代码转换模式（`keep` / `strip_one` / `strip_all` / `pad4` / `pad5`）。
* `data.eodhd.period` / `data.eodhd.order` / `data.eodhd.fmt`：用于日线接口的参数透传。

## 运行

```bash
csxgb run
csxgb run --config config/cn.yml
csxgb run --config config/hk.yml
csxgb run --config config/us.yml
```

不传 `--config` 时使用内置 `default.yml`。

若需要生成可编辑的本地配置文件，可运行：`csxgb init-config --market cn`（或 `hk` / `us` / `default`）。

## CLI 命令速览

```bash
# 主流程
csxgb run

# RQData 信息 / 配额
csxgb rqdata info
csxgb rqdata quota
```

注：`rqdata` 命令需要安装可选依赖 `.[rqdata]`。

```bash

# 指数成分与港股通股票池（参数透传给原脚本）
csxgb universe index-components --index-code 000300.SH --month 202501
csxgb universe hk-connect --mode daily

# TuShare token 验证
csxgb tushare verify-token
```

输出包含：

* CV IC 与 Daily IC
* 分位数收益与长短组合收益
* Top-K 换手率估计与成本拖累
* 简易 long-only 回测（按再平衡周期持有到下一次）
* 特征重要性排序
* 评估与回测产物默认落盘到 `out/runs/<run_name>_<timestamp>_<hash>/`
  * 典型产物：`summary.json`、`config.used.yml`、`ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`、`feature_importance.csv`

## 回测假设与限制

* 回测为 long-only Top-K 等权组合，按再平衡周期持有。
* 成交价使用 `price_col`（默认 close）并在 `rebalance_date + shift_days` 入场、下一次再平衡/持有期结束出场；近似 EOD 策略。
* 成本模型：`transaction_cost_bps` 为单边成本；首期建仓只计单边成本，后续按换手率计算双边成本。
* 换手率已考虑权重漂移后的再平衡需求，但仍忽略滑点、成交冲击、停牌/涨跌停限制等。
* `exit_mode=label_horizon` 不支持与再平衡频率重叠（若持有期 > 再平衡间隔会直接跳过/报错）；需保持间隔≈持有期，或改用 `exit_mode=rebalance`。

## 数据偏差声明

* 静态 `symbols`/`symbols_file` 会在历史回测中产生前视偏差；严谨回测应使用 `by_date_file`（PIT）。
* `fetch_index_components.py` 默认导出静态成分列表，适合研究/当期池；历史回测请使用 `--by-date-out` 生成 PIT 成分并接入 `by_date_file`。
* `drop_st` 基于名称匹配，`drop_suspended` 基于成交量/成交额近似过滤，均非严格 PIT。

## 工具脚本

CLI 已封装常用脚本（见上方命令速览），也可直接运行：

* `python -m csxgb.project_tools.verify_tushare_tokens`：验证 TuShare Token 是否可用
* `project_tools/combine_code.py`：打包项目源码为单文件文本（用于归档/审查）
* `project_tools/run_grid.py`：批量跑 Top-K × 成本敏感性并汇总 CSV
* `python -m csxgb.project_tools.fetch_index_components`：拉取指数成分并导出为 `symbols_file` 列表
* `python -m csxgb.project_tools.build_hk_connect_universe`：基于港股通 PIT + 成交额筛选生成 `out/universe/universe_by_date.csv`

## 自定义参数

在 `config/default.yml` 或各市场配置中调整：

* `universe`：股票池、过滤条件、最小截面规模（支持 `by_date_file` 动态池）
* `market`：`cn` / `hk` / `us`
* `data`：`provider`、`rqdata` / `eodhd` 或 `daily_endpoint` / `basic_endpoint` / `column_map`（字段映射为 `trade_date/ts_code/close/vol/amount`）、`cache_tag`、`retry`
* `label`：预测窗口、shift、winsorize
* `features`：特征清单与窗口
* `model`：XGBoost 参数
* `eval`：切分、分位数、换手成本、embargo/purge、`signal_direction_mode`，以及可选的 `report_train_ic`、`save_artifacts`、`permutation_test` 与 `walk_forward`
* `backtest`：再平衡频率、Top-K、成本、`long_only/short_k`、基准与 `exit_mode`

示例（生成指数成分列表）：

```bash
csxgb universe index-components \
  --index-code 000300.SH \
  --month 202501 \
  --out hs300_symbols.txt
```

生成 PIT 股票池（用于 `by_date_file`）：

```bash
csxgb universe index-components \
  --index-code 000300.SH \
  --start-date 20200101 \
  --end-date 20251231 \
  --by-date-out out/universe/index_universe_by_date.csv
```

示例（港股通 PIT + 流动性池）：

默认配置（建议先用默认跑一遍）：

```bash
csxgb universe hk-connect
```

覆盖单个参数：

```bash
csxgb universe hk-connect --top-quantile 0.9
```

明确日期区间（回测）：

```bash
csxgb universe hk-connect \
  --start-date 20200101 \
  --end-date 20251231
```

日常更新（默认 T-1）：

```bash
csxgb universe hk-connect --mode daily
```

然后在配置中设置：

```yaml
universe:
  by_date_file: "out/universe/universe_by_date.csv"
```

说明：

* `drop_suspended` 通过成交量/成交额为 0 的数据近似过滤停牌。
* `drop_st` 基于 `stock_basic` 的名称匹配，仅适用于 A 股，属于粗过滤。
* `eval.embargo_days` / `eval.purge_days` 按交易日索引计数（基于已排序的 `trade_date`）。
* 日线缓存文件名统一为 `{market}_{provider}_daily_{symbol}_{START}_{END}.parquet`。
  * 若设置 `data.cache_tag`（或 `cache_version`），文件名会变为 `{market}_{provider}_{cache_tag}_daily_{symbol}_{START}_{END}.parquet`。
* 若 `data.end_date` 使用 `today/now`，每天都会生成新的缓存键；想复用缓存请固定 `start_date/end_date`。
* 港股通股票池默认配置来自内置模板 `universe.hk_connect.yml`（仓库内同名文件在 `config/universe.hk_connect.yml`），CLI 参数可覆盖。
* `mode=backtest` 要求固定 `end_date`；`mode=daily` 默认使用最近一个已完成交易日 (T-1)，并在输出文件名后追加日期。
* `top_quantile` 的语义是“保留分位数以上的标的”，例如 `0.8` 会保留流动性最高的 20%。
* 默认会在 CSV 旁输出 `*.meta.yml`，记录最终生效参数与每期股票池数量（默认路径在 `out/universe/`）。

## SOP：港股通池 + 成本/Top-K 网格对照

推荐的最小流程（先生成 PIT 流动性池，再跑 Top-K × 成本敏感性）：

```bash
# 1) 生成港股通 PIT + 流动性池
csxgb universe hk-connect --config config/universe.hk_connect.yml

# 2) 批量跑 Top-K / 交易成本组合
python project_tools/run_grid.py --config config/hk.yml

# 3) 查看汇总结果
ls -lh out/runs/grid_summary.csv
```

默认网格为 `top_k = 5,10,20`，`transaction_cost_bps = 15,25,40`（单边）。可按需覆盖：

```bash
python project_tools/run_grid.py \
  --config config/hk.yml \
  --top-k 5,10 \
  --cost-bps 25,40 \
  --output out/runs/my_grid.csv \
  --run-name-prefix hk_grid
```
