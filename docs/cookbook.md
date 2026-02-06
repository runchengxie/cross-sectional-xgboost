# Cookbook

这份流程的目标是：从 0 到可复现实验，再到可落地的 live 持仓快照。

## 1) 先选市场和数据源

建议先导出模板，再改参数：

```bash
csxgb init-config --market hk --out config/
```

优先确认：

1. `market`（`cn/hk/us`）
1. `data.provider`（`tushare/rqdata/eodhd`）
1. 对应鉴权变量已在 `.env` 设置

## 2) 先决定股票池模式（static vs PIT）

历史回测尽量用 PIT（按日期股票池），避免前视偏差。

HK 示例（先构建港股通 PIT 池）：

```bash
csxgb universe hk-connect --config config/universe.hk_connect.yml
```

然后在策略配置里使用 `universe.by_date_file`，并设置：

```yaml
universe:
  mode: pit
  require_by_date: true
```

## 3) 跑一次基线 `run`

```bash
csxgb run --config config/hk.yml
```

跑完先看：

1. `summary.json`（关键指标）
1. `config.used.yml`（实际生效配置）
1. `positions_current.csv`（最新持仓）

## 4) 跑 `grid` 做 Top-K/成本敏感性

```bash
csxgb grid --config config/hk.yml
ls -lh out/runs/grid_summary.csv
```

可覆盖默认参数：

```bash
csxgb grid --config config/hk.yml \
  --top-k 5,10 \
  --cost-bps 25,40 \
  --output out/runs/my_grid.csv \
  --run-name-prefix hk_grid
```

## 5) 选参并解读结果

从 `grid_summary.csv` 选一组参数后，复制配置并修改：

1. `eval.top_k`、`backtest.top_k`
1. `eval.transaction_cost_bps`、`backtest.transaction_cost_bps`

再跑一次正式单次：

```bash
csxgb run --config config/hk_selected.yml
```

解读优先级建议：

1. `eval.ic` / `eval.pearson_ic`：信号稳定性
1. `eval.quantile_mean` / `eval.long_short`：分位单调性
1. `eval.turnover_mean` + 成本参数：交易可行性
1. `backtest.stats`：收益、回撤、尾部风险

指标定义详见 `docs/metrics.md`。

## 6) 固化可复现实验

每次正式实验都归档：

1. `config.used.yml`
1. `summary.json`
1. `cache/`（或最少保留本次运行依赖缓存）

如果需要长期复现，尽量把 `data.end_date` 写死为绝对日期。

## 7) 生成 live 快照（最小链路）

live 配置建议单独文件（例如 `config/hk_selected_live.yml`），关键设置：

```yaml
eval:
  output_dir: out/live_runs
  save_artifacts: true

backtest:
  enabled: false

live:
  enabled: true
  as_of: t-1
```

执行方式：

```bash
# 完整流程：先 run，再输出当期持仓
csxgb snapshot --config config/hk_selected_live.yml

# 已有最新 run，仅导出快照
csxgb snapshot --config config/hk_selected_live.yml --skip-run

# 直接读取 live 持仓
csxgb holdings --config config/hk_selected_live.yml --source live
```
