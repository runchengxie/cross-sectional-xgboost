# 常见问题排查

## 1. 启动即退出：鉴权变量缺失

常见报错：

* `Please set TUSHARE_TOKEN ... first.`
* `rqdatac is required for provider='rqdata'`
* `Please set EODHD_API_TOKEN ... first.`

排查顺序：

1. 检查 `.env` 是否存在并已填真实值。
1. 检查 `data.provider` 与凭证是否匹配。
1. 重新激活环境后再运行命令。

快速验证：

```bash
csxgb tushare verify-token
csxgb rqdata info
csxgb rqdata quota --pretty
```

## 2. `last_trading_day` 结果看起来不对

现象：设置了 `last_trading_day`，但日期像自然日。

原因：仅 `provider=rqdata` 且交易日历可用时，`last_trading_day` 才严格按交易日解析；否则会回退自然日并给 warning。

建议：

1. 需要严格交易日时使用 `provider=rqdata`。
1. 需要强复现时直接写绝对日期（如 `20260131`）。

## 3. 回测结果为空或样本很少

常见原因：

1. `universe` 过滤太严（`min_symbols_per_date`、`min_turnover`、停牌过滤）。
1. 股票池文件格式不符合要求（`by_date_file` 缺 date/symbol 列）。
1. `label.shift_days` + 末端样本不足，导致可交易窗口过短。

先看这些产物：

1. `summary.json -> data.rows_model / data.dropped_dates`
1. `summary.json -> universe`
1. `out/runs/<run_dir>/dropped_dates.csv`（若有）

## 4. “当月持仓”与预期不一致

典型原因：`shift_days=1` 时，月末信号会在下一交易日入场，`positions_current.csv` 可能仍显示上期组合。

建议：

1. 结合 `signal_asof`、`entry_date`、`next_entry_date`、`holding_window` 一起看。
1. 读取持仓时用 `csxgb holdings --as-of <date>` 明确查询时点。

## 5. Live/snapshot 命令报错

常见报错：

* `live.enabled=true requires eval.save_artifacts=true`
* `live.enabled=true but no live positions were generated`

排查：

1. live 配置里确保 `live.enabled=true` 且 `eval.save_artifacts=true`。
1. 确认 `top_k`、股票池和数据窗口不是空集。
1. 先执行 `csxgb run --config <live.yml>`，再 `csxgb holdings --source live`。

## 6. 结果每天都变，无法复现

常见原因：

1. 使用了 `today/t-1/now`。
1. 数据源发生历史回补。
1. 命中缓存时触发末端刷新。

建议：

1. 固定 `start_date/end_date`。
1. 固定 `data.cache_tag` 并归档 `cache/`。
1. 保留 `config.used.yml`、`summary.json` 与 git commit。

## 7. 参数校验失败

高频配置错误：

1. `eval.save_dataset=true` 但 `eval.save_artifacts=false`
1. `backtest.exit_mode=label_horizon` 与再平衡间隔不匹配
1. `features.cross_sectional.method` / `winsorize_pct` 不合法
1. `eval.bucket_ic.method` 不是 `spearman/pearson`

建议先用内置模板起步，再逐项改动：

```bash
csxgb init-config --market hk --out config/
```

