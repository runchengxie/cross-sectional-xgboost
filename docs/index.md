# 文档导航

这份文档用于回答两个问题：

1. 第一次进入仓库，怎样 10 分钟跑通一次实验。
1. 跑通后，结果该看哪里、配置该改哪里。

## 推荐阅读顺序

1. `README.md`：安装、命令入口、核心假设。
1. `docs/cookbook.md`：照抄命令跑出可复现结果。
1. `docs/config.md`：理解并修改配置参数。
1. `docs/metrics.md`：解读 IC、回测与风险指标。
1. `docs/outputs.md`：消费 `summary.json` 与持仓文件字段。
1. `docs/providers.md`：多数据源差异、symbol 规则、缓存影响。
1. `docs/troubleshooting.md`：排查常见报错与结果偏差。
1. `docs/dev.md`：本地开发、测试与代码贡献流程。
1. `docs/full_function.md`：项目功能全景与实现细节。

## 10 分钟起步

```bash
uv venv --seed
uv sync
cp .env.example .env
csxgb run --config config/hk.yml
```

跑完后优先看：

1. `out/runs/<run_dir>/summary.json`
1. `out/runs/<run_dir>/config.used.yml`
1. `out/runs/<run_dir>/positions_current.csv`

## 起步时优先改的参数

1. `data.provider`：先选数据源（`tushare/rqdata/eodhd`）。
1. `data.end_date`：回测截止日，复现时尽量用固定日期（避免 `today/now`）。
1. `universe.mode` + `universe.by_date_file`：历史回测优先 PIT，避免前视偏差。
1. `eval.top_k` 与 `backtest.top_k`：选股数量。
1. `eval.transaction_cost_bps` 与 `backtest.transaction_cost_bps`：成本敏感性。
1. `label.shift_days`：信号到入场的延迟，直接影响“当前持仓”解释。

## 常见坑

1. 使用静态 `symbols/symbols_file` 做长历史回测，容易产生幸存者偏差。
1. `last_trading_day` 只有在 `provider=rqdata` 且可用交易日历时才是严格交易日，否则会回退自然日。
1. 数据源会回补历史，缓存策略不同会导致同配置不同结果。
1. `live.enabled=true` 时必须配合 `eval.save_artifacts=true`，否则无法产出 snapshot 所需文件。
