# Cookbook

这份 v2 流程用于做“同一研究设置下的多模型对比”，核心顺序是：

1. 线性基线（`ridge` / `elasticnet`）
1. 非线性对照（`xgb_regressor` / `xgb_ranker`）
1. `csml summarize` 跨 run 汇总比较

## 0) 先固定实验基准

开始前先统一一个原则：除了 `model` 和 `eval.run_name`，尽量不改别的参数（尤其是 `universe/label/features/eval/backtest`）。

本仓库已提供一组 HK 配置文件：

1. `config/hk_selected__baseline.yml`（建议作为 sweep/base 配置入口）
1. `config/hk_selected__xgb_regressor.yml`
1. `config/hk_selected__ridge_a1.yml`
1. `config/hk_selected__elasticnet_a0.1_l0.5.yml`
1. `config/hk_selected__xgb_ranker_pairwise.yml`

每次运行后优先看 `config.used.yml`，它是“本次 run 实际生效配置”。

## 0.5) 先过“能不能信”的门槛

建议先做这三个检查，再看夏普：

1. 样本长度：`backtest_periods >= 24`（`summarize` 默认短样本阈值就是 24）。
1. 交易可实现性：`backtest_avg_turnover <= 0.7`（默认高换手阈值 0.7）。
1. 数据稳定性：`data.end_date` 优先固定绝对日期，避免 `today/t-1` 引入复现漂移。

可直接用下面命令筛出“先过门槛”的 run：

```bash
csml summarize \
  --runs-dir out/runs \
  --exclude-flag-short-sample \
  --exclude-flag-high-turnover \
  --exclude-flag-relative-end-date \
  --sort-by score
```

## 1) 先跑线性基线（Ridge / ElasticNet）

`ridge` / `elasticnet` 常用来做稳定基线：参数少、共线性鲁棒、复现成本低。

先跑两条默认基线：

```bash
csml run --config config/hk_selected__ridge_a1.yml
csml run --config config/hk_selected__elasticnet_a0.1_l0.5.yml
```

需要扩展网格时，建议先扫：

1. `ridge.alpha`: `0.01, 0.1, 1, 10, 100`
1. `elasticnet.alpha`: `0.01, 0.1, 1`
1. `elasticnet.l1_ratio`: `0.1, 0.5, 0.9`

如果希望一次性批跑并自动导出对比表，可直接用：

```bash
csml sweep-linear --sweep-config config/sweeps/hk_selected__linear_a.yml
```

临时覆盖少量参数时再叠加 CLI 参数（CLI 会覆盖 sweep config）：

```bash
csml sweep-linear \
  --sweep-config config/sweeps/hk_selected__linear_a.yml \
  --tag hk_linear_a_debug \
  --dry-run
```

命令会在 `out/sweeps/<tag>/` 下生成：

1. `configs/`：本次 sweep 的临时配置文件
1. `jobs.csv`：参数组合清单
1. `run_results.csv`：每个组合执行状态
1. `runs_summary.csv`：自动执行 `csml summarize` 的聚合结果

## 2) 再跑非线性对照组（XGBRegressor / XGBRanker）

```bash
csml run --config config/hk_selected__xgb_regressor.yml
csml run --config config/hk_selected__xgb_ranker_pairwise.yml
```

说明：`xgb_ranker` 与回归模型不是同一个学习问题。项目会按 `trade_date` 分组（query group）训练 ranker，而不是仅仅改一个 objective。

## 3) 用 summarize 汇总所有 run

```bash
csml summarize \
  --runs-dir out/runs \
  --run-name-prefix hk_sel_ \
  --output out/runs/hk_sel_models_summary.csv
```

汇总表会聚合每个 run 的 `summary.json + config.used.yml`，并生成 `flag_*` 与 `score` 字段，方便筛选稳定策略。

## 4) 结果怎么解读（建议顺序）

1. 先看稳定性：`eval.ic`、`eval.pearson_ic`、walk-forward 指标。
1. 再看可交易性：`eval.turnover_mean`、成本拖累、`backtest.stats`。
1. 最后看“是否值得更复杂”：非线性模型若只在单期更高、但波动和换手更差，通常不如线性基线可用。

常见情形：

1. 线性和 XGB 差不多，甚至更好：优先选线性，把精力放在特征和交易假设。
1. XGB 指标更高但不稳定：优先怀疑过拟合，重点看 walk-forward/permutation test。
1. 收益提升但换手飙升：先调 `buffer_exit/buffer_entry` 或降低模型复杂度。
1. `long_short` 经常翻向：多见于弱信号或噪声主导，需回看标签与特征对齐。

## 5) 可选：`csml grid` 做 Top-K/成本/buffer 形状分析

`csml grid` 不是模型超参网格搜索。它会先基于给定配置跑一次 base pipeline，读取 `eval_scored.parquet`，然后在同一份 scored 数据上循环 `Top-K × cost_bps × buffer_exit × buffer_entry` 组合，不会为每个网格点重训模型。

示例：

```bash
csml grid \
  --config config/hk_selected__ridge_a1.yml \
  --top-k 5,10,20 \
  --cost-bps 10,25,40 \
  --buffer-exit 8,10 \
  --buffer-entry 4,5
```

## 6) 最小可执行清单

```bash
csml run --config config/hk_selected__ridge_a1.yml
csml run --config config/hk_selected__elasticnet_a0.1_l0.5.yml
csml run --config config/hk_selected__xgb_regressor.yml
csml run --config config/hk_selected__xgb_ranker_pairwise.yml
csml summarize --runs-dir out/runs --run-name-prefix hk_sel_ --output out/runs/hk_sel_models_summary.csv
```
