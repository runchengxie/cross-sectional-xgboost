# 这个项目整体在做哪些事

你命令行量化全流程：给它一份配置，它会产出训练结果 + 回测结果 + 持仓文件 + 可追溯的文档记录。

### 1) 选市场、选股票池（Universe）

配置里支持 static universe（手写 symbols）也支持 PIT/by-date universe（按日期变化的股票池），还有港股通 PIT 股票池构建脚本。

并且对停牌、ST、最小流动性等有策略开关（例如 `drop_suspended`、`suspended_policy`）。

### 2) 拉数据（而且不止一种来源）

它会根据配置决定数据提供方（tushare / rqdata / eodhd），并且还要做不同市场的代码格式转换（港股的补零、后缀、交易所标记等）。

同时 README 里明确了需要环境变量（token/账号）来对接外部数据源。

配置还支持 fundamentals（比如 daily_basic 这类字段映射、衍生特征如 log_mcap、ffill 等）。

### 3) 打标签（label）

对数据进行按再平衡周期定义的未来收益这种更贴近组合交易的标签逻辑，比如 `horizon_mode: next_rebalance`、`rebalance_frequency: M`、`shift_days` 等。

### 4) 做特征（features）

包含技术指标（SMA、RSI、MACD、波动率、量比等）和横截面标准化（zscore/winsorize）。

### 5) 训练模型（XGBoost 回归）

配置里就是 XGB regressor，带一堆常用参数（n_estimators、max_depth、subsample…）以及 sample_weight 方案。

### 6) 评估

* n_splits、分位数分组、IC/IR 等（你还带了 `signal_direction_mode`、`min_abs_ic_to_flip` 这类“信号方向自适应/反转阈值”的逻辑）。
* permutation test（打乱标签检验是不是靠运气）。
* walk-forward（滚动窗口验证，还可以连带回测）。

### 7) 回测（Top-K 组合 + 成本 + 交易规则）

回测参数很全：Top-K、再平衡频率、单边成本 bps、buffer 规则、exit_mode、价格/缺失处理策略等。

而且它会落盘持仓文件：`positions_by_rebalance.csv`、`positions_current.csv`，以及 OOS 和 live 版本。

### 8) 把这一切包装成 CLI 工具，可复现、可“实盘化”

完整 CLI：`run/grid/holdings/snapshot/rqdata quota/.../universe hk-connect`。
输出目录结构也规定了：每次 run 一个带 hash/timestamp 的目录，里面有 `summary.json`、`config.used.yml`、回测/IC/特征重要性/持仓等。
`holdings` 会去读最新 run 的 `summary.json` 来决定用 backtest 还是 live 的持仓文件，并支持 `t-1/today` 这种日期 token。

## 项目难点

把它难点拆开

1. 数据层永远不干净
   多数据源、多市场、不同 symbol 格式、字段命名不一致、缺失值、停牌处理。代码里专门有 column candidates 和 symbol 转换逻辑。

2. 股票池是按日期变的才像真的
   PIT universe、港股通过滤、流动性筛选，这些一做就会牵扯交易日历、上市天数、停牌策略等。

3. 避免无意中泄漏未来信息
   配置里放了很多避免未来信息的机制：按再平衡日期抽样、embargo/purge、next_rebalance horizon 等。

4. 评估不是一个指标能解决的
   IC/分位数、方向翻转阈值、walk-forward、permutation test 这些都是为了回答同一个问题：模型是不是在“撞运气”。  

5. 回测细节
   成本、buffer、退出机制、价格延迟/填充策略、再平衡间隔和标签周期的一致性检查。

6. 可复现和可操作性是工程难点，不是模型难点
   run 目录、summary/config.used、positions 输出、`holdings/snapshot` 读最新结果。

7. 网格搜索是批量实验
   你的 grid 会为每个组合写临时 config，然后直接 `pipeline.run()`，再去磁盘找 `summary.json` 抄一行进 CSV。
   这也是你前面觉得“怎么这么慢”的根本原因之一。

---

## 没有 AI 的时代，这大概要多少总工时？

先说结论：很难低于 200 小时，而且如果你还要边做边学、边调数据源、边修泄漏和回测细节，300–600 小时非常常见。

给你一个比较可信的拆分（按“有经验的 Python/量化工程师单人”粗估）：

| 模块                                | 典型工时（小时） | 为什么耗时                |
| --------------------------------- | -------: | -------------------- |
| 数据接入 + 缓存 + 字段清洗（多 provider、多市场）  |   40–120 | 最多坑，最难一次做对。          |
| 股票池（static + PIT/by-date + 港股通构建） |    20–60 | 交易日历/停牌/流动性筛选全在这。    |
| 标签 + 特征工程 + 标准化                   |    20–60 | 做出来不难，做得“不会泄漏/不会炸”难。 |
| 训练 + CV + 指标（IC/分位数等）             |    20–80 | 评估逻辑一复杂就容易出 bug。     |
| 回测（Top-K、成本、buffer、退出策略、持仓落盘）     |   40–120 | 细节一多，验证成本极高。         |
| walk-forward + permutation test   |    20–80 | “防自欺”越强，代码越复杂。       |
| CLI/配置系统/产物组织/文档                  |    20–80 | 让项目能被“重复使用”的工程部分。    |

把这些加起来，中位数大概 250–350 小时。
如果开发者对量化回测细节不熟，或者数据源/市场规则更复杂，乘个 1.5 倍很正常，也就是 350–600 小时。
