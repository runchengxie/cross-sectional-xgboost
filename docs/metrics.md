# cross-sectional-xgboost 评估指标说明书

## 0. 这个项目在干什么（用一句话讲清楚）

它用 XGBoost 回归在每天的股票截面上预测一个目标列 `future_return`（也就是“未来一段持有期的收益”），然后用 IC、分位数组合、换手估计、Top-K 回测来评估这堆预测到底有没有交易意义。配置里明确是回归目标 `reg:squarederror`，label 目标列是 `future_return`。

项目典型产物会落到 `out/runs/<run_name>_<timestamp>_<hash>/`，包括 `summary.json`、`ic_*.csv`、`quantile_returns.csv`、`backtest_*.csv`、`feature_importance.csv` 等。

## 1) 数据与股票池相关指标

> 这些是防止模型形成前视/幸存者偏差的基础

这些都在 `summary.json -> data / universe` 里：

### 1.1 symbols / rows / rows_model

* symbols：本次跑了多少只股票。
* rows：全量数据行数。
* rows_model：真正进入建模（含特征、label、过滤后）的行数。
  这些能快速反映数据是不是大面积缺失/被过滤掉。

### 1.2 min_symbols_per_date / dropped_dates

* min_symbols_per_date：每天至少要有多少只股票才算该交易日可用作训练。
* dropped_dates：有多少个交易日因为股票数不足被丢掉。
  如果 dropped_dates 很多，IC、分位收益都会波动明显。

### 1.3 universe.mode（pit/static）与停牌处理

* universe.mode：支持 `pit`（按日期动态股票池）等模式，且可以要求 `require_by_date`。
* drop_suspended / suspended_policy：是否丢弃停牌、怎么处理停牌。

## 2) 标签相关（模型到底在预测什么）

配置里 label 常见字段：

* target_col：默认 `future_return`。
* horizon_days / horizon_mode / rebalance_frequency / shift_days：决定未来收益怎么算，例如 `next_rebalance` + 月度再平衡，和回测节奏绑定。

---

## 3) 因子/预测质量指标

这些主要来自 `src/csxgb/metrics.py`，并汇总进 `summary.json -> eval`。

### 3.1 日度 IC 序列（Spearman Rank IC）

* daily IC（ic_test.csv）：对每天的截面，计算预测值 `pred` 和真实 `future_return` 的 Spearman 秩相关。
* 为什么用 Spearman：更关注排序对不对，对极端值不那么敏感。

### 3.2 IC 汇总统计（summary.eval.ic）

对上面的 IC 序列做汇总：`n / mean / std / ir / t_stat / p_value`。

* n：有效交易日数（有些天会被跳过，比如当日 `future_return` 没有足够离散度）。
* mean：平均 IC。
* std：IC 的波动。
* IR（Information Ratio, 这里是 ic_mean / ic_std）：越大代表IC 更稳定。
* t_stat / p_value：把 IC 当成样本均值检验（非常粗糙但常用）。`p_value` 只有在装了 SciPy 才算。

> 友情提示：金融时间序列并不独立同分布，t/p 只能当简易指标。

### 3.3 训练集 IC / CV IC（用来决定信号方向）

* train_ic：训练集的 IC 汇总（可开关）。
* cv_ic：时间序列 CV 得到的 IC 分数列表 + mean/std（用于更稳地判断方向）。
* signal_direction_mode：支持 `fixed / train_ic / cv_ic`；当你选 `cv_ic` 时，会用 CV 的 ic_mean 符号决定是否把预测整体乘以 -1（翻转方向），并可设置最小阈值 `min_abs_ic_to_flip`。

> 模型可能学到“负相关因子”，翻一下方向就变成正向信号，因此这一步就是自动完成此项操作。

### 3.4 分位数组合收益（quantile_returns.csv）

* 把每天预测值做分位分组（默认 5 组），计算每个分位的平均 `future_return`，得到一个 “trade_date × quantile” 的表。
  用途：
* 看单调性：高分位是否系统性优于低分位。
* 看尾部：最高分位是不是特别好，还是只是中间平平。

### 3.5 quantile_mean 与 long_short

* quantile_mean：每个分位的平均收益（跨时间取平均后写进 summary）。
* long_short：最高分位均值 - 最低分位均值（一个非常粗暴但直观的截面收益跨度）。

---

## 4) 换手与交易成本相关（不看这个，回测就是童话）

项目会在 eval 和 backtest 两处都输出换手相关信息。

### 4.1 turnover（换手率）与 turnover_mean

* turnover series：每次调仓点的换手率序列（`turnover_eval.csv` / `backtest_turnover.csv`）。
* turnover_mean：换手均值写进 `summary.eval.turnover_mean`。
* 换手怎么定义（核心直觉）：用新旧持仓的重叠程度来算，近似是 `1 - overlap/k`（重叠越少，换手越高）。

### 4.2 buffer_entry / buffer_exit（缓冲区机制）

* 这是为了降低换手：只有排名掉出更远才卖出（exit buffer），只有排名足够靠前才买入（entry buffer）。参数会写进 summary。
  回测里选股函数也显式接收 `buffer_exit/buffer_entry` 和上一期持仓。

### 4.3 transaction_cost_bps 与 cost drag

* transaction_cost_bps：按 bps 设定的单边交易成本（eval/backtest 都有）。
* avg_cost_drag / avg_cost_drag：回测统计里会输出平均成本拖累（本质是因为换手付出的成本把收益吃掉多少）。

---

## 5) 回测绩效指标（Top-K 策略到底赚不赚钱）

回测核心是 `backtest_topk` + `summarize_period_returns`，输出在 `summary.json -> backtest.stats`，并落 CSV。

### 5.1 净收益 vs 毛收益

* gross_return（backtest_gross.csv）：不扣交易成本的周期收益。
* net_return（backtest_net.csv）：扣完成本后的周期收益（更该看这个）。

### 5.2 核心绩效统计（summarize_period_returns）

* total_return：样本期总收益（nav 最后 - 1）。
* ann_return：按持有天数折算的年化收益（不是按自然年，是按交易日近似）。
* ann_vol：年化波动（用周期收益 std × sqrt(periods_per_year)）。
* sharpe：均值/波动 × sqrt(periods_per_year)。
* max_drawdown：净值序列的最大回撤（nav/nav_max - 1 的最小值）。
* avg_holding / periods_per_year：平均持有期长度，以及由此推算的一年多少期。

### 5.3 回测额外统计（更贴近交易）

回测在 stats 里额外塞了：

* avg_turnover：回测调仓点换手平均。
* avg_cost_drag：平均成本拖累。
* mode / long_k：long-only 还是 long-short、持仓数等。

---

## 6) 基准与主动收益指标（有 benchmark 的时候才有）

如果回测配置了 `benchmark_symbol`，会构建基准收益序列，并计算主动收益统计。

### 6.1 active_return（策略 - 基准）

* active_return series（backtest_active.csv）：每期策略净收益减去基准收益。

### 6.2 主动收益统计（summarize_active_returns）

输出字段：

* tracking_error：主动收益的年化波动（std × sqrt(periods_per_year)）。
* information_ratio：主动均值/主动波动 × sqrt(periods_per_year)。
* beta：策略对基准的 beta（协方差/基准方差）。
* alpha（ann）：用 `strategy.mean - beta*benchmark.mean` 再乘 `periods_per_year` 得到年化 alpha。
* corr：策略与基准收益相关。
* active_total_return：策略总收益相对基准总收益的比值超额。

---

## 7) 鲁棒性与“别自嗨”测试

### 7.1 permutation test（置换检验）

* 配置里可以开 `eval.permutation_test.enabled`。
* 做法：训练阶段把 label 打乱再训练，看看测试集 IC 均值分布长啥样（本质是随机情况模型预测能力有多强）。代码里每次 run 都把 mean(IC) 记下来。
* 落盘：会输出 `permutation_test.csv`（列名 ic）。

怎么解读：

* 如果真实模型的 `IC mean` 只比 permutation 分布好一点点，那基本在训练噪声。

### 7.2 walk-forward（滚动窗口验证）

* 配置里默认 `walk_forward.enabled: true`。
* 落盘：`walk_forward_summary.csv`，并且 summary 里也会带 `walk_forward` 结果。

怎么解读：

* 看每个窗口的 IC/回测是不是都差不多，还是只有某一个窗口很好，其他窗口表现一般。

---

## 8) 特征重要性（解释模型在看啥）

* 输出 `feature_importance.csv`，来自 XGBoost 的 `feature_importances_`（注意：这不是因果解释，只是模型内部的分裂贡献类权重）。

怎么解读：

* 用来发现明显异常（比如某个本不该重要的字段突然 0.99），以及做特征筛选的线索。
* 不要用它证明这个因子有效，它证明不了。

# 9) 补充指标（简单、直观、实用）

## 9.1 预测层面的补充

### A) Pearson IC（线性相关）

* 是什么：每天截面上 `corr(pred, future_return)`（不做 rank）。
* 解决什么：Spearman 只看排序，不敏感于“幅度是否有意义”。Pearson 能补上这一块。
* 怎么看：如果 Spearman 很高但 Pearson 接近 0，可能只是排序有点用，但强度不稳。

### B) MAE / RMSE / R²（误差指标）

* 是什么

  * MAE：平均绝对误差
  * RMSE：均方根误差（更惩罚大错）
  * R²：解释方差比例（可选）
* 解决什么：快速抓 bug/退化，比如模型输出几乎常数、或 label 缩放错误。

> 现实里你可能不会把它们当 KPI，但它们是非常好的错误预警。

### C) 分桶 IC（稳定性拆解）

* 是什么：按行业/市值/流动性分组分别算 IC。
* 解决什么：避免某个小角落扛起全场，或者策略其实是在吃某个风格暴露。

## 9.2 策略层面的补充

### D) rolling IC / rolling Sharpe（滚动表现）

* 是什么：比如 6M/12M 滚动窗口计算 IC mean/IR、Sharpe。
* 解决什么：识别策略是不是只在某段行情有效。

### E) Drawdown duration（回撤持续时间）与 recovery time（恢复时间）

* 是什么：最大回撤跌了多久、从谷底爬回前高用了多久。
* 解决什么：最大回撤只说跌多深，不说持续多久。而持续多久往往感官更痛苦。

### F) Sortino / Calmar（更适合交易语境的比率）

* Sortino：只用下行波动算风险，适合非对称收益。
* Calmar：年化收益 / 最大回撤，低频策略常用。

### G) hit rate（方向命中）/ top-K 正收益占比

* 是什么

  * hit rate：`sign(pred) == sign(real)` 的比例
  * top-K positive ratio：Top-K 里未来收益为正的占比
* 解决什么：有些策略不追求高 IC，但追求选出来的大概率别是不良标的。

## 9.3 风险与尾部补充

### H) 收益分布统计（skew/kurtosis、VaR/CVaR）

* 是什么：看收益是否来自几次偶发的情况，或者是不是高度右偏靠极少数大赚撑起来。
* 至少上 95% VaR + CVaR（Expected Shortfall），关注尾部风险。
