# cross-sectional-xgboost

使用 TuShare 美股日线数据与 XGBoost 回归进行截面因子挖掘和评估。流程包含特征工程、时间序列切分、IC 评估、分位数组合收益、换手率估计与特征重要性输出。

## 功能概览
- 拉取 TuShare `us_daily` 数据并缓存到 `cache/`（Parquet）
- 计算 SMA、RSI、MACD、成交量等技术指标
- 训练 XGBoost 回归模型并评估截面 IC
- 输出分位数组合收益、长短组合收益、换手率估计

## 环境与依赖
- Python >= 3.9
- 依赖见 `pyproject.toml`
- 可选：`uv` + `direnv`（仓库内已提供 `.envrc.example`）

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

## 配置 TuShare Token
主程序读取 `TUSHARE_API_KEY`，工具脚本会读取 `TUSHARE_TOKEN` / `TUSHARE_TOKEN_2`。
如果你已从 `.env.example` 复制到 `.env`，请确保补充 `TUSHARE_API_KEY`。

示例 `.env`：
```bash
TUSHARE_API_KEY="replace-with-your-tushare-pro-token"
TUSHARE_TOKEN="replace-with-your-tushare-pro-token"
TUSHARE_TOKEN_2="replace-with-your-second-tushare-pro-token"
```

若使用 `direnv`：
```bash
cp .envrc.example .envrc
direnv allow
```

## 运行
```bash
python main.py
```

输出包含：
- CV IC 与 Daily IC
- 分位数收益与长短组合收益
- Top-K 换手率估计与成本拖累
- 特征重要性排序

## 工具脚本
- `project_tools/verify_tushare_tokens.py`：验证 TuShare Token 是否可用
- `project_tools/combine_code.py`：打包项目源码为单文件文本（用于归档/审查）

## 自定义参数
可在 `main.py` 中调整：
- `SYMBOLS`：股票池
- `LABEL_HORIZON_DAYS`：收益预测窗口
- `FEATURES`：特征列表
- `XGB_PARAMS`：模型参数
- `REBALANCE_FREQUENCY` / `TRANSACTION_COST_BPS`：评估参数
