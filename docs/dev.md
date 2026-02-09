# 开发与测试

## 环境准备

推荐使用 `uv`：

```bash
uv venv --seed
uv sync --extra dev
```

如需 RQData 相关能力：

```bash
uv sync --extra dev --extra rqdata
```

如需完整统计检验（`p_value` 等）：

```bash
uv sync --extra dev --extra stats
```

## 本地运行

```bash
csml run --config config/default.yml
```

调试时建议先用较短日期区间，确认流程可跑通后再放大样本窗口。

## 测试

项目使用 `pytest`，默认参数见 `pyproject.toml`（含 `--cov=csml`）。

```bash
uv run pytest
```

常见用法：

```bash
# 只跑某个测试文件
uv run pytest tests/test_metrics.py

# 跑集成测试
uv run pytest -m integration

# 真实 provider 集成测试（需显式启用 + 配置对应 token/账号）
CSML_RUN_PROVIDER_INTEGRATION=1 uv run pytest tests/test_provider_integration.py -m integration
```

## 测试分层约定

建议按以下分层维护测试，避免把“离线回归”与“端到端验证”混在一起：

1. `unit`（默认日常回归）：不依赖外部账号、网络与真实行情接口。
1. `integration`：覆盖跨模块流程（可包含较慢测试或更重的 fixture）。
1. `slow`：显式标注高耗时用例，便于 CI 按需拆分执行。

常用命令：

```bash
# 离线回归（建议本地高频执行）
uv run pytest -m "not integration and not slow"

# 仅集成测试
uv run pytest -m integration
```

## 提交前检查建议

1. 至少跑一遍 `uv run pytest`。
1. 用你修改过的配置跑一次 `csml run --config ...`。
1. 检查 `README.md` 与 `docs/` 是否同步更新。

## 贡献入口

若你准备提交 PR，请同时附上：

1. 变更动机与影响范围。
1. 新增/修改的配置项说明。
1. 回归验证方式（测试命令与关键产物）。
