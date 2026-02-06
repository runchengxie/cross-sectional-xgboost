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
csxgb run --config config/default.yml
```

调试时建议先用较短日期区间，确认流程可跑通后再放大样本窗口。

## 测试

项目使用 `pytest`，默认参数见 `pyproject.toml`（含 `--cov=csxgb`）。

```bash
uv run pytest
```

常见用法：

```bash
# 只跑某个测试文件
uv run pytest tests/test_metrics.py

# 跑集成测试
uv run pytest -m integration
```

## 提交前检查建议

1. 至少跑一遍 `uv run pytest`。
1. 用你修改过的配置跑一次 `csxgb run --config ...`。
1. 检查 `README.md` 与 `docs/` 是否同步更新。

## 贡献入口

若你准备提交 PR，请同时附上：

1. 变更动机与影响范围。
1. 新增/修改的配置项说明。
1. 回归验证方式（测试命令与关键产物）。
