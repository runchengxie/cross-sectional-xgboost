# 数据源行为差异与限制

本项目支持 `tushare`、`rqdata`、`eodhd`。同一份配置在不同 provider 下，结果可能不同（字段覆盖、交易日历、历史回补、停牌口径都不同）。

## 快速对照

| 项 | TuShare | RQData | EODHD |
| --- | --- | --- | --- |
| `data.provider` | `tushare` | `rqdata` | `eodhd` |
| 必需鉴权 | `TUSHARE_TOKEN` | `RQDATA_USERNAME` + `RQDATA_PASSWORD` | `EODHD_API_TOKEN` |
| 额外兼容变量 | `TUSHARE_TOKEN_2` / `TUSHARE_API_KEY` | `RQDATA_USER`（用户名别名） | `EODHD_API_KEY` |
| 日线接口 | TuShare endpoint（可配） | `rqdatac.get_price` | EODHD HTTP API |
| 基础信息（name/list_date） | TuShare basic endpoint | `rqdatac.instruments/all_instruments` | `exchange-symbol-list` |
| `last_trading_day` 严格交易日 | 否（回退自然日） | 是（可用交易日历时） | 否（回退自然日） |
| 基本面 `fundamentals.source=provider` | 支持 | 不支持 | 不支持 |

说明：`last_trading_day / last_completed_trading_day` 只有在 `provider=rqdata` 且交易日历可用时才严格按交易日解析，否则会给 warning 并回退自然日。

补充：`csml holdings/snapshot/alloc --as-of last_trading_day` 在能识别到 `provider=rqdata` + `market` 上下文时同样按交易日解析；缺少上下文时回退自然日（会输出 warning）。

## Symbol 规则（重点是 HK）

内部统一格式：

* HK：`00001.HK`（5 位补零 + `.HK`）
* CN/US：按输入与 provider 映射规则处理

provider 侧转换：

1. RQData（HK）：内部 `00001.HK` 会转换为 `00001.XHKG` 调接口。
1. EODHD（HK）：`data.eodhd.hk_symbol_mode` 支持 `strip_one`（常见默认：`00001.HK -> 0001.HK`）、`strip_all`、`pad4`、`pad5`。

## 缓存与“同配置结果变化”

关键配置：

* `data.cache_mode` / `data.daily_cache_mode`：`symbol` 或 `range/window`
* `data.cache_refresh_days`
* `data.cache_refresh_on_hit`
* `data.cache_tag`（或 `cache_version`）

行为差异：

1. `symbol` 模式（默认）：单票一个缓存文件，会按 `cache_refresh_days` 增量刷新末端区间。
1. `range/window` 模式：按请求时间窗口缓存，不做“末端刷新”合并。
1. 不同 `cache_tag` 会形成独立命名空间，适合隔离实验版本。

结果变化常见来源：

1. provider 回补历史数据。
1. 命中缓存后触发末端刷新（`cache_refresh_days > 0`）。
1. 使用相对日期（`today/t-1`）导致样本窗口每日漂移。

## 速率限制与重试

可用 `data.retry` 控制失败重试：

```yaml
data:
  retry:
    max_attempts: 3
    backoff_seconds: 0.5
    max_backoff_seconds: 5.0
    rotate_tokens: true
```

说明：

1. `rotate_tokens` 仅对 TuShare 有效（在 `TUSHARE_TOKEN` 与 `TUSHARE_TOKEN_2` 间轮换）。
1. 高频批量请求前建议先做小窗口验证，确认权限和配额。

## 复现建议

1. 固定 `data.start_date/end_date` 为绝对日期。
1. 固定 `data.provider` 与 provider 专属参数。
1. 保留 `cache/`、`config.used.yml`、`summary.json`。
1. 使用 `data.cache_tag` 隔离关键实验版本。
