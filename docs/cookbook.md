# Cookbook

## 港股通 PIT + 成本/Top-K 网格对照

```bash
# 1) 生成港股通 PIT + 流动性池
csxgb universe hk-connect --config config/universe.hk_connect.yml

# 2) 批量跑 Top-K / 交易成本组合
csxgb grid --config config/hk.yml

# 3) 查看汇总结果
ls -lh out/runs/grid_summary.csv
```

覆盖默认网格参数：

```bash
csxgb grid --config config/hk.yml \
  --top-k 5,10 \
  --cost-bps 25,40 \
  --output out/runs/my_grid.csv \
  --run-name-prefix hk_grid
```
