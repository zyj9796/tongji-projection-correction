# 投影校正

本目录收纳建筑三维投影、全局/局部偏移优化和 DSM 模拟 SAR 校正流程。

- `touying/`：早期投影校正工作包。
- `touying_roof_workflow/`：屋顶投影、全局偏移优化和论文图件流程；目录内保留其独立 Git 元数据。

两个工作包的 `data/` 中使用软链接访问 `geocoding/data/` 和 `geocoding/results/`，不复制大体量数据。

```bash
bash projection_correction/touying/run_full_area_projection.sh
bash projection_correction/touying/run_optimize_global_projection_shift.sh
bash projection_correction/touying/run_apply_projection_correction.sh
bash projection_correction/touying_roof_workflow/run_full_area_projection.sh
```

每个工作包的生成物保存在自身 `results/` 中。
