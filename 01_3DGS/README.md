# 01 3DGS

本实验使用仓库内 `runtime/` 的已验证 3D Gaussian Splatting 运行源码。训练、渲染和指标命令由上级 `main.py` 根据 `experiment.json` 生成，数据固定来自 `../data/<数据集>/train|test`。

该运行版本保留了本数据集使用的自定义 `points3D.bin` 读取逻辑，包括 XYZ、RGB、法向量和 `is_ground.txt` 标签；不要替换为官方只支持标准 COLMAP 点云记录的 `repo/`。配置仍标记为 `metrics_only`，表示统一评估阶段使用 mask，正式论文结果需说明训练阶段的 mask 使用方式。
