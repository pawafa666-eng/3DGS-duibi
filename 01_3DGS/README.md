# 01 3DGS

标准 3D Gaussian Splatting 基线，源码来自 `graphdeco-inria/gaussian-splatting`。训练、渲染和指标命令由上级 `main.py` 根据 `experiment.json` 生成，数据固定来自 `../data/<数据集>/train|test`。

该上游基线未改训练损失，因此配置标记为 `metrics_only`：mask 会保留给统一评估协议，正式论文结果需单独注明训练阶段未使用 native mask。

