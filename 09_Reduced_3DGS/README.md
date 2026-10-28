# 09 Reduced-3DGS

Reduced-3DGS 轻量化表示方法，源码来自 `graphdeco-inria/reduced-3dgs`。该方法的 PLY 格式与原版不同，必须保持本目录独立输出，不能与其它方法共用模型目录。

半精度量化保存兼容 NumPy 1.x 和 2.x。如果 30000 次训练已经产生 `point_cloud_quantised.ply`，但最后的半精度保存因 `np.cast` 报错，可从该文件恢复，无需重训。下面以原 Ubuntu 运行目录 `output/03` 为例，恢复文件会新建在用户主目录，原文件及运行状态不变：

```bash
python 09_Reduced_3DGS/recover_quantised_half.py output/03/09_Reduced_3DGS/train/point_cloud/iteration_30000/point_cloud_quantised.ply "$HOME/point_cloud_quantised_half_03.ply"
```

脚本要求 NumPy 和 plyfile，支持标准 SH3 小端二进制量化 PLY；输出已存在、结构不符或数值不能表示为 float16 时会停止，不覆盖文件。完成后状态仍可能为 `failed`，不要用 `--resume ... --only 09` 重新启动原训练；恢复模型需另外保存和使用。

