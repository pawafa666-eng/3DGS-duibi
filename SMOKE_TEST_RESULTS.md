# 合成输入短训练实测记录

日期：2026-10-26。此记录来自实际 GPU 运行，使用本仓库内置 `runtime/`，不是模拟填表，也不是正式精度对比结果。

环境：WSL2 Ubuntu 26.04.1、RTX 5060 Laptop 8 GB、Python 3.11、PyTorch 2.7.1+cu128、CUDA Toolkit 12.8、GCC/G++ 13。详细安装命令见 [UBUNTU_SETUP.md](UBUNTU_SETUP.md)。

输入：256×256 合成图像及掩码，12 个训练视角、4 个测试视角，6000 个带法线的表面点，匹配的相机与 COLMAP/自定义点云文件。

| 编号 | 算法 | 本次实测结果 | 保存与读取 |
|---|---|---|---|
| 01 | 3DGS | 10 iterations 通过 | iteration_10 PLY 非空且参数有限 |
| 02 | 2DGS | 10 iterations 通过 | iteration_10 PLY 非空且参数有限 |
| 03 | Gaussian Surfels | 10 iterations 通过 | iteration_10 PLY 非空且参数有限 |
| 04 | SuGaR | **部分通过**：vanilla 10 iterations + coarse 10 次实际更新 | 原生 SuGaR checkpoint 已保存并读取；mesh/refine 未验证 |
| 05 | Mip-Splatting | 10 iterations 通过 | iteration_10 PLY 非空且参数有限 |
| 06 | Scaffold-GS | 10 iterations 通过 | PLY、opacity/cov/color/appearance 网络及训练断点均读取成功 |
| 07 | GaussianPro | 10 iterations 通过 | iteration_10 PLY 非空且参数有限；后期 propagation 未触发 |
| 08 | Octree-GS | 10 iterations 通过 | PLY、opacity/cov/color/appearance 网络及训练断点均读取成功 |
| 09 | Reduced-3DGS | 10 iterations 通过 | 按 vertex_0..3 原生分组读取 PLY，参数有限 |
| 10 | Concrete-GS | 10 iterations 通过 | 读取 12 个 mask，使用指定学习率与模式 F，保存 PLY 成功 |

汇总为 **9 passed / 1 partial_coarse / 0 failed**。包含 04 的全选命令返回退出码 1，以保留 mesh/refine 尚未验证的状态。

另外用正式 `main.py` 入口检查了 01、02、09 的 10 次短训练、数据格式适配和算法扩展隔离；02 的定期验证已在修复 Matplotlib 接口后通过。06、08 的断点方法还通过了真实 CPU 参数/Adam 的保存、恢复、继续优化一致性测试。

Ubuntu 中的自动化检查结果：`140 passed, 96 subtests passed`，包括数据格式适配、断点恢复、保存文件验证、子进程超时清理和扩展导入路径检查。

还将待发布 Git 清单独立导出到 Ubuntu 的新目录，在没有本机旧 `repo/`、数据和输出的情况下检查：`140 passed, 2 skipped, 94 subtests passed`。两项跳过涉及未随发布包携带的上游 Git 工作树补丁重放，运行源码检查仍然通过。

内置源码再次编译后，21 个扩展包检查均成功（7 个重新编译、14 个通过真实导入检查后复用），再执行整套短训练，结果仍为 9 passed / 1 partial_coarse / 0 failed。最后一次整套记录保存在本机 `output/synthetic_smoke/20261026_212747_060331/`，含命令、环境、导入路径、模型、日志、`report.json` 和各算法的 `result.json`。输出和模型不上传 GitHub；在目标机器按配置文档重新运行 `synthetic_smoke.py` 会生成自己的独立记录。

这次验证覆盖小输入的短训练与保存链路。30000 次训练、真实场景显存、后期增密/剪枝、传播阶段、SuGaR 网格阶段和最终论文精度仍需分别验证。部分上游实现最后一次 iteration 不做 optimizer.step，不能把 10 iterations 一概表述为 10 次参数更新；SuGaR coarse 的 10 次更新有独立计数和优化器状态证据。
