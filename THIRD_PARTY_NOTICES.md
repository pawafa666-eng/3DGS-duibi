# 上游源码与许可证

本仓库为了在另一台 Ubuntu 电脑直接获得相同的已修复源码，内置了各方法运行所需的源文件和 CUDA 扩展源文件。各上游作者仍保留原有版权和许可条件；本项目的打包、兼容补丁与环境说明不改变这些条件，也不把研究用途代码重新许可为可任意商用。

| 目录 | 来源 |
|---|---|
| `01_3DGS/runtime` | 本项目使用的自定义 3DGS 版本，基于 GraphDeco Gaussian Splatting |
| `02_2DGS/runtime` | https://github.com/hbb1/2d-gaussian-splatting |
| `03_Gaussian_Surfels/runtime` | https://github.com/turandai/gaussian_surfels |
| `04_SuGaR/runtime` | https://github.com/Anttwo/SuGaR |
| `05_Mip_Splatting/runtime` | https://github.com/autonomousvision/mip-splatting |
| `06_Scaffold_GS/runtime` | https://github.com/city-super/Scaffold-GS |
| `07_GaussianPro/runtime` | https://github.com/kcheng1021/GaussianPro |
| `08_Octree_GS/runtime` | https://github.com/city-super/Octree-GS |
| `09_Reduced_3DGS/runtime` | https://github.com/graphdeco-inria/reduced-3dgs |
| `10_Concrete_GS/runtime` | 本项目使用的 Concrete-GS 版本，保留基础代码许可与作者标注 |
| `third_party/pytorch3d` | https://github.com/facebookresearch/pytorch3d |

上游版本与已应用补丁在 `algorithms.json` 和 `RUNTIME_SOURCES.json` 中记录。源码清单摘要对排序后的相对路径和 Git blob ID 计算 SHA256，覆盖实际选入仓库的源文件，不受目标系统工作树换行风格影响；PyTorch3D 原始下载归档的 SHA256 单独记录。

各 `runtime/` 中保留了上游 `LICENSE`/`LICENSE.md`、源文件版权头，以及 CUDA 子模块和 GLM 的许可文件。PyTorch3D 的许可和第三方声明在 `third_party/pytorch3d/LICENSE` 与 `LICENSE-3RD-PARTY`。少量 LPIPS 校准权重是上游运行资源；大型预训练 AlexNet/VGG 权重仍由对应库首次运行时下载，未随本项目打包。

未内置 Git 历史、SIBR 可视化器、论文展示素材、上游测试/示例数据、真实研究数据、训练模型和编译产物。若需要上游完整开发仓库、查看器或原论文完整资产，请使用上表链接。科研使用时仍应按原方法要求引用对应论文。
