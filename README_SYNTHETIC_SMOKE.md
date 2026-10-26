# 用合成输入检查短训练与保存

在项目目录的 Windows 终端运行：

```powershell
python .\run_wsl_smoke.py
```

入口使用本机 WSL `Ubuntu` 中的 `~/3dgs-comparison-env/bin/python`，依次检查全部 10 个算法。终端实时显示原算法的 iteration、loss 和错误信息。每次产生独立的 `output/synthetic_smoke/日期时间/`，不会读取真实数据，也不会覆盖原来的训练结果。

只复测指定算法：

```powershell
python .\run_wsl_smoke.py --only 02 06 08
```

修改了 CUDA 扩展源码后先编译，再短训练：

```powershell
python .\run_wsl_smoke.py --build
```

这不是环境安装器。本机已经配置了下面的环境；换电脑仍需先安装对应依赖。`--build` 只编译项目中的扩展。

## 输入与判定

`synthetic_smoke.py` 生成 256×256 的纹理立方体图像、有效区域掩码、6000 个带法线的表面点、匹配的相机，以及标准 COLMAP 和本项目自定义二进制两种点云格式。训练集有 12 个视角，测试集有 4 个视角。03 还会得到合成法线和深度文件。

01、02、03、05、06、07、08、09、10 调用各自真实训练代码，默认运行 10 iterations，要求正常退出，保存 `iteration_10/point_cloud.ply`，并成功读取非空、有限数值的模型参数。06、08 同时保存网络权重和可恢复的训练断点；09 按原生的 `vertex_0..3` 分组格式读取。原算法中标为 `metrics_only` 的方法不使用训练 mask，报告保留这个区别。

04（SuGaR）分阶段检查：官方 vanilla 初始化 10 iterations，再进行 10 次真实 coarse 优化更新并保存、读取 SuGaR checkpoint。结果记为 `partial_coarse`，**不记为完整算法通过**。原算法的表面正则在较晚迭代才启用，mesh/refine 依赖已经收敛的 coarse 模型，无法由这 10 次训练证明可用。因此全选运行即使其余算法通过，仍会返回非零退出码以提示这个未完成阶段。

成功只说明本机小输入的短训练和保存链路可运行。它不能证明 30000 次训练、真实 500 万点场景的显存、后期增密/剪枝、GaussianPro propagation、SuGaR 网格阶段或精度指标都已验证。部分上游实现最后一次 iteration 只计算、反传和保存，不执行 optimizer.step；10 iterations 与 10 次参数更新有所区别，SuGaR coarse 单独记录了实际更新数。

每次输出包含 `REPORT.md`、`report.json`、`environment.json`、合成数据、各算法的 `commands.json`、原始日志、模型和 `result.json`。仅出现 `Training progress: 0%` 或 `Training complete` 不会被判定成功。

## 本机 WSL 环境

- Ubuntu，Python 3.11：`/home/pwf/3dgs-comparison-env`。
- PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、CUDA Toolkit 12.8、conda GCC/G++ 13。
- RTX 5060 Laptop，扩展针对本机架构编译。
- PyTorch3D 0.7.9 从官方源码编译；torch-scatter 使用匹配 PyTorch 2.7/cu128 的包。
- 各算法扩展分别位于 `/home/pwf/3dgs-comparison-extensions/01` 到 `/10`。不要把不同算法的同名 rasterizer 混装到同一个路径。
- 编译缓存放在 `/home/pwf/3dgs-comparison-builds`，避免在 Windows 挂载盘内编译。
- 依赖清单保存在 `output/synthetic_smoke/wsl-pip-freeze.txt` 和 `wsl-conda-explicit.txt`，编译日志在 `output/wsl_build_logs/`。

直接在 Ubuntu 内运行：

```bash
cd '/mnt/d/AAA-研究生/1. 研究/2 改进的3DGS/全部实验/04 对比实验'
~/3dgs-comparison-env/bin/python synthetic_smoke.py --extension-root ~/3dgs-comparison-extensions
```

正式入口 `main.py` 在 Linux 中会自动使用 `~/3dgs-comparison-extensions/<算法编号>`，也可用 `--extension-root` 或 `GS_EXTENSION_ROOT` 指定其他根目录。它会为各算法子进程分别设置导入路径；正式数据、迭代计划和短训练结果相互独立。正式多阶段训练仍须另外验证，尤其是 SuGaR。

## 修复如何在重新下载源码后保留

全部 10 个算法的必要源码已内置到各自 `runtime/`，修复已经应用；`patches/` 和 `RUNTIME_SOURCES.json` 保留来源与修复记录。`setup_repositories.py` 会检查内置入口，不再要求另外克隆 `repo/`。Ubuntu 的完整配置命令见 `UBUNTU_SETUP.md`。
