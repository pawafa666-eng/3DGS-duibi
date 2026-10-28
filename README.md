# 04 对比实验：10 个 3DGS 方法统一运行框架

本目录把 10 个方法按同一数据集、同一训练迭代参数和同一输出协议组织起来。输入数据只读取：

```text
data/地大隧道/
├── train/images, masks, sparse/0/
└── test/images, masks, sparse/0/
```

## 方法清单

| 编号 | 方法 | 主要对比点 | mask 策略 |
|---|---|---|---|
| 01 | 3DGS（随仓库附带的可训练版本） | 与本数据集自定义 `points3D.bin` 格式匹配 | metrics_only |
| 02 | [2DGS](https://github.com/hbb1/2d-gaussian-splatting) | 2D 表面基元 | native |
| 03 | [Gaussian Surfels](https://github.com/turandai/gaussian_surfels) | 表面重建 | native |
| 04 | [SuGaR](https://github.com/Anttwo/SuGaR) | 表面对齐与网格 | metrics_only |
| 05 | [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) | 抗混叠 | metrics_only |
| 06 | [Scaffold-GS](https://github.com/city-super/Scaffold-GS) | anchor 结构化高斯 | metrics_only |
| 07 | [GaussianPro](https://github.com/kcheng1021/GaussianPro) | 渐进传播 | metrics_only |
| 08 | [Octree-GS](https://github.com/city-super/Octree-GS) | LOD 层次化表示 | metrics_only |
| 09 | [Reduced-3DGS](https://github.com/graphdeco-inria/reduced-3dgs) | 轻量化表示 | metrics_only |
| 10 | Concrete-GS（随仓库附带的可训练版本） | 本文方法 | native |

每个方法都有独立目录 `NN_算法/`，包括 `experiment.json`、`run.py`、说明文档和内置源码目录 `runtime/`。10 个方法的必要训练源码、CUDA 源码和本项目修复均随仓库分发；来源和上游版本见 `RUNTIME_SOURCES.json`，总清单是 `algorithms.json`。

**Ubuntu 安装与复现请先阅读 [UBUNTU_SETUP.md](UBUNTU_SETUP.md)**，包括驱动、CUDA、Conda、固定依赖、PyTorch3D 编译、算法扩展隔离、10 次迭代检查和正式训练命令。合成输入检查的范围见 [README_SYNTHETIC_SMOKE.md](README_SYNTHETIC_SMOKE.md)。

本次实际结果：9 个算法完成短训练与保存检查；SuGaR 为 coarse 部分通过，完整网格/refine 未验证。见 [SMOKE_TEST_RESULTS.md](SMOKE_TEST_RESULTS.md)。

## 第一次准备源码

已有 Ubuntu 环境时，可以使用 `python setup_and_train.py --resume 03 --only 03 04 06 08 --continue-on-error`，先准备选中算法的扩展再训练。示例跳过已经训练或保存的 01、02、07、09、10，以及暂缓的 05。CUDA 路径会自动识别，也可通过 `--cuda /usr/local/cuda-11.8` 显式指定；准备失败不会进入训练。

```powershell
git pull origin main
python .\setup_repositories.py
```

在当前 Conda 环境运行即可，补全脚本只依赖 Python 标准库和 Git，不需要先导入 PyTorch。也可以双击 `setup_repositories.bat`；原来的 `setup_repositories.ps1` 入口保留，调用同一脚本。

此版本的 10 个方法均使用仓库内置 `runtime/`，脚本检查源码是否齐全；正常克隆完整仓库后不需要再下载 02～09 的 `repo/`。必要修复已经应用，`patches/` 另存修复记录。脚本仍兼容旧版外部仓库清单，但不会用官方源码覆盖本项目内置实现。

下载时显示 Git 进度，每条网络命令默认尝试 3 次；单个方法失败会继续处理后面的算法，最后返回非零退出码并列出失败编号。看到 `Source setup: 10/10 ready` 后再启动训练。报告在 `output/.repository_setup/report.json`，每个算法的来源和 commit 另存到 `source_commit.txt`。

可重复运行同一命令：保留已有源码和修改，不会自动拉取或重置已有版本；下载中断的暂存目录及已下载的子模块会继续使用。非空但不完整的未知目录、补丁冲突会明确报错，不会覆盖文件。只重试部分算法也可以：

```powershell
python .\setup_repositories.py --only 02 03 04 05 06 07 08 09
```

本 GitHub 仓库保存运行框架、全部 10 个方法的必要源码、兼容补丁和 PyTorch3D 源码快照，不保存 `data/`、训练结果、Conda 环境或编译二进制。补全脚本不改数据、不启动训练，也不安装 Python 包或编译 CUDA 扩展；按 Ubuntu 配置文档准备环境后，扩展分别编译到编号目录，避免同名 rasterizer 相互覆盖。

02--09 使用的当前数据是自定义 `points3D.bin`，总控运行时会在 `output/.dataset_compat/` 建立隔离的标准 COLMAP 兼容视图：复制 `cameras.bin`、`images.bin`，把自定义点记录转换为标准格式，并通过 junction 复用 images/masks。源数据不会被改写，兼容缓存按数据文件指纹复用。01 和 10 继续直接使用原始自定义格式。

01 的运行源码位于 `01_3DGS/runtime`，包含与当前数据匹配的自定义 `points3D.bin` 解析器：每个点记录包含坐标、颜色、法向量、误差和轨迹长度，另读取同目录的 `is_ground.txt`。因此不要把 01 改回官方 `repo`，否则会再次出现 `OverflowError`。

## 运行全部 10 个实验

默认只训练并保存模型，不进入 test 阶段，也不执行训练中的定期整组验证。大多数方法默认 30000 次迭代；04 SuGaR 使用自身的分阶段训练计划，详见其说明：

```powershell
python .\main.py --mode train
```

如果只想快速检查各算法环境，不想等待完整训练，可以运行：

```powershell
python .\smoke_train.py
```

该脚本会逐个启动算法训练，检测到真实的 Training progress 或迭代输出后立即停止当前进程，再启动下一个算法。结果和日志写入 output/smoke/；它只检查训练能否真正进入迭代，不执行完整测试，也不等待 30000 次迭代。只检查指定算法时，例如：

```powershell
python .\smoke_train.py --only 02 03 04
```

运行时总控会实时转发每个方法的训练、测试子进程输出，因此可以直接看到算法自身的 `iteration` 或 `tqdm` 进度条。每个阶段会显示方法编号和 `train/test` 标签；如果连续约 30 秒没有输出，会提示该阶段仍在运行。完整原始输出仍分别保存在对应目录的 `train_stdout.log`、`train_stderr.log`、`test_stdout.log` 和 `test_stderr.log` 中。没有原生迭代进度输出的方法只能显示运行心跳，不会伪造 iteration。

也可以双击或命令行调用：

```powershell
.\run_all.bat
```

只运行指定方法：

```powershell
python .\main.py --only 02 05 10
python .\main.py --mode train --only 01 02 10 --iterations 30000
python .\main.py --mode test --resume 01 --only 02
```

## 断点续跑

如果 `output/05` 中 01、10 已训练完成，02--09 因缺少源码而 blocked，补全源码后执行：

```powershell
python .\main.py --mode train --resume 05 --only 02 03 04 05 06 07 08 09 --continue-on-error
```

这只训练剩余 8 个算法，不重训 01/10，也不运行测试。

新运行自动使用 `output/01`、`output/02`、`output/03` 等目录，不覆盖旧结果。比如 `output/03` 跑到第 4 个方法失败后，直接执行：

```powershell
python .\main.py 03
```

或者：

```powershell
python .\main.py --resume 03
```

恢复逻辑只跳过同时满足以下条件的方法：`status.json` 为 `completed`，并且其中列出的 artifacts 全部存在。`failed`、`blocked`、`planned` 或缺产物的方法会重新执行。默认遇到第一个错误停止并保留现场；若希望继续尝试其它方法：

```powershell
python .\main.py --resume 03 --continue-on-error
```

正式运行前查看命令而不启动训练：

```powershell
python .\main.py --dry-run
```

## 输出结构

```text
output/03/
├── run_config.json
├── resume_config.json       # 仅断点运行时生成
├── summary.csv
├── summary.json
├── selected_experiments.json
└── 02_2DGS/
    ├── command.txt
    ├── status.json
    ├── metrics.json
    ├── train_stdout.log
    ├── train_stderr.log
    ├── test_stdout.log
    ├── test_stderr.log
    ├── train/
    └── test/
```

`summary.csv` 不会补造指标；没有成功生成的 PSNR、SSIM、LPIPS、训练时间、FPS、高斯数量或模型大小保持为空。`metrics_only` 表示上游训练代码未原生读取 mask，统一评估阶段仍记录 mask 策略，不能直接当作与 native mask 方法完全等价的正式结果。

## 效率与资源统计

每个实验的 `metrics.json` 以及总 `summary.csv`/`summary.json` 还会记录：

- `training_time_sec`：训练阶段总墙钟时间。
- `avg_iteration_time_sec`：训练总时间除以配置迭代数；没有单一 `--iterations` 参数的多阶段算法留空。
- `render_fps`：渲染命令期间实际新增或修改的渲染 PNG 数除以渲染耗时。
- `peak_gpu_memory_mb`：训练和测试子进程及其子进程的峰值显存，依赖 `nvidia-smi`；不可用时保留空值并写原因。
- `gaussian_count`：最终模型 PLY 头部的 `element vertex` 数量。
- `model_file`、`model_size_bytes`、`model_size_mb`：最终模型文件路径和大小。

失败时，`status.json` 会保存 `stderr_tail`，主控器也会直接显示错误尾部。断点运行不会覆盖原始 `run_config.json`，新的参数写入 `resume_config.json`。

## WSL 合成数据短训练检查

本机已配置 WSL Ubuntu 环境，可运行 `python .\run_wsl_smoke.py`，依次用合成图片、掩码和点云检查真实训练及保存。详见 [README_SYNTHETIC_SMOKE.md](README_SYNTHETIC_SMOKE.md)。04 的 coarse 检查与完整 mesh/refine 流程分别记录，短训练不代表完整 30000 次训练已验证。

## 单独运行一个方法

每个实验目录都可以独立调用：

```powershell
python .\02_2DGS\run.py --stage train --data data\地大隧道
```

单独入口会创建新的运行目录；需要接着已有运行时传入 `--run-dir output\03`。02--09 的独立入口也会使用同一套标准 COLMAP 兼容缓存；`--dry-run` 只生成命令计划，不会转换 500 万点文件。
