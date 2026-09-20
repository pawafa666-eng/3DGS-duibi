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

每个方法都有独立目录 `NN_算法/`，包括 `experiment.json`、`run.py`、说明文档和源码目录。01 和 10 的源码目录是仓库内的 `runtime/`，其余外部方法默认使用 `repo/`。`algorithms.json` 是总清单。

## 第一次准备源码

```powershell
git pull origin main
python .\setup_repositories.py
```

在当前 Conda 环境运行即可，补全脚本只依赖 Python 标准库和 Git，不需要先导入 PyTorch。也可以双击 `setup_repositories.bat`；原来的 `setup_repositories.ps1` 入口保留，调用同一脚本。

脚本检查全部 10 个方法：01、10 使用本仓库自带的 `runtime/`；02--09 缺少 `repo/` 时自动下载官方源码，并递归补齐未初始化的 Git 子模块（包括 CUDA 扩展源码）。02、03 新下载的源码固定到清单中的版本，并应用随仓库分发的 mask 接口补丁，避免现有命令的 `--masks` 参数不被识别。无需依赖本机的其它实验目录。

下载时显示 Git 进度，每条网络命令默认尝试 3 次；单个方法失败会继续处理后面的算法，最后返回非零退出码并列出失败编号。看到 `Source setup: 10/10 ready` 后再启动训练。报告在 `output/.repository_setup/report.json`，每个算法的来源和 commit 另存到 `source_commit.txt`。

可重复运行同一命令：保留已有源码和修改，不会自动拉取或重置已有版本；下载中断的暂存目录及已下载的子模块会继续使用。非空但不完整的未知目录、补丁冲突会明确报错，不会覆盖文件。只重试部分算法也可以：

```powershell
python .\setup_repositories.py --only 02 03 04 05 06 07 08 09
```

本 GitHub 仓库保存运行框架、01/10 必要源码和兼容补丁，不保存 `data/`、训练结果、02--09 外部仓库及下载缓存。补全脚本不改数据、不启动训练、也不安装 Python 包或编译 CUDA 扩展；各方法仍需按其 README 准备匹配的环境，不保证它们能共用同一个 `3dgs` 环境。

02--09 使用的当前数据是自定义 `points3D.bin`，总控运行时会在 `output/.dataset_compat/` 建立隔离的标准 COLMAP 兼容视图：复制 `cameras.bin`、`images.bin`，把自定义点记录转换为标准格式，并通过 junction 复用 images/masks。源数据不会被改写，兼容缓存按数据文件指纹复用。01 和 10 继续直接使用原始自定义格式。

01 的运行源码位于 `01_3DGS/runtime`，包含与当前数据匹配的自定义 `points3D.bin` 解析器：每个点记录包含坐标、颜色、法向量、误差和轨迹长度，另读取同目录的 `is_ground.txt`。因此不要把 01 改回官方 `repo`，否则会再次出现 `OverflowError`。

## 运行全部 10 个实验

默认训练再测试，每个实验 30000 次迭代：

```powershell
python .\main.py --mode both
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

## 单独运行一个方法

每个实验目录都可以独立调用：

```powershell
python .\02_2DGS\run.py --stage both --data data\地大隧道
```

单独入口会创建新的运行目录；需要接着已有运行时传入 `--run-dir output\03`。02--09 的独立入口也会使用同一套标准 COLMAP 兼容缓存；`--dry-run` 只生成命令计划，不会转换 500 万点文件。
