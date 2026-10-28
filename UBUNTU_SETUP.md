# Ubuntu 环境配置与复现说明

本文对应本仓库的合成输入短训练检查。仓库内置 10 个算法的训练源码和必要 CUDA 扩展源码，位于各算法的 `runtime/`。02～09 不再要求另外下载 `repo/`，修复已包含在内置源码中；上游来源、版本和打包信息见 `RUNTIME_SOURCES.json`。编译得到的 `.so`、Conda 环境、真实数据和训练模型不放进 GitHub，需要在目标电脑准备。

## 1. 已验证的环境与验证范围

本次实际运行环境是 **WSL2 Ubuntu 26.04.1、RTX 5060 Laptop 8 GB、Windows NVIDIA 驱动 581.80**。Python 与关键依赖组合如下。原生 Ubuntu 可按同样的 Python/CUDA 组合配置，但本次没有在另一台原生 Ubuntu 机器上重新测试。

| 组件 | 本次使用版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| CUDA Toolkit / nvcc | 12.8 |
| C/C++ 编译器 | conda-forge GCC/G++ 13 |
| NumPy | 1.26.4 |
| PyTorch3D | 0.7.9，仓库内置源码快照编译 |
| torch-scatter | 2.1.2+pt27cu128 |

其他 Python 依赖的固定版本在 `requirements-ubuntu.txt`。本次用的新版 Matplotlib 是 3.11.2，02 的相关兼容性修复已包括在源码里。不要把原作者不同年份的 `environment.yml` 混合安装到这个环境，也不要在安装完 CUDA 扩展后单独升级 Python/PyTorch。

短训练使用 12 张训练图、4 张测试图、256×256 合成图像和掩码、6000 个表面点。它检查真实训练、反向传播、保存以及保存文件的可读取性；不证明真实大场景、30000 次训练或最终精度。SuGaR 的 vanilla 初始化及 10 次 coarse 更新单独记录，mesh/refine 和较晚启用的表面损失不包含在这次短训练成功范围内。

## 2. 准备 Ubuntu 与 NVIDIA GPU

以下命令在 **Ubuntu 终端** 执行，不能直接贴到 Windows CMD。原生 Ubuntu 24.04 LTS 可以作为配置起点；本次实际验证的是上面的 WSL Ubuntu 26.04.1。

先运行：

```bash
nvidia-smi
```

必须能看到 NVIDIA GPU。WSL 中若命令不在 PATH，可试 `/usr/lib/wsl/lib/nvidia-smi`。WSL 使用 Windows 侧的 NVIDIA 驱动；原生 Ubuntu 使用 Linux 驱动。这一步都不成功时，先修复驱动/GPU 可见性，再安装 Python 包。

安装基础系统库：

```bash
sudo apt-get update
sudo apt-get install -y git wget ca-certificates build-essential cmake ninja-build \
  libgl1 libglib2.0-0 libusb-1.0-0 libglu1-mesa
```

`libusb-1.0-0`、`libglu1-mesa` 用于 Open3D/PyMeshLab 等库，本次已遇到并解决缺少这类动态库的问题。

## 3. 安装 CUDA Toolkit 12.8

先检查是否已有正确版本：

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

如果已经存在，不必重新安装。需要安装时，按系统选择下面 **一组** 命令。

**WSL Ubuntu：**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

**原生 Ubuntu 24.04：**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

原生 Ubuntu 22.04 对应仓库名是 `ubuntu2204`。其他发行版版本请根据 NVIDIA 官方安装指南选择，不要盲用 `ubuntu2404`。WSL 只安装这里的 toolkit 包，不在 WSL 内安装 Linux 显示驱动，也不把 `cuda`/`cuda-12-8` 全量驱动元包替代这里的 `cuda-toolkit-12-8`。[NVIDIA CUDA 12.8 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)、[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/archive/12.8.2/wsl-user-guide/index.html)说明了这两类安装方式。

`nvidia-smi` 中显示的 CUDA Version 是驱动能力信息，不能代替 `nvcc --version`，也不能代替 PyTorch 自己的 CUDA 构建版本检查。

## 4. 创建独立的 Python 环境

先安装 Linux 版 Miniconda/Anaconda，并确认 `conda --version` 能执行；Windows 中已经安装 Conda 不等于 Ubuntu 中也安装好了。如果尚未安装，可使用官方 Miniconda 安装器：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

按安装器提示初始化终端后重新打开 Ubuntu 终端。已有 Linux Conda 时直接继续，不要覆盖原环境。

创建本项目独立环境：

```bash
conda create -p "$HOME/3dgs-comparison-env" -y --override-channels -c conda-forge \
  python=3.11 pip gcc_linux-64=13 gxx_linux-64=13 ninja
conda activate "$HOME/3dgs-comparison-env"
python -m pip install setuptools==75.8.0 wheel
```

使用 GCC 13 是为了避免本次 Ubuntu 默认 GCC 15 与 CUDA 12.8 编译不兼容的问题。不要仅靠系统 `gcc --version` 判断后续扩展实际用了哪个编译器，下一节会显式指定。

## 5. 获取代码，安装 PyTorch 和 Python 依赖

建议放在 Ubuntu 用户主目录，使用简洁、不带空格的项目路径。WSL 的编译缓存也放在 Linux 文件系统里。

```bash
cd "$HOME"
git clone https://github.com/pawafa666-eng/3DGS-duibi.git
cd 3DGS-duibi
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-ubuntu.txt
python -m pip install --no-index --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html \
  'torch-scatter==2.1.2+pt27cu128'
```

这是 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)列出的 2.7.1/cu128 与 torchvision 0.22.1 组合。`torch-scatter` 的 wheel 页面虽然标为 `torch-2.7.0`，本次实际与 2.7.1+cu128 配合运行成功。

如果 `git clone` 提示仓库不存在或没有权限，先确认 GitHub 登录与仓库权限；这和 CUDA 环境无关。仓库已经含运行所需源码，不必再逐个克隆上游算法。

先验证真正的 CUDA 运算：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
x = torch.ones(8, device='cuda')
print('CUDA tensor sum:', (x * x).sum().item())
torch.cuda.synchronize()
PY
```

应看到 CUDA tensor sum 为 `8.0`。仅能 `import torch` 不够；缺少 `sm_120` 等架构支持时，往往是在真正执行 CUDA kernel 后才报错。

## 6. 设置编译环境，编译 PyTorch3D

仍在已激活的 `3dgs-comparison-env` 中：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')"
nvcc --version
"$CC" --version
```

本机 RTX 5060 Laptop 对应 `12.0`。不同 GPU 由上面的命令实际查询，不能把另一张卡的架构值直接复制过来。`MAX_JOBS=1` 限制并行编译内存，本次 WSL 仅约 7.4 GiB 内存也完成了编译，但耗时较长。

PyTorch3D 已固定为仓库内置的源代码快照，避免将来下载浮动 `main` 产生版本差异：

```bash
mkdir -p "$HOME/3dgs-comparison-builds/pytorch3d-source"
cp -a third_party/pytorch3d/. "$HOME/3dgs-comparison-builds/pytorch3d-source/"
FORCE_CUDA=1 python -m pip install --no-deps --no-build-isolation \
  "$HOME/3dgs-comparison-builds/pytorch3d-source"
```

验证实际 CUDA 算子：

```bash
python - <<'PY'
import torch
from pytorch3d.ops import knn_points
from torch_scatter import scatter_max
x = torch.rand(1, 32, 3, device='cuda')
print('PyTorch3D KNN:', knn_points(x, x, K=2).dists.shape)
print('torch-scatter:', scatter_max(torch.tensor([1., 3.], device='cuda'),
                                  torch.tensor([0, 0], device='cuda'))[0])
torch.cuda.synchronize()
PY
```

## 7. 分别编译各算法 CUDA 扩展

编译脚本现在优先识别当前环境的 `CUDA_HOME`、PATH 中的 `nvcc` 和已经安装的 toolkit，选择与 `torch.version.cuda` 匹配的版本，不再写死 CUDA 12.8。已有 PyTorch CUDA 11.8 且 `/usr/local/cuda-11.8/bin/nvcc` 存在时，可以直接使用现有工具链，不需要为了本脚本另外安装 12.8。

显式指定已有 CUDA 11.8：

```bash
python wsl_build_extensions.py --cuda /usr/local/cuda-11.8 --jobs 1
```

编译前会打印当前 Python、PyTorch CUDA 版本和实际选择的 nvcc 路径。找不到编译器或 CUDA 主版本不匹配时，在构建任何包之前停止，避免所有算法重复报相同错误。已有工具链的选择和编译不需要 Ubuntu 的 sudo 密码。

从项目根目录执行：

```bash
python wsl_build_extensions.py --jobs 1
```

也可只编译准备处理的算法，例如：

```bash
python wsl_build_extensions.py --only 02 06 --jobs 1
```

默认导入目录分别为 `~/3dgs-comparison-extensions/01`～`/10`，源码编译副本位于 `~/3dgs-comparison-builds/`。不同算法虽使用相同的 Python 包名，rasterizer 的参数和返回值并不相同，所以不能逐个安装进同一个 site-packages 覆盖彼此。

编译脚本按源码、PyTorch、GPU 架构、Python ABI 记录缓存；复用前还会实际导入已安装的扩展。编译错误详见 `output/wsl_build_logs/日期时间/`。更新 Python/PyTorch 或换 GPU 后应重新执行本步骤。

## 8. 先运行合成输入短训练

```bash
python synthetic_smoke.py --extension-root "$HOME/3dgs-comparison-extensions"
```

每个算法在独立子进程中依次运行，默认 10 iterations 后保存，终端显示 iteration 与 loss。单独复测：

```bash
python synthetic_smoke.py --only 06 08 --extension-root "$HOME/3dgs-comparison-extensions"
```

Windows 侧也可进入项目目录运行 `python .\run_wsl_smoke.py`，它调用 WSL，不要求 Windows 安装相同的 CUDA 训练依赖。

检查 `output/synthetic_smoke/日期时间/REPORT.md` 与 `report.json`。`passed` 要求训练正常退出、最终迭代模型存在且可读；06、08 还会读取具体的 MLP 权重和断点，核查优化器更新数。出现进度条并不代表通过。每次都使用全新的合成输入和输出目录。

代码自身的自动化检查可用 `python -m pytest tests -q` 运行，不会启动完整训练。本次实测摘要见 [SMOKE_TEST_RESULTS.md](SMOKE_TEST_RESULTS.md)。

**SuGaR 当前返回 `partial_coarse`。**它实际运行 vanilla 10 iterations 和 coarse 10 次优化更新并保存真实 checkpoint，但没有验证 mesh/refine。全选时因此会返回退出码 1，用于保留这个未验证事实，不能误读为另外 9 个算法都失败。

## 9. 准备真实数据并开始正式训练

数据根目录默认是 `data/地大隧道`，结构如下：

```text
data/地大隧道/
  train/images/          # 训练图像
  train/masks/           # 与图像匹配的掩码
  train/sparse/0/        # cameras.bin、images.bin、points3D.bin
  test/images/
  test/masks/
  test/sparse/0/
```

本项目含法线的自定义 `points3D.bin` 与标准 COLMAP 二进制格式不同。01、10 使用自定义格式；02～09 通过主控器建立独立兼容副本，不覆盖原始点云文件。不要只把不匹配格式的文件改成同样的文件名。若训练数据是标准 COLMAP 而非本项目的自定义数据，应先单独核对 01、10 的输入契约。

先查看命令计划：

```bash
python main.py --mode train --data 'data/地大隧道' --only 02 --dry-run
```

再按算法逐个训练、确认长训练阶段：

```bash
python main.py --mode train --data 'data/地大隧道' --only 02 --iterations 30000
```

`main.py` 在 Linux 中自动选择 `~/3dgs-comparison-extensions/<编号>`；也可以显式传入 `--extension-root "$HOME/3dgs-comparison-extensions"`。需要接着已有实验运行时，使用对应的 `--resume NN --only ...`，不要凭文档中的示例数字覆盖自己的输出目录。

Concrete-GS 保留本项目指定的 `position_lr_init=0.0000016`、`position_lr_final=0.00000016`、`rotation_lr=0.0000001`、`scaling_lr=0.001` 与实验模式 F。并非所有对比算法都原生在训练阶段使用 mask，具体策略见 `algorithms.json`。

## 10. 常见错误的处理位置

### 只训练时续跑，以及 01 的扩展混用

推荐使用一个入口，顺序完成源码检查、选中算法的独立扩展编译、训练：

```bash
python setup_and_train.py --cuda /usr/local/cuda-11.8 --resume 03 --only 03 04 06 08 --continue-on-error
```

这条命令针对已有 `output/03` 的情况，跳过已经训练或保存的 01、02、07、09、10，以及暂缓处理的 05；编译失败时不会启动训练。09 最后一步保存失败时可从已有量化模型恢复半精度文件，命令见 `09_Reduced_3DGS/README.md`；不要通过重新训练来恢复。请整行复制。选中的算法、数据目录和迭代参数会转发给正式主控入口，但 04 SuGaR 使用自己的分阶段迭代计划。省略 `--cuda` 时会自动识别 toolkit。

`main.py` 和每个算法的 `run.py` 默认只训练；主控入口也可显式使用 `--mode train`。此模式不启动 test 阶段，并关闭训练中的定期整组验证。06、08 额外跳过训练后自带的渲染/评估。训练本身需要的损失、梯度、增密和模型保存仍照常进行。

如果 01 提示 `GaussianRasterizationSettings` 缺少 `kernel_size`、`subpixel_offset`，说明当前导入的扩展接口与 01 不匹配，这两个字段属于 05 Mip-Splatting。不要给 01 补这两个参数来绕过错误；应把各算法扩展分别编译到自己的目录：

```bash
git pull origin main
python wsl_build_extensions.py --jobs 1
python main.py --mode train --resume 03 --continue-on-error \
  --extension-root "$HOME/3dgs-comparison-extensions"
```

仅修复 01 时可用 `python wsl_build_extensions.py --only 01 --jobs 1`，不会改动 02 的已训练模型。运行前会打印实际 rasterizer 路径，并核对它与算法内置接口及 Python 源码的一致性，发现混用时在训练前停止，给出重编译命令。

上述 `03` 对应已经有训练结果的 `output/03`，请按自己的目录选择。若某算法已训练完成，仅在后续 test 阶段失败，`--mode train --resume 03` 会复用训练结果；原来的失败状态保存在 `status_before_train_only.json`，不会重新训练，也不会修复/重跑测试。真正失败在 train 阶段的算法仍会重试。

| 现象 | 处理方法 |
|---|---|
| 缺少 `repo` | 此版本默认使用仓库内置 `runtime/`；先确认已同步完整的新版本和 `experiment.json` |
| `diff_*_rasterization` / `simple_knn` 缺失 | 在正确环境执行第 7 节，并核对对应编号的扩展目录 |
| 扩展从错误路径导入、函数参数不匹配 | 检查 `PYTHONPATH`；不要把不同算法的同名 rasterizer 混装 |
| `unsupported GNU version` | 检查 `CC`、`CXX`、`CUDAHOSTCXX` 是否指向 conda GCC 13 |
| `no kernel image` / 不支持 GPU 架构 | 核对 GPU、PyTorch CUDA 构建版本、nvcc 与 `TORCH_CUDA_ARCH_LIST`，重新编译 |
| CUDA 编译被 `Killed` | 使用 `--jobs 1`；检查 Linux/WSL 内存与 swap，避免同时编译多套扩展 |
| Open3D 缺少 libusb、PyMeshLab 缺少 GL 库 | 安装第 2 节系统动态库 |
| `points3D.bin` 溢出、回退后缺少 `points3D.txt` | 先核对自定义/标准二进制格式，通过主控器的兼容层转换 |
| 06、08 保存 `_local` 属性报错 | 使用本仓库内置修复版；旧的不完整 tuple 断点不能等价恢复 |
| 02 验证阶段 `tostring_rgb` 报错 | 同步内置 Matplotlib 修复；短训练与正式的定期验证都需检查 |
| `CUDA out of memory` | 合成数据通过不代表真实 500 万点可容纳；分别检查图像分辨率、初始点数量、增密阶段和当前显存 |

训练前保存自己的 `python -m pip freeze`、`nvcc --version`、GPU 型号、Git commit 与运行命令，便于复现实验。报告中的短训练 loss/PSNR 不是正式论文的精度结果。
