# 2D Gaussian Splatting for Robust Surface Reconstruction with LiDAR Point Cloud Geometry Prior

## Overview

​	本算法是基于3D Gaussian Splatting 改进的算法，主要改进点有以下几个方面：

- 引入**雷达点云初始化**代替SfM点云初始化；
- 使用**2D Gaussians**作为表示场景的基元；
- 引入雷达点云的**法向量先验**，将2D Gaussians与场景表面对齐；
- 针对有噪声的地面点云进行了基于平面网格的**点云重构**；

​	经过以上几个改进，增加了3DGS新视图合成的鲁棒性，有效减少新视图的伪影。

## Dependencies

​	这里仅记录了预处理模块能够正常运行时的所有的依赖：

| pkg / envoirment | version   |
| ---------------- | --------- |
| python3          | 3.8.10    |
| pyYAML           | 6.0.1     |
| scipy            | 1.10.1    |
| numpy            | 1.24.4    |
| opencv-python    | 4.10.0.84 |
| open3d           | 0.18.0    |

​	还有一个尚未解决的问题，就是目前的预处理模块能够正常运行的环境与3DGS原始算法要求的环境不一致，后面可能需要统一。

## Running

​	在训练之前，需要进行**数据预处理**。将输入数据转化为3DGS可读取的格式，才能开始训练。

### Preprocess

​	**数据预处理**是指将原始数据转化为3DGS的输入数据的过程。主要分5个方面的数据：位姿、图片、相机、点云、掩膜。

​	这部分的代码原本是以每个独立的脚本的形式存在的，后来把输入输出对接以后整合到了一起。

​	首先需要在`preprocess/config/dataset.yaml`中，修改数据集中**原始数据的路径**和一些数据预处理的**配置参数**。文件中默认的配置是以百度的`MKZ098_20220926144823`序列数据为例的。对于原始数据，使用**绝对路径**；对于预处理的输出数据，使用相对路径。配置参数完成后，在工作空间目录`gaussian-splatting`运行以下命令执行预处理：

```python
python3 preprocess/DataPreprocess.py
```

​	运行脚本后，如果配置参数正确、原始文件组织形式和`MKZ098_20220926144823`序列数据组织形式一致，应该不会出错。运行过程中有一个**区域增长**算法在完成后会弹出一个窗口，用来查看地面点分割效果，没问题的话点**叉**关闭即可自动进行后续流程。整个处理过程大约2分钟。

​	运行脚本过程中，会在每个子处理模块内生成`debug_file`，用来存放每个子任务的输出文件，报错后可通过查看过程文件定位debug。

​	完成以后，`data`目录应该具有以下文件：

```
├── images
│   ├── 1664*.jpg
│   ├──........
└── sparse
    └── 0
        ├── cameras.bin
        ├── images.bin
        ├── is_ground.txt
        └──points3D.bin
```

### Train

​	数据与处理完成以后，在工作空间目录`gaussian-splatting`运行以下命令，进行训练：

```python
conda activate gaussian_splatting
python train.py -s data -m data/output --mask data/masks --position_lr_init 0.0000016 --position_lr_final 0.00000016 --rotation_lr 0.0000001 --scaling_lr 0.001 --densify_from_iter 30001
```

​	以上命令，关闭了自适应密度控制功能，给非地面区域的高斯的位置、旋转、尺寸设置了比较小的学习率。

​	如果没有输入mask，只需在命令中去除mask参数即可。

​	改变场景规模可能要重新调试学习率。