# 10 Concrete-GS

本文 Concrete-GS 完整方法。源码位于仓库内的 `runtime/`，主控使用 `--experiment_mode F`，并显式传入 train/test masks 及本文使用的学习率参数。

训练参数固定为：`position_lr_init=0.0000016`、`position_lr_final=0.00000016`、`rotation_lr=0.0000001`、`scaling_lr=0.001`。源码同时兼容 `--mask` 和 `--masks` 两种写法。
