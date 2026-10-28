# 04 SuGaR

SuGaR 表面对齐与网格重建方法，源码来自 `Anttwo/SuGaR`。官方流程是多阶段 pipeline，且 README 明确提示 Windows 兼容性和 mask 功能仍有限；本框架会保留实际失败状态，不把它伪装成已完成结果。

只训练模式保留完整算法流程：先训练 7000 次 vanilla 3DGS，再完成 coarse（终止迭代 15000）、网格提取和 2000 次 refined。`main.py --iterations` 不改变这套分阶段计划；网格提取是算法的一部分。训练时关闭数据验证划分及 vanilla 定期测试。阶段产物位于 `output/<运行号>/04_SuGaR/train/attempt_<标识>/`，完整成功时生成 `pipeline_results.json`。每次重试使用新的 attempt 目录。

实测覆盖空格路径和阶段失败传播的自动化检查，以及合成数据上 vanilla 10 次和 coarse 10 次短训练；完整真实数据的网格和 refined 阶段尚未验证，依赖需在运行环境单独准备。

