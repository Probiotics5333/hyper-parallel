# DeepSeek-V4.1-Flash HyperMegaMhc 八卡验证记录

## 结论

DeepSeek-V4.1-Flash 的 single-pass mHC 正反向已替换为 multicore 的 `HyperMegaMhc` 和
`HyperMegaMhcGrad`，并在 8 张 Ascend 910B3 上完成 baseline 与 Hyper 各 100 step 的对比训练。
两条路径的 loss 均稳定下降；排除共享服务器在 Hyper 第 57 step 出现的一次 50.62 秒外部干扰后，
Hyper 的 100-step 总耗时减少 8.34 秒（1.50%），稳态中位 step time 减少 2.01%。Hyper 同时将
单卡峰值 allocated / reserved 显存分别降低 18.49% / 16.07%。

本次验证参考了
[DeepSeek-V4.1-Flash 官方训练文章](https://www.mindspore.cn/technology-blogs/zh/2026-9-13)，
但服务器只有 8 卡，因此并行规模和数据口径与文章中的 16 卡实验不同，绝对 loss 和 step time 不能直接横向比较。

## 环境和配置

| 项目 | 本次八卡验证 | 官方文章 |
| --- | --- | --- |
| 硬件 | 8 × Ascend 910B3，64 GiB | Atlas 800T A3，16 个 NPU die |
| CANN | 9.1 | 8.5.1 |
| PyTorch / torch-npu | 2.9 / 2.9 | 2.9 / 2.9 |
| Transformers | 5.13 | 5.13 |
| 并行策略 | TP1、CP1、EP8、FSDP8、PP1 | TP1、CP1、EP16、FSDP16、PP1 |
| Batch | micro 1、global 8 | micro 1、global 16 |
| 序列长度 | 4096 | 4096 |
| 模型裁剪 | 4 个 LLM layer、1 个 vision layer、16 个 routed expert | 文章使用的裁剪配置 |
| 数据 | 512 条确定性 Online VLM JSONL | 512 条 Online JSONL |
| 精度 | BF16 参数/前向、FP32 梯度通信与优化器 | 相同 |

本次每个 global step 约有 168 个有效监督 token，且 512 条确定性样本会被重复训练，因此 loss 会比官方
文章更快下降。这能验证替换前后的数值稳定性和相对性能，不能代表完整预训练任务的收敛曲线。

服务器工作树为 `/home/zxl/hyper-parallel-dsv41-mhc`，Python 环境为
`/home/zxl/venvs/dsv41_mhc`。自定义算子使用
`/home/zxl/omni_training_custom_ops_cann9`，模型和数据资产位于
`/home/zxl/DeepSeek-V4.1-Flash-assets`。

## 100-step 结果

| 指标 | single-pass baseline | HyperMegaMhc | 变化 |
| --- | ---: | ---: | ---: |
| 完成 step | 100 / 100 | 100 / 100 | 均通过 |
| 原始 100-step 总耗时 | 556.3881 s | 593.2712 s | +6.63% |
| 去除第 57 step 外部干扰后的总耗时 | 556.3881 s | 548.0499 s | **-1.50%** |
| step 1～99 中位耗时 | 5.5055 s | 5.3949 s | **-2.01%** |
| 前 10 step 平均 loss | 10.343153 | 10.340477 | 一致 |
| 后 10 step 平均 loss | 1.426907e-5 | 1.038420e-5 | 均正确下降 |
| 单卡峰值 allocated | 40.0500 GiB | 32.6437 GiB | **-18.49%** |
| 单卡峰值 reserved | 45.2520 GiB | 37.9824 GiB | **-16.07%** |

Hyper 原始总耗时中，第 57 step 为 50.6162 秒；同一时间另一位用户启动了占用 8 卡的任务。用其余
99 个 step 的中位数替换这一个异常点后得到 548.0499 秒。原始值和校正值同时保留，避免把共享服务器
争用误报为内核性能变化。受设备持续被外部任务占用影响，最终 clean build 后未能再做一轮无干扰 100-step
复测，所以当前性能收益应视为约 1.5%～2.0% 的初步结果，而不是最终基准结论。

上述 100-step 数据来自修正设备核数前的版本：调度图生成了 48 个逻辑初始化任务，由 910B3 的 40 个
物理 AIV round-robin 执行，其中前 8 个 AIV 各执行两个分片。该版本功能和 loss 验证通过，但初始化负载
不均衡。当前实现已改为读取设备实际 `vector_core_num`，910B3 上生成 40 个等分任务；动态 40-task 版本的
调度 UT 已通过，尚需在设备空闲后补跑原生反向和 100-step 性能，因此本表不能用于比较 40-task 修正前后的收益。

服务器日志：

- baseline：`output/training_demo/deepseek_v41/run_baseline_100step.log`
- Hyper：`output/training_demo/deepseek_v41/run_hyper_100step.log`
- clean build 后设备占用诊断：`output/training_demo/deepseek_v41/run_hyper_cleanbuild_1step.log`

## 接入和修复

模型侧沿用 `HyperMegaMoe` 的模块替换方式：在 DeepSeek-V4.1 decoder layer 中创建 Hyper mHC 模块，
训练时由 autograd 路径调用 `HyperMegaMhcGrad`，而 baseline 保留原 single-pass `PipelinedMhcModule`，从而
可以在相同模型、数据、并行策略和随机种子下切换对比。

本轮发现并处理的问题如下。

| 问题 | 根因 | 处理 |
| --- | --- | --- |
| Hyper 第二次 backward 报 507015 / buffer 越界 | 一个 AIV 通过 `InitOutput<float>` 清零完整 `gradPhi`，4096×5120 配置下约 1.875 MiB，超过内部标量清零缓冲能力 | 调度图按设备实际 `vector_core_num` 生成初始化任务；910B3 上为 40 个。每个 Vector worker 按 32-byte block 分片，并用 1 KiB VECCALC 零块循环 DataCopy |
| mHC/Engram 边界出现 FP32 状态被转为 BF16 | FSDP `output_dtype: bfloat16` 会转换 Hyper 系数状态 | 两个 DeepSeek-V4.1 recipe 将 `output_dtype` 设为 `null` |
| 部分环境无法解析 `torchrun` 可执行文件 | 虚拟环境脚本入口解析不稳定 | 启动脚本改用 `python -m torch.distributed.run` |
| CANN 8.5.1 无法构建当前 multicore SHMEM/算子 | 移植代码依赖较新的原生构建接口 | 服务器验证环境使用 CANN 9.1；未把环境差异伪装成源码修复 |
| clean build 后启动报 507033 / TsdOpen failed | 共享服务器已有其他用户任务占用部分设备 | 未结束他人任务；记录诊断并停止重复抢占设备 |
| 同进程同时保留两个约 616 MiB 对比快照时退出报 double free | 诊断进程内存/分配器压力；两个快照隔离执行均正常退出 | 将 fused 和 semantic reference 分进程验证，不作为内核正确性失败 |
| 辅助 ST 的 forward-cache 构造报 507057 | `_make_case` 的 forward cache 诊断路径与当前 NPU 环境不兼容 | 反向问题改用真实训练输入快照缩小；该辅助路径不纳入本次性能结论 |

## 缩小复现和验证

从真实训练 rank 0 的第二次 backward 捕获了 21 个原生输入，形成约 616 MiB 的
`/home/zxl/hyper_grad_second_inputs_actual.pt`。分阶段执行后确认只有 reduction-output 初始化阶段失败，
RMSNormGrad 阶段可独立通过。修复后的验证包括：

- 根因修复时，48-logical-task zero-only 阶段及 48 次 init event trigger 均通过；
- 设备核数修正后，910B3 的 40-task 调度图、任务索引、split 数量和 40 次事件触发 CPU UT 均通过；
- `48` 仅保留为 runtime/profiler 的最大容量，不再作为实际初始化任务数；
- 真实 4096×5120 RMS-only 阶段通过；
- 完整真实反向快照通过；
- 恢复原有 RMS double-buffer 后完整快照仍通过，证明无需临时 single-buffer 绕过；
- clean native build 成功；
- 新增 CPU 单测固定初始化任务数量、索引、split 数量和事件触发次数。

随机非零输入与语义 reference 对比时，主要输出 cosine similarity 为 0.99999285～1.0；相对 L2 误差为
3.87e-6～3.78e-3，符合 BF16 融合反向路径的预期。真实训练快照的主要输出 cosine similarity 为
0.999996～0.999999。

## 后续建议

在服务器空闲时，用相同 wheel、数据、随机种子各重跑 baseline 和 Hyper 100 step，至少重复三次并报告
P50/P95、去除首 step 的平均值及总耗时置信区间。当前结果已经证明接入正确、loss 可下降且显存收益明确，
但约 2% 的耗时收益较小，仍需要独占设备复测后再作为正式性能数字。
