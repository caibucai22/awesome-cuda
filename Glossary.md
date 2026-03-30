## LLM

| Noun       | Explain |
| ---------- | ------- |
| KV-Cache   | (Key-Value Cache) 注意力机制的键值缓存，存储过去 token 的 K/V 以复用，显著加速自回归生成 |
| prefilling | 预填充阶段，处理用户输入的全部 token，生成初始 KV-Cache |
| decoding   | 解码阶段，逐 token 生成，每次只需计算当前 token 的 Q 并与 KV-Cache 交互 |
| SDPA       | (scaled dot product attention，缩放点积注意力) Transformer 标准注意力公式 |
| MHA        | (multi-head attention，多头注意力) 标准多头，每组 Q/K/V 有独立投影 |
| MQA        | (multi-query-attention，多查询注意力) 所有头共享 K/V，减少内存带宽 |
| GQA        | (grouped-query-attention，分组查询注意力) MHA 与 MQA 折中，K/V 分组共享 |
| flash_attn/FlashAttention | 通过分块计算和重计算，将注意力 I/O 复杂度从 O(N²) 降至 O(N²/√d) |
| FMHA       | (Fused Multi-Head Attention, 融合多头注意力) 算子层面融合，减少多次内核启动开销 |
| FMHCA      | (Fused Multi-Head Cross-Attention, 融合多头交叉注意力) 交叉注意力融合版本 |
| MLA        | (multi-head latent attention，多头潜在注意力) DeepSeek-V2/V3 使用，低维潜在向量代替原始 K/V |

## Hardware

| Hardware | Explain |
| -------- | ------- |
| SRAM     | (static random access memory，静态随机存储器) GPU 片上缓存，带宽 > 1TB/s，容量 ~200KB/SM |
| HBM      | (High Bandwidth Memory) GPU 显存，带宽 500~1200GB/s，容量 16~80GB |
| TFLOPS   | (Tera Floating-Point Operations Per Second) 每秒万亿次浮点运算 |
| SM       | (Streaming Multiprocessor) GPU 计算核心，每个 SM 包含多个 CUDA Core、Tensor Core、SRAM |
| Tensor Core | Volta+ 架构专用矩阵单元，支持 WMMA/MMA 指令，INT8/FP16/BF16/FP8 矩阵累加 |
| Ampere   | NVIDIA GPU 架构（2020），支持 TF32、稀疏 Tensor Core、MIG |
| Hopper  | NVIDIA GPU 架构（2022），支持 Transformer Engine、FP8、DPX 指令 |
| Ada Lovelace | NVIDIA 消费级架构（2022），基于 Hopper 微架构 |

## Optimization Techniques

| Term | Explain |
| ---- | ------- |
| Bank Conflict | 共享内存 bank 冲突，多个线程同时访问同一 bank 导致串行化 |
| Memory Coalescing | 内存合并访问，相邻线程访问连续内存地址，最大化内存带宽 |
| Occupancy | SM 上活跃 warp 比例，高 occupancy 可隐藏延迟但不一定提升性能 |
| Divergence | 线程分支发散，warp 内线程执行不同路径导致串行执行 |
| Tile | 分块策略，将大问题分解为适合 cache/SRAM 的小块 |
| Vectorized Load/Store | 向量化访存，一次加载多个元素（float4、half2） |
| Kernel Fusion | 内核融合，多个操作合并为单内核，减少 global memory 读写 |
| LDS | (Local Data Share / Shared Memory) 线程块内共享内存，用于数据复用 |
