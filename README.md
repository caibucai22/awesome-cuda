<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/downloads/caibucai22/awesome-cuda/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey >
  <img src=https://img.shields.io/github/stars/caibucai22/awesome-cuda.svg?style=social >
  <img src=https://img.shields.io/badge/Release-preparing-brightgreen.svg >
 </div>


# awesome-cuda

收集整理有关深度学习网络cuda加速的各种项目、脚本、代码（包括cuda前后处理及其他kernel，Tensort重写网络等，偏部署，加速方向）

## Table of Contents 

- Basic

  - [How_to_optimize_in_GPU](#how_to_optimize_in_gpu-star) :star:
  - [cuda-learn-note](#cuda-learn-note)
  - [AI-Interview-Code](#ai-interview-code-)
  - [MatmulTutorial](#matmultutorial)
  - [how-to-optim-algorithm-in-cuda](#how-to-optim-algorithm-in-cuda)
- LLM

  - [llmam.cpp](#llamacpp-starstarstarstarstar) :star::star::star::star::star:
  - [CUDA-Learn-Notes](#cuda-learn-notes-starstarstar) :star::star::star:
  - [ffpa-attn-mma](#ffpa-attn-mma)
  - [FlashMLA](#flashmla)
  - [DeepGEMM](#deepgemm)
  - [grouped_gemm](#grouped_gemm)
  - [SpargeAttn](#spargeattn)
- CV-Image

  - [tensorrtx](#tensorrtx-starstarstar) :star::star::star:
- CV-PointCloud
  - [CUDA-PointPillars](#cuda-pointpillars)
  - [Pointcept](#pointcept-starstarstar) :star::star::star:



- [Blogs](#Blogs) 



## Basic

### CUDA_Freshman 

https://github.com/Tony-Tan/CUDA_Freshman

来自谭升博客对 CUDA_C_Programing  内容的实现，原博客有更多对于CUDA知识的理解，质量很高，是入门CUDA编程很好的教程，推荐



### How_to_optimize_in_GPU :star:

https://github.com/Liu-xiandong/How_to_optimize_in_GPU

一个GPU kernel优化方面的教程，介绍了如何优化CUDA内核，以达到接近理论峰值的性能。包括

- elementwise
- reduce
- sgemv
- sgemm

可以作为理论知识的一种实践学习



### cuda-learn-note <img src=https://img.shields.io/badge/tip-interview-brightgreen.svg >

https://github.com/whutbd/cuda-learn-note fork 了 https://github.com/DefTruth/CUDA-Learn-Notes:main

面试向、个人总结

提供了很多面试常见的kernel实现，以及优化手段的总结、block-tile、k-tile、vec4



### AI-Interview-Code <img src=https://img.shields.io/badge/tip-interview-brightgreen.svg >

https://github.com/bbruceyuan/AI-Interview-Code 仅引流

真实地址 https://bruceyuan.com/hands-on-code/

面试向，手写注意力机制等，Python 实现




### MatmulTutorial

https://github.com/KnowingNothing/MatmulTutorial

提供一个关于CUDA矩阵乘法 (Matrix Multiplication, Matmul) 的学习教程，初学者友好，使用了不同实现方式包括 cutlass、triton、自行手撸、以及在不同平台的实现

https://github.com/KnowingNothing/MatmulTutorial/tree/main/examples/matmul 点击直达

此外也提供了 reduction、attention操作的基础kernel实现



### how-to-optim-algorithm-in-cuda

https://github.com/BBuf/how-to-optim-algorithm-in-cuda

结合很多当下的模型 /框架 展开讨论 如何优化cuda算法，如

- oneflow 中的 elementwise
- pytorch 中的 index_add
- FastTransformer
- OpenAI 的 triton
- ...

提供了很多代码示例可供学习参考，

作者的原创学习笔记

https://github.com/BBuf/how-to-optim-algorithm-in-cuda?tab=readme-ov-file#20-%E5%8E%9F%E5%88%9B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0 点击直达

同时收录整理了很多优秀的博客（非常值得阅读，推荐）

https://github.com/BBuf/how-to-optim-algorithm-in-cuda?tab=readme-ov-file#%E6%96%87%E7%AB%A0 点击直达



## LLM

### llama.cpp :star::star::star::star::star:

https://github.com/ggerganov/llama.cpp

为llama等大模型提供一个高效灵活的推理框架，使得能够在本地设备上运行大语言模型。

针对性能进行了高度优化，支持多种模型格式，包括GGML，GPTQ等

主要功能

- **模型推理：** 可以对各种 LLM 模型进行推理，生成文本、翻译语言、编写不同类型的创意内容等等。

- **模型量化：** 支持模型量化，以减少内存占用，提高运行速度。



方便研究LLM的工作原理，部署后在本地进行各种实验，以及构建llm-based应用程序。

对于大语言模型的量化，加速优化操作学习具有重要参考意义。



### CUDA-Learn-Notes :star::star::star:

https://github.com/DefTruth/CUDA-Learn-Notes

一个cuda的学习仓库，收录了150+的kernel，包括了许多 transformer中使用的注意力模块加速实现；同时收录了 100+的 相关博客，包括 大模型、CV、推理部署加速。

包括

- **150+ Tensor/CUDA Cores Kernels:** 提供了大量的 CUDA 内核示例，涵盖了张量操作、卷积、全连接层等常见的深度学习操作。

- **flash-attn-mma, hgemm with WMMA, MMA and CuTe:** 高度优化的内核，达到接近 cuBLAS 的性能，特别适用于大规模的深度学习模型。

- **98%~100% TFLOPS of cuBLAS:** 仓库中的很多内核都能够达到 cuBLAS 库的 98% 以上的性能，这表明这些内核的优化程度非常高。

- **PyTorch bindings:** 许多内核都提供了 PyTorch 的绑定，方便在 PyTorch 中直接调用这些高性能的 CUDA 内核。

- **LLM/CUDA Blogs:** 大量的博客文章，详细介绍了各种 CUDA 优化技巧和深度学习加速方法。

作者还有许多其他优秀的相关工作【集中在大模型推理、diffusion推理等方面】，大佬！！！

值得学习！！！



### ffpa-attn-mma

https://github.com/DefTruth/ffpa-attn-mma

关注于使用 CUDA 实现一种名为 FFPA 注意力机制的加速算法，并结合 MMA (Matrix Multiply-Accumulate) 指令进行性能优化。

- **FFPA (Faster Flash Prefill Attention):**  指的是仓库实现的核心算法， 当注意力头维度headdim > 256时，实现了 O(1)级别的 SRAM 复杂度，比 SDPA-EA 快 1.8x~3x

- **MMA (Matrix Multiply-Accumulate):**  指的是矩阵乘法-累加指令。  MMA 指令是现代 NVIDIA GPU (例如 Volta, Turing, Ampere 架构) 中专门用于加速矩阵乘法运算的硬件指令，可以显著提高矩阵乘法的吞吐量和能效比。

作者在不同 NVIDA GPU 平台进行了测试 https://github.com/DefTruth/ffpa-attn-mma/tree/main/bench 点击直达

源码实现 https://github.com/DefTruth/ffpa-attn-mma/tree/main/csrc/cuffpa 点击直达 ，同时使用pybind 提供了 python接口



### FlashMLA

https://github.com/deepseek-ai/FlashMLA

FlashMLA 是针对 Hopper GPU 的高效 MLA (multi-head latent attention)解码内核，针对可变长度（用户输入token不规则）序列服务进行了优化（在生产环境中经过实战检验）。

提供了原始kernel实现，以及python 调用

当前发布特性：

- 支持BF16 
- 分页 KV 缓存，块大小为 64

> 提及受到 FlashAttention2/3 启发



### DeepGEMM

https://github.com/deepseek-ai/DeepGEMM

DeepGEMM 是 DeepSeek-V3 中提出的一个库，旨在利用细粒度缩放实现简洁高效的 FP8 通用矩阵乘法（GEMM）。它支持普通和专家混合（MoE）分组 GEMM。该库采用 CUDA 编写，在运行时使用轻量级即时 (JIT) 模块编译所有内核，安装时无需编译。

DeepGEMM 只支持英伟达公司的 Hopper tensor core。为了解决不精确的 FP8 tensor core 累加问题，它采用了 CUDA core 两级累加（promotion）技术。

虽然它利用了 CUTLASS 和 CuTe 的一些概念，但避免了对它们的模板或代数的严重依赖。相反，该库的设计非常简单，只有一个核心内核函数，约 300 行代码。这使它成为学习 Hopper FP8 矩阵乘法和优化技术的简洁易用的资源。

设计轻巧，DeepGEMM 在各种矩阵形状下的性能仍可媲美甚至超越经过专家调整的库



### grouped_gemm

https://github.com/tgale96/grouped_gemm

一个轻量级的库在Pytorch中提供 由 cutlass 实现的 grouped gemm kernels 



### SpargeAttn

https://github.com/thu-ml/SpargeAttn

SpargeAttention的官方实现，一种无训练的稀疏注意力，可以加速任何模型推断

论文：SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference

Paper: https://arxiv.org/abs/2502.18137



## CV-Image

## tensorrtx :star::star::star:

https://github.com/wang-xinyu/tensorrtx 

对经典的图像网络使用C++ Tensort Api进行了了重写，包括网络的权重加载，网络结构定义、构建 TensorRT 引擎，加载 TensorRT 引擎并运行推理

涉及的网络众多，如

- yolov3~yolov11

- unet
- detr、swin-transfommer

模型的前后处理多采用 cuda 进行加速，值得学习！



## CV-Pointcloud

### CUDA-PointPillars

https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars

NVIDIA官方提供的一个基于 CUDA 加速的、高性能 PointPillars 3D 物体检测网络的实现

从数据预处理到网络推理，都充分利用了 CUDA 提供的各种优化技术，例如 Kernel 优化、内存管理优化等，以实现极致的性能。

提供了完整的 PointPillars 网络的 CUDA 实现，包括网络结构定义、前向推理代码、以及训练和部署相关的工具。  可以直接使用该仓库提供的代码进行 PointPillars 网络的开发和应用

https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/tree/main/src/pointpillar 点击直达代码

可学习

- **点云数据预处理 (Preprocessing with CUDA)**：体素化、柱状特征编码 
- **PointPillars 网络推理 (Network Inference with CUDA)**
- **CUDA Kernel 优化**：  kernel launch 优化、内存管理优化
- **后处理 (Postprocessing with CUDA)**：NMS、BBox 解析转换

是学习使用cuda 加速点云网络的优秀实例



### Pointcept :star::star::star:

https://github.com/Pointcept/Pointcept

实现并统一了众多的点云网络，其中涉及点云的耗时操作，使用cuda编写加速，并提供python接口调用

放在libs下

https://github.com/Pointcept/Pointcept/tree/main/libs 点击直达

涉及点云操作有

- grouping 聚类
- 插值
- 采样
- knnquery
- subtraction
- 注意力机制、相对位置编码rpe等

# Blogs

- [知乎-国内大厂GPU CUDA高频面试问题汇总（含部分答案）](https://zhuanlan.zhihu.com/p/678602674)
- [知乎-深入浅出GPU优化系列：reduce优化 ](https://zhuanlan.zhihu.com/p/426978026)
- [NVIDIA-CUDA reduce optimization](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [unsloth-blogs](https://unsloth.ai/blog)
- [整理的名词一览](https://github.com/caibucai22/awesome-cuda/blob/main/Glossary.md)
- [谭升的博客 CUDA_C_Programing系列]([分类: CUDA | 谭升的博客](https://face2ai.com/categories/CUDA/))





