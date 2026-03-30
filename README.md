<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/downloads/caibucai22/awesome-cuda/total?color=ccf&label=downloads&logo=github&logoColor=lightgrey >
  <img src=https://img.shields.io/github/stars/caibucai22/awesome-cuda.svg?style=social >
  <img src=https://img.shields.io/badge/Release-maintaining-brightgreen.svg >
</div>

# awesome-cuda

> 系统化整理深度学习 **CUDA 加速与部署** 生态资源，聚焦生产环境优化方案

**维护状态**: ✅ 活跃更新 | **最后更新**: 2026-03-30 | **分支**: [`new-organization`](https://github.com/caibucai22/awesome-cuda/tree/new-organization)

---

## 📖 关于本仓库

本仓库旨在为 **CUDA 深度学习部署工程师** 和 **性能优化爱好者** 提供高质量的精选资源导航。内容偏重：

- 🎯 **实战导向**: 优先收录经过大规模生产验证的框架和工具
- ⚡ **加速核心**: 聚焦算子优化、注意力机制、Tensor Core 利用
- 🛠️ **部署优先**: 涵盖从模型转换、推理引擎到端侧部署的全链路
- 📚 **中文友好**: 重点收录中文开发者社区的优质教程和博客

**适合人群**:
- 深度学习推理部署工程师
- CUDA 性能优化开发者
- 准备 GPU 相关面试的求职者
- 希望理解 LLM/CV 底层加速原理的研究者

**使用建议**: 按 [快速索引](#-快速索引) 选择类别，再系统性入门。

---

> 💡 **版本说明**: 当前展示为 **new-organization 分支**（按资源类型重新分类重构，更简洁，结构清晰）。如需查看原始版本（按应用领域分类，手工整理，介绍更为细致），请访问 [main 分支 README](https://github.com/caibucai22/awesome-cuda/blob/main/README.md)。

| 类别 | 项目数 | 适合人群 | 描述 |
|------|--------|----------|------|
| 📚 [学习教程](#-学习教程) | 16 | 初学者 → 进阶 | CUDA 编程入门、优化技巧、专项深入 |
| 🚀 [生产框架](#-生产框架) | 10 | 部署工程师 | LLM/CV/点云推理服务框架 |
| 🔧 [算子库 & 内核](#-算子库--内核) | 15 | 性能优化者 | Attention、GEMM、CV/点云算子 |
| 🛠️ [工具 & 生态](#-工具--生态) | 6 | 全栈开发者 | NVIDIA 官方库、编译器、数据加载 |
| 📖 [面试 & 知识](#-面试--知识) | 5 | 求职者 | 面试题、知识总结、**12 篇博客** |

---

## 📚 学习教程

> 系统性学习 CUDA 编程和优化方法

### CUDA 入门

| 项目 | Stars | 简介 |
|------|-------|------|
| [cuda-samples](https://github.com/NVIDIA/cuda-samples) | ⭐ 2k+ | **NVIDIA 官方示例**，从基础到进阶全覆盖 |
| [CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman) | ⭐ 1k+ | 谭升博客系列教程，质量极高，入门首选 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | ⭐ 3k+ | Python 可视化学习，理解 GPU 并行思维 |
| [MatmulTutorial](https://github.com/KnowingNothing/MatmulTutorial) | ⭐ 500+ | 矩阵乘法从零到优化的完整教程 |

### 优化进阶

| 项目 | Stars | 简介 |
|------|-------|------|
| [How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU) | ⭐ 1k+ | 接近理论峰值的 kernel 优化实战 |
| [how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | ⭐ 2k+ | 结合主流框架代码分析优化技巧 |
| [Cute-Learning](https://github.com/DD-DuDa/Cute-Learning) | ⭐ 500+ | Cutlass Cute 编程入门，Hopper 架构必备 |

### 专项深入

| 项目 | Stars | 简介 |
|------|-------|------|
| [tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn) | ⭐ 1k+ | AI 编译器（TVM + MLIR）学习 |
| [tutorial-multi-gpu](https://github.com/FZJ-JSC/tutorial-multi-gpu) | ⭐ 200+ | 分布式 GPU 编程 |
| [CUDA-Related](https://github.com/sungenglab/CUDA-Related) | ⭐ 2k+ | 分阶段系统教程（含 LLM 推理） |
| [tensorrt-cookbook](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook) | ⭐ 1k+ | NVIDIA 官方 TensorRT 实践教程（Hackathon） |
| [llm.c](https://github.com/karpathy/llm.c) | ⭐ 5k+ | 纯 C/CUDA 实现 LLM training，学习基础算子最佳实践 |

---

## 🚀 生产框架

> 经过大规模生产验证的推理服务框架

### LLM 推理

| 项目 | Stars | 适用场景 |
|------|-------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | ⭐ 40k+ | 高吞吐在线服务 |
| [sglang](https://github.com/sgl-project/sglang) | ⭐ 10k+ | 快速部署、结构化输出 |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | ⭐ 70k+ | 本地推理、资源受限环境 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ⭐ 8k+ | NVIDIA 生态最优性能 |
| [Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | ⭐ 2k+ | LLM 推理全景图（论文/框架/技术） |

### CV 推理

| 项目 | Stars | 简介 |
|------|-------|------|
| [jetson-inference](https://github.com/dusty-nv/jetson-inference) | ⭐ 8k+ | Jetson 平台全栈部署指南 |
| [tensorrtx](https://github.com/wang-xinyu/tensorrtx) | ⭐ 6k+ | TensorRT C++ API 重写经典网络 |
| [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) | ⭐ 500+ | Jetson 平台 TensorRT 模型演示（多种网络） |

### CV PointCloud

| 项目 | Stars | 简介 |
|------|-------|------|
| [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) | ⭐ 500+ | NVIDIA 官方 PointPillars CUDA 实现（端到端优化） |
| [Pointcept](https://github.com/Pointcept/Pointcept) | ⭐ 3k+ | 点云网络统一框架，提供 CUDA 加速算子 |

---

## 🔧 算子库 & 内核

> 高性能算子实现，可直接集成或学习参考

### Attention 优化

| 项目 | Stars | 核心优势 | 架构支持 |
|------|-------|----------|----------|
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | ⭐ 18k+ | 算法与硬件极致联合优化 | Ampere+ |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | ⭐ 3k+ | Hopper 高效 MLA 解码 | Hopper |
| [SpargeAttn](https://github.com/thu-ml/SpargeAttn) | ⭐ 1k+ | 无训练稀疏注意力 | Any |
| [cuda_self-attention](https://github.com/Fizzmy/cuda_self_attention) | ⭐ 200+ | 细粒度算子拆分教学 | Any |
| [ffpa-attn-mma](https://github.com/DefTruth/ffpa-attn-mma) | ⭐ 300+ | O(1) SRAM 复杂度 | Ampere+ |

### GEMM 实现

| 项目 | Stars | 特性 | 精度 |
|------|-------|------|------|
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | ⭐ 2k+ | DeepSeek-V3 FP8 GEMM，JIT 编译 | FP8 |
| [CUDA_gemm](https://github.com/Cjkkkk/CUDA_gemm) | ⭐ 500+ | 块稀疏 + 非均匀量化 GEMM | INT4/FP16 |
| [grouped_gemm](https://github.com/tgale96/grouped_gemm) | ⭐ 200+ | MoE 分组 GEMM（Cutlass） | FP16/BF16 |

### 通用算子库

| 项目 | Stars | 内容 |
|------|-------|------|
| [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes) | ⭐ 3k+ | **150+ kernel**，98%~100% cuBLAS 性能 |
| [CUDA_Kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples) | ⭐ 500+ | 高频面试算子题目（naive → 优化） |
| [Turbo-Softmax](https://github.com/LongWeihan/Turbo-Softmax) | ⭐ 100+ | exp/div 多项式近似加速 |
| [gpu-topk](https://github.com/anilshanbhag/gpu-topk) | ⭐ 200+ | GPU 并行 Top-K 选择算法 |
| [Pointcept/knn](https://github.com/Pointcept/Pointcept/tree/main/libs/pointops2/src/knnquery)、 [cuda-kmeans](https://github.com/krulis-martin/cuda-kmeans)、[kmcuda](https://github.com/src-d/kmcuda) | ⭐ 1k+ | 点云 、KNN/Kmeans 实现集合 |

### CV 算子 & 工具

| 项目 | Stars | 简介 |
|------|-------|------|
| [CudaSift](https://github.com/Celebrandil/CudaSift) | ⭐ 200+ | SIFT 特征提取 CUDA 实现（1.2ms@1080p） |
| [tsne-cuda](https://github.com/CannyLab/tsne-cuda) | ⭐ 300+ | GPU 加速 t-SNE，比 sklearn 快 1200x |

---

## 🛠️ 工具 & 生态

> 官方库、编译器、数据加载等基础设施

| 项目 | Stars | 类别 | 简介 |
|------|-------|------|------|
| [cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) | ⭐ 300+ | 神经网络库 | cuDNN C++ wrapper，简化 API |
| [cuda-python](https://github.com/NVIDIA/cuda-python) | ⭐ 1k+ | Python 绑定 | CUDA 官方 Python 接口 |
| [DALI](https://github.com/NVIDIA/DALI) | ⭐ 2k+ | 数据加载 | GPU 加速数据预处理 |
| [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) | ⭐ 1k+ | 图像处理 | NVIDIA + ByteDance 联合开发 |
| [CCCL](https://github.com/NVIDIA/cccl) | ⭐ 2k+ | C++ 核心库 | Thrust + CUB + libcudacxx 集合 |

---

## 📖 面试 & 知识

> 面试准备、知识总结、博客资源

### 面试向

| 项目 | Stars | 内容 |
|------|-------|------|
| [cuda-learn-note](https://github.com/whutbd/cuda-learn-note) | ⭐ 1k+ | 面试常见 kernel 实现 + 优化总结 |
| [AI-Interview-Code](https://bruceyuan.com/hands-on-code/) | ⭐ 500+ | 手写注意力机制等（Python） |
| [CUDA_Kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples) | ⭐ 500+ | 高频面试算子（重复收录见算子库）|

### 博客合集

| 作者/来源 | 链接 | 主题 |
|-----------|------|------|
| **知乎** | [GPU CUDA 高频面试问题汇总](https://zhuanlan.zhihu.com/p/678602674) | 面试汇总 |
| **知乎** | [从 0.44ms 到 0.04ms - 10倍softmax优化](https://zhuanlan.zhihu.com/p/1964020134839576011) | Softmax 优化实战 |
| **知乎** | [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026) | Reduce 优化 |
| **知乎-PTX** | [Nvidia Tensor Core-MMA PTX编程入门](https://zhuanlan.zhihu.com/p/621855199) | Tensor Core 底层编程 |
| **谭升** | [CUDA_C_Programming 系列](https://face2ai.com/categories/CUDA/) | 入门到进阶 |
| **奔跑的IC** | [CUDA_C_Programming 学习](https://zmurder.github.io/categories/CUDA/) | 学习笔记 |
| **ZOMI** | [AI 体系知识](https://chenzomi12.github.io/index.html) | 硬件、编译、推理 |
| **Ken He** | [NVIDIA TensorRT 博客](https://developer.nvidia.com/zh-cn/blog/author/ken-he/) | TensorRT 官方 |
| **NVIDIA** | [reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | Reduce 优化 |
| **NVIDIA** | [cuSparse documentation](https://docs.nvidia.com/cuda/cusparse/index.html) | 稀疏矩阵库 |
| **NVIDIA** | [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/) | 基本线性代数库 |
| **unsloth** | [unsloth.ai/blog](https://unsloth.ai/blog) | LLM 量化与优化 |
|  | 《How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores》 | Tensor Core 矩阵乘法系统教程 |
|  | 《Implementing a fast Tensor Core matmul on the Ada Architecture》 | Ada 架构 Tensor Core 优化 |

---

## 📑 附录

### 术语表

详细术语解释请见 [Glossary.md](Glossary.md)

### 贡献指南

欢迎 PR 补充优秀项目！请确保：
1. 项目与 CUDA 加速/部署强相关
2. 维护状态活跃（近期有 commit）
3. 提供 GitHub 链接 + 50 字以内简介

### License

CC0 1.0 Universal - 公共领域贡献
