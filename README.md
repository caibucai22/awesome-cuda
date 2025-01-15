收集整理有关深度学习网络cuda加速的各种项目、脚本、代码（包括cuda前后处理，Tensort重写网络等，偏部署方向）



## llama.cpp

https://github.com/ggerganov/llama.cpp

为llama等大模型提供一个高效灵活的推理框架，使得能够在本地设备上运行大语言模型。

针对性能进行了高度优化，支持多种模型格式，包括GGML，GPTQ等

主要功能

- **模型推理：** 可以对各种 LLM 模型进行推理，生成文本、翻译语言、编写不同类型的创意内容等等。

- **模型量化：** 支持模型量化，以减少内存占用，提高运行速度。



方便研究LLM的工作原理，部署后在本地进行各种实验，以及构建llm-based应用程序。

对于大语言模型的量化，加速优化操作学习具有重要参考意义。



## CUDA-Learn-Notes

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



## tensorrtx

https://github.com/wang-xinyu/tensorrtx

对经典的图像网络使用C++ Tensort Api进行了了重写，包括网络的权重加载，网络结构定义、构建 TensorRT 引擎，加载 TensorRT 引擎并运行推理

涉及的网络众多，如

- yolov3~yolov11

- unet
- detr、swin-transfommer

模型的前后处理多采用 cuda 进行加速，值得学习！



## Pointcept

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



