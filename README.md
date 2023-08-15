# trt_hackathon_2023 初赛第20名方案

## 基于stable diffusion源码修改

### 优化的思路

1. clip batch=2且fp16=True
2. vae decoder fp16=True
3. controlnet 和 unet合并为1个onnx并输出，并且fp16=true， batch=2，且builderOptimizationLevel=4
4. 简单的graphsurgeon处理，包含fold_constants，cleanup。
5. 共享内存
