# 简单记录下速度
_所记录的速度都是第二次推理之后的结果_
## 1, 使用 controlnet.engine 与 unet.engine， 开启FP16

| Module      | Cost time|
|---          |---       |
| Preprocess  |    7.415 | \ 
| Clip        |   35.348 | \ 
| Ctrl & Unet |  836.029 | \ 
| Decode      |   10.542 | \ 
| Postprocess |   47.244 | \ 
| Total       |  936.578 | \ 

## 2, 使用 controlnet.engine、unet.engine 与 clip.engine， 开启FP16

clip部分有bug，输出Nan

| Module      | Cost time|
|---          |---       |
| Preprocess  |    7.620 | \ 
| Clip        |    5.493 | \ 
| Ctrl & Unet |  843.790 | \ 
| Decode      |    9.568 | \ 
| Postprocess |   47.595 | \ 
| Total       |  914.066 | \ 

## 3, 使用 controlnet.engine、unet.engine 与 decoder.engine， 开启FP16


| Module      | Cost time|
|---          |---       |
| Preprocess  |    4.008 | \ 
| Clip        |   29.684 | \ 
| Ctrl & Unet |  828.990 | \ 
| Decode      |   23.349 | \ 
| Postprocess |    0.932 | \ 
| Total       |  886.963 | \

## 3, 使用 controlnet.engine、unet.engine 与 decoder.engine， 开启FP16 预存 controlnet 与 unet 使用的buffer

| Module      | Cost time|
|---          |---       |
| Preprocess  |    4.542 | \ 
| Clip        |   28.044 | \ 
| Ctrl & Unet |  694.356 | \ 
| Decode      |   24.171 | \ 
| Postprocess |    0.922 | \ 
| Total       |  752.035 | \ 


## trtexec计算延迟

unet.engine \
[07/29/2023-04:28:03] [I] Average on 10 runs - GPU latency: 12.0265 ms - Host latency: 13.0377 ms (enqueue 8.85208 ms) \
[07/29/2023-04:28:03] [I] Average on 10 runs - GPU latency: 11.9674 ms - Host latency: 12.9602 ms (enqueue 8.82845 ms) \
[07/29/2023-04:28:03] [I] Average on 10 runs - GPU latency: 11.6893 ms - Host latency: 12.7035 ms (enqueue 8.60209 ms) 

controlnet.engine \
[07/29/2023-04:30:07] [I] Average on 10 runs - GPU latency: 4.8683 ms - Host latency: 5.9763 ms (enqueue 2.47917 ms) \
[07/29/2023-04:30:07] [I] Average on 10 runs - GPU latency: 4.72924 ms - Host latency: 5.916 ms (enqueue 2.80463 ms) \
[07/29/2023-04:30:07] [I] Average on 10 runs - GPU latency: 4.72996 ms - Host latency: 6.00052 ms (enqueue 2.96591 ms) 

controlUnet.engine batch=1 \
[07/29/2023-04:30:50] [I] Average on 10 runs - GPU latency: 19.1373 ms - Host latency: 19.3018 ms (enqueue 19.043 ms) \
[07/29/2023-04:30:50] [I] Average on 10 runs - GPU latency: 18.7807 ms - Host latency: 18.9513 ms (enqueue 18.566 ms) \
[07/29/2023-04:30:50] [I] Average on 10 runs - GPU latency: 18.8119 ms - Host latency: 18.9785 ms (enqueue 18.5955 ms) 

controlUnet.engine batch=2 \
[07/29/2023-04:32:51] [I] Average on 10 runs - GPU latency: 18.8324 ms - Host latency: 19.0031 ms (enqueue 18.6409 ms) \
[07/29/2023-04:32:51] [I] Average on 10 runs - GPU latency: 18.8187 ms - Host latency: 18.9871 ms (enqueue 18.6112 ms) \
[07/29/2023-04:32:51] [I] Average on 10 runs - GPU latency: 18.7858 ms - Host latency: 18.9533 ms (enqueue 18.5757 ms) 

## 使用 controlUnet.plan 与 decoder.engine， FP=16， batch=1

| Module      | Cost time|
|---          |---       |
| Preprocess  |    4.557 | \ 
| Clip        |   29.419 | \ 
| Ctrl & Unet |  769.527 | \ 
| Decode      |   23.810 | \ 
| Postprocess |    0.883 | \ 
| Total       |  828.196 | \ 

融合后Ctrl & Unet速度变慢！！

## 使用 controlUnet.plan 与 decoder.engine， FP=16， batch=2

| Module      | Cost time|
|---          |---       |
| Preprocess  |    7.631 | \ 
| Clip        |   27.056 | \ 
| Ctrl & Unet |  545.225 | \ 
| Decode      |   25.067 | \ 
| Postprocess |    1.149 | \ 
| Total       |  606.128 | \