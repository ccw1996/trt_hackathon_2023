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