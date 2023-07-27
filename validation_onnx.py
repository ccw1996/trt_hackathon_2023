import onnxruntime
import numpy as np
import torch

device_name='cuda:0'
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

onnx_model = onnxruntime.InferenceSession('vae_encoder.onnx', providers=providers)

print(onnx_model.get_inputs()[0])

onnx_input = {onnx_model.get_inputs()[0].name: prompts[0]}
outputs = onnx_model.run(None, onnx_input)
