import onnxruntime as ort
import tensorrt as trt




if __name__ == '__main__':
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, '')
    
    H = 256
    W = 384
    with open("./engine_model/unet_first_half.engine", 'rb') as f:
        engine_str = f.read()
        unet_first_half_engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
        unet_first_half_context = unet_first_half_engine.create_execution_context()
        
        unet_first_half_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        unet_first_half_context.set_binding_shape(1, (1,))
        unet_first_half_context.set_binding_shape(2, (1, 77, 768))
    
    