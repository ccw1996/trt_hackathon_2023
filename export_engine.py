import os
import onnx
import tensorrt as trt

from typing import Dict, Sequence, Union


def from_onnx(onnx_model: str,
              file_name: str,
              input_shapes: Dict[str, Sequence[int]],
              max_workspace_size: int = 0,
              fp16_mode: bool = True,
              device_id: int = 0,
              log_level: trt.Logger.Severity = trt.Logger.ERROR,
              tf32 : bool = True) -> trt.ICudaEngine:
    
    os.environ['CUDA_DEVICE'] = str(device_id)
    
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_graph = onnx.load(onnx_model)

    if not parser.parse(onnx_graph.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        
    if tf32 is False:
        config.clear_flag(trt.BuilderFlag.TF32)

    engine = builder.build_serialized_network(network, config)
    assert engine is not None, 'Failed to create TensorRT engine'

    with open(file_name, mode='wb') as f:
        f.write(bytearray(engine))
    return engine



def export_engine():
    image_height = 256
    image_width = 384
    
    latent_height = image_height // 8
    latent_width = image_width // 8
    
    # ---------------------------------
    # Export clip.plan
    # ---------------------------------
    clip_input_shape = dict(input_ids = dict(min_shape = [1, 77],
                                             opt_shape = [1, 77],
                                             max_shape = [1, 77]))
    from_onnx('./clip.onnx',
              './clip.plan',
              clip_input_shape,
              1 << 32,
              )
    print("-------- Export clip.plan : Done! --------")
    
    # ---------------------------------
    # Export vae_decoder.plan
    # ---------------------------------
    decoder_input_shape = dict(latent = dict(min_shape = [1, 4, latent_height, latent_width],
                                             opt_shape = [1, 4, latent_height, latent_width],
                                             max_shape = [1, 4, latent_height, latent_width]))
    from_onnx('./vae_decoder.onnx',
              './vae_decoder.plan',
              decoder_input_shape,
              1 << 32)
    print("----- Export vae_decoder.plan : Done! ----")
    
    
    # ---------------------------------
    # Export controlnet.plan
    # ---------------------------------
    controlunet_input_shape = dict(x_in = dict(min_shape = [1, 4, latent_height, latent_width],
                                              opt_shape = [1, 4, latent_height, latent_width],
                                              max_shape = [1, 4, latent_height, latent_width]),
                                  h_in = dict(min_shape = [1, 3, image_height, image_width],
                                              opt_shape = [1, 3, image_height, image_width],
                                              max_shape = [1, 3, image_height, image_width]),
                                  t_in = dict(min_shape = [1],
                                              opt_shape = [1],
                                              max_shape = [1]),
                                  c_in = dict(min_shape = [1, 77, 768],
                                              opt_shape = [1, 77, 768],
                                              max_shape = [1, 77, 768]))
    
    # from_onnx('./controlunet.onnx',
    #           './controlunet.plan',
    #           controlunet_input_shape,
    #           1 << 32)
    print("----- Export controlnet.plan : Done! -----")     
        
    export_unet_shell = 'trtexec --onnx=./controlunet.onnx --saveEngine=./controlunet.plan --fp16 --optShapes=x_in:1x4x32x48,t_in:1,c_in:1x77x768,h_in:1x3x256x384'
    print(export_unet_shell)
    
    os.system(export_unet_shell)

if __name__ == "__main__":
    export_engine()