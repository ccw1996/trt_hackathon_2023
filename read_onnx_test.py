import tensorrt as trt
import onnx



def test1(onnx_model):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)
    if isinstance(onnx_model, str):
        onnx_graph = onnx.load(onnx_model)

    # if onnx_graph.ByteSize() > 2147483648:  # for unet
    #     onnx.shape_inference.infer_shapes_path(onnx_model)
    #     onnx_graph = onnx.load(onnx_model)

    if not parser.parse(onnx_graph.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')
    
    
    
def test2(onnx_model):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    # if onnx_graph.ByteSize() > 2147483648:  # for unet
    #     onnx.shape_inference.infer_shapes_path(onnx_model)
    #     onnx_graph = onnx.load(onnx_model)

    with open(onnx_model, "rb") as model:
        if not parser.parse(model.read()):
            error_msgs = ''
            for error in range(parser.num_errors):
                error_msgs += f'{parser.get_error(error)}\n'
            raise RuntimeError(f'Failed to parse onnx, {error_msgs}')
    
if __name__ == '__main__':

    # test1('./controlUnet_onnx/controlUnet.onnx')
    test2('./controlUnet_onnx/controlUnet.onnx')
    # test2('onnx_model/clip.onnx')
    # test1('onnx_model/clip.onnx')