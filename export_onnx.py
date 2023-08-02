from cldm.model import create_model, load_state_dict
import torch
import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
from onnx import shape_inference


def export_onnx(model, input, file, input_names, output_names, dynamic_axes):
    with torch.no_grad():
        if(type(input) == list):
            input = tuple(input)
        torch.onnx.export(model,
                        input,
                        file,
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
                        )


def export_hackathon_onnx(model):    
    
    image_height = 256
    image_width = 384
    
    model = model.cuda()
    
    # ------------------------------
    # Export clip
    # ------------------------------
    # Clip has two output
    print("------------ Export Clip ------------")    
    clip_model = model.cond_stage_model.transformer
    batch_size = 1
    inputs_clip=torch.zeros(batch_size, 77, dtype=torch.int32, device="cuda:0")
    
    export_onnx(model=clip_model,
                input=inputs_clip,
                file='./clip.onnx',
                input_names=['input_ids'],
                output_names=['text_embeddings', 'pooler_output'],
                dynamic_axes={'input_ids': {0: 'B'}, 
                               'text_embeddings': {0: 'B'}}
                )

    # ------------------------------
    # Export vae_decoder
    # ------------------------------
    print("--------- Export vae_decoder ---------")    
    
    vae_decoder = model.first_stage_model
    vae_decoder.forward = vae_decoder.decode
    
    latent_height = image_height // 8
    latent_width = image_width // 8
    
    inputs_decoder=torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    
    export_onnx(model=vae_decoder, 
                input=inputs_decoder,
                file="./vae_decoder.onnx",
                input_names=['latent'],
                output_names=['images'],
                dynamic_axes={'latent': {0: 'B', 2: 'H', 3: 'W'}, 
                              'images': {0: 'B', 2: '8H', 3: '8W'}}
                )
    
    # ------------------------------
    # Export controlnet with unet
    # ------------------------------
    print("--------- Export unet with controlnet ---------")    

    model.forward=model.fusion_forward
    #check validation
    controlunet_model=model
    x_in = torch.randn(2, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    h_in = torch.randn(2, 3, image_height, image_width, dtype=torch.float32, device='cuda:0')
    t_in = torch.zeros(2, dtype=torch.int32, device='cuda:0')
    c_in = torch.randn(2, 77, 768, dtype=torch.float32, device='cuda:0')

    dynamic_table = {'x_in': {0 : '2B', 2 : 'H', 3 : 'W'}, 
                     'h_in': {0 : '2B', 2 : '8H', 3 : '8W'}, 
                     't_in': {0 : '2B'},
                     'c_in': {0 : '2B'},
                     'output':{0 : '2B', 2 : 'H', 3 : 'W'}}
    
    export_onnx(model=controlunet_model,
                input=[x_in, h_in, t_in, c_in],
                file="./controlunet.onnx",
                input_names=['x_in', "h_in", "t_in", "c_in"],
                output_names=['output'],
                dynamic_axes=dynamic_table
                )

class Optimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


if __name__ == '__main__':
    
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
    
    export_hackathon_onnx(model)
    opt = Optimizer(onnx.load('./clip.onnx'))
    opt.select_outputs([0])  # delete graph output#1
    opt.cleanup()
    opt.fold_constants()
    opt.infer_shapes()
    opt.select_outputs([0], names=["text_embeddings"])  # rename network output
    opt_onnx_graph = opt.cleanup(return_onnx=True)
    onnx.save(opt_onnx_graph, "./clip_optimize.onnx")
    opt = Optimizer(onnx.load('./vae_decoder.onnx'))
    opt.cleanup()
    opt.fold_constants()
    opt.infer_shapes()
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    onnx.save(opt_onnx_graph, "./vae_decoder_optimize.onnx")
    opt = Optimizer(onnx.load('./controlunet.onnx'))
    opt.cleanup()
    opt.fold_constants()
    opt.infer_shapes()
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    onnx.save(opt_onnx_graph, "./controlunet_optimize.onnx")