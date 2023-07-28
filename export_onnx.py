from cldm.model import create_model, load_state_dict
import torch


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
    clip_model = model.cond_stage_model.transformer.text_model
    batch_size = 1
    inputs_clip=torch.zeros(batch_size, 77, dtype=torch.int64, device="cuda:0")
    
    export_onnx(model=clip_model,
                input=inputs_clip,
                file='./onnx_model/clip.onnx',
                input_names=['input_ids'],
                output_names=['text_embeddings', 'other_out'],
                dynamic_axes={'input_ids': {0: 'B'}, 
                               'text_embeddings': {0: 'B'},
                               'other_out' : {0 : 'B'}}
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
    
    # export_onnx(model=vae_decoder, 
    #             input=inputs_decoder,
    #             file="./onnx_model/vae_decoder.onnx",
    #             input_names=['latent'],
    #             output_names=['images'],
    #             dynamic_axes={'latent': {0: 'B', 2: 'H', 3: 'W'}, 
    #                           'images': {0: 'B', 2: '8H', 3: '8W'}}
    #             )
    
    # ------------------------------
    # Export controlnet
    # ------------------------------
    print("--------- Export controlnet ---------")    

    control_model = model.control_model
    x_in = torch.randn(1, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    h_in = torch.randn(1, 3, image_height, image_width, dtype=torch.float32, device='cuda:0')
    t_in = torch.zeros(1, dtype=torch.int64, device='cuda:0')
    c_in = torch.randn(1, 77, 768, dtype=torch.float32, device='cuda:0')
    output_names = []
    for i in range(13):
        output_names.append("control_"+ str(i))

    dynamic_table = {'x_in': {0 : 'B', 2 : 'H', 3 : 'W'}, 
                     'h_in': {0 : 'B', 2 : '8H', 3 : '8W'}, 
                     't_in': {0 : 'B'},
                     'c_in': {0 : 'B'}}
    
    # export_onnx(model=control_model,
    #             input=[x_in, h_in, t_in, c_in],
    #             file="./onnx_model/controlnet.onnx",
    #             input_names=['x_in', "h_in", "t_in", "c_in"],
    #             output_names=output_names,
    #             dynamic_axes=dynamic_table
    #             )
    
    # ------------------------------
    # Export unet 
    # ------------------------------
    print("-------- Export unet -------")    

    unet_model = model.model.diffusion_model
        
    x_in = torch.randn(1, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    t_in = torch.tensor([951], dtype=torch.int64, device='cuda:0')
    c_in = torch.randn(1, 77, 768, dtype=torch.float32, device='cuda:0')
    
    control = []
    control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 320, 16, 24, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 640, 16, 24, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 640, 16, 24, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 640, 8, 12, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 8, 12, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 8, 12, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    
    unet_input_names = ['x_in', "t_in", "c_in"]
    unet_input_names += [f'control_{i}' for i in range(len(control))]
    
    dynamic_table = {'x_in': {0 : 'B', 2 : 'H', 3 : 'W'},
                     't_in': {0 : 'B'},
                     'c_in': {0 : 'B'}}
    
    for i in range(len(control)):
        dynamic_table[f'control_{i}'] = {0 : 'B'}
    

    # export_onnx(model=unet_model,
    #             input=[x_in, t_in, c_in, control],
    #             file="./unet_onnx/unet.onnx",
    #             input_names=unet_input_names,
    #             output_names=['unet_output'],
    #             dynamic_axes=dynamic_table
    #             )
    
    
        
    
    # # ------------------------------
    # # Export unet first half
    # # ------------------------------
    # print("-------- Export unet first half -------")    

    # unet_model = model.model.diffusion_model
    # unet_model.forward = unet_model.first_half_forward
    # x_in = torch.randn(1, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    # t_in = torch.tensor([951], dtype=torch.int64, device='cuda:0')
    # c_in = torch.randn(1, 77, 768, dtype=torch.float32, device='cuda:0')

    # unet_fh_input_names = ['x_in', "t_in", "c_in"]
    # dynamic_table = {'x_in': {0 : 'B', 2 : 'H', 3 : 'W'},
    #                  't_in': {0 : 'B'},
    #                  'c_in': {0 : 'B'}}

    # unet_fh_output_names = ['h']
    # unet_fh_output_names += [f'hs_{i}' for i in range(12)]
    # unet_fh_output_names += ['emb']

    # export_onnx(model=unet_model,
    #             input=[x_in, t_in, c_in],
    #             file="./onnx_model/unet_first_half.onnx",
    #             input_names=unet_fh_input_names,
    #             output_names=unet_fh_output_names,
    #             dynamic_axes=dynamic_table
    #             )
    
    
    # # ------------------------------
    # # Export unet second half
    # # ------------------------------
    # print("-------- Export unet second half -------") 
    # unet_model.forward = unet_model.second_half_forward
    
    # h = torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0')
    # emb = torch.randn(1, 1280, dtype=torch.float32, device='cuda:0')
    
    # control = []
    # control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 320, 32, 48, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 320, 16, 24, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 640, 16, 24, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 640, 16, 24, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 640, 8, 12, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 8, 12, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 8, 12, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    # control.append(torch.randn(1, 1280, 4, 6, dtype=torch.float32, device='cuda:0'))
    
    # hs_shape = []
    # hs_shape.append([1, 320, 32, 48])
    # hs_shape.append([1, 320, 32, 48])
    # hs_shape.append([1, 320, 32, 48])
    # hs_shape.append([1, 320, 16, 24])
    # hs_shape.append([1, 640, 16, 24])
    # hs_shape.append([1, 640, 16, 24])
    # hs_shape.append([1, 640, 8, 12])
    # hs_shape.append([1, 1280, 8, 12])
    # hs_shape.append([1, 1280, 8, 12])
    # hs_shape.append([1, 1280, 4, 6])
    # hs_shape.append([1, 1280, 4, 6])
    # hs_shape.append([1, 1280, 4, 6])
    # hs = [torch.rand(hs_shape[i], dtype=torch.float32, device='cuda:0') for i in range(len(hs_shape))]
    
    
    # unet_sh_input_names = ['h']
    # unet_sh_input_names += [f'hs_{i}' for i in range(12)]
    # unet_sh_input_names += ['emb', 'c_in']
    # unet_sh_input_names += [f'control_{i}' for i in range(len(control))]
    
    
    # dynamic_table = {'h': {0 : 'B'}}
    # for i in range(len(hs_shape)):
    #     dynamic_table[f'hs_{i}'] = {0 : 'B'}
        
    # dynamic_table['emb'] = {0 : 'B'}
    # dynamic_table['c_in'] = {0 : 'B'}
    # for i in range(len(control)):
    #     dynamic_table[f'control_{i}'] = {0 : 'B'}
    
    
    # export_onnx(model=unet_model,
    #             input=[h, hs, emb, c_in, control],
    #             file="./onnx_model/unet_second_half.onnx",
    #             input_names=unet_sh_input_names,
    #             output_names=['unet_output'],
    #             dynamic_axes=dynamic_table
    #             )
    

    

if __name__ == '__main__':
    
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    
    export_hackathon_onnx(model)
    