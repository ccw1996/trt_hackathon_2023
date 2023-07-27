from share import *
import config
import cv2
import numpy as np
import torch
import random
from PIL import Image

from annotator.canny import CannyDetector
from cldm.model import create_model,load_state_dict
from cldm.ddim_hacked import DDIMSampler

prompts=['best quality, extremely detailed']
negative_prompts=['blurry, poor quality, painting, worst quality, lowres, low quality']
repeat_prompts=1
image_width=384
image_height=256
denoising_steps=20
onnx_opset=17
seed=1000

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(1000)

prompt=prompts*repeat_prompts
negative_prompt=negative_prompts*len(prompt)

max_batch_size=16
batch_size=len(prompt)
print(prompt)
print(batch_size)
assert 0
apply_canny=CannyDetector()

def image_process(input_image):
    canny_image = cv2.Canny(canny_image, 100, 200)
    image = canny_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_schedule = DDIMSampler(model).make_schedule(ddim_num_steps=20,ddim_eta=0.0,verbose=True)

clip_model=model.cond_stage_model
#print(clip_model)
clip_encoder_model=model.cond_stage_model.transformer.text_model
clip_tokenizer=model.cond_stage_model.transformer.text_model.embeddings
print(model.get_learned_conditioning(prompts*1))
#prompts_id=clip_tokenizer(prompt)
#print("51")
#print(prompts_id)
with torch.inference_mode(), torch.autocast("cuda"):
    inputs_clip=torch.zeros(batch_size, 77, dtype=torch.int32, device="cuda:0")
    # torch.onnx.export(clip_encoder_model,
    #                                 inputs_clip,
    #                                 "clip.onnx",
    #                                 export_params=True,
    #                                 opset_version=onnx_opset,
    #                                 do_constant_folding=True,
    #                                 input_names=['input_ids'],
    #                                 output_names=['text_embeddings'],
    #                                 dynamic_axes= {
    #                                     'input_ids': {0: 'B'},
    #                                     'text_embeddings': {0: 'B'}
    #                                 },
    #                         )
with torch.no_grad():
    result=clip_model(prompts)
torch.save(result,"clip_origin.pth")


vae_encoder=model.first_stage_model.encoder
with torch.inference_mode(), torch.autocast("cuda"):
    inputs_vae_encoder=torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device='cuda:0')
    # torch.onnx.export(vae_encoder,
    #                                 inputs_vae_encoder,
    #                                 "vae_encoder.onnx",
    #                                 export_params=True,
    #                                 opset_version=onnx_opset,
    #                                 do_constant_folding=True,
    #                                 input_names=['images'],
    #                                 output_names=['latent'],
    #                                 dynamic_axes= {
    #                                     'images': {0: 'B', 2: '8H', 3: '8W'},
    #                                     'latent': {0: 'B', 2: 'H', 3: 'W'}
    #                                 },
    #                         )
    
vae_decoder=model.first_stage_model.decoder
with torch.inference_mode(), torch.autocast("cuda"):
    latent_height=image_height//8
    latent_width=image_width//8
    inputs_vae_decoder=torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device='cuda:0')
    torch.onnx.export(vae_decoder,
                                    inputs_vae_decoder,
                                    "vae_decoder.onnx",
                                    export_params=True,
                                    opset_version=onnx_opset,
                                    do_constant_folding=True,
                                    input_names=['latent'],
                                    output_names=['images'],
                                    dynamic_axes= {
                                        'latent': {0: 'B', 2: 'H', 3: 'W'},
                                        'images': {0: 'B', 2: '8H', 3: '8W'}
                                    },
                            )
    
control_model = model.control_model
x_in = torch.randn(1, 4, image_height//8, image_width //8, dtype=torch.float32, device='cuda:0')
h_in = torch.randn(1, 3, image_height, image_width, dtype=torch.float32, device='cuda:0')
t_in = torch.zeros(1, dtype=torch.int64, device='cuda:0')
c_in = torch.randn(1, 77, 768, dtype=torch.float32, device='cuda:0')
controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)
output_names = []
for i in range(13):
    output_names.append("out_"+ str(i))

dynamic_table = {'x_in': {0 : 'B', 2 : 'H', 3 : 'W'}, 
                    'h_in': {0 : 'B', 2 : '8H', 3 : '8W'}, 
                    't_in': {0 : 'B'},
                    'c_in': {0 : 'B'}}

for i in range(13):
    dynamic_table[output_names[i]] = {0 : "B"}

# torch.onnx.export(control_model,               
#                     (x_in, h_in, t_in, c_in),  
#                     "./sd_control_test.onnx",   
#                     export_params=True,
#                     opset_version=17,
#                     do_constant_folding=True,
#                     input_names = ['x_in', "h_in", "t_in", "c_in"], 
#                     output_names = output_names, 
#                     dynamic_axes = dynamic_table)

unet_model=model.model.diffusion_model
diffusions=unet_model(x=x_in,timesteps=t_in,context=c_in,controls=controls,only_mid_control=False)
print("x_in : ")
print(x_in.shape)
print("timestep : ")
print(t_in.shape)
print("context : ")
print(c_in.shape)

print("control : ")
for i in range(13):
    print(controls[i].shape)

    
control_1 = torch.randn(1, 320, 32,48 , dtype=torch.float32, device='cuda:0')
control_2 = torch.randn(1, 320, 32,48 , dtype=torch.float32, device='cuda:0')
control_3 = torch.randn(1, 320, 32,48 , dtype=torch.float32, device='cuda:0')
control_4 = torch.randn(1, 320,16,24 , dtype=torch.float32, device='cuda:0')
control_5 = torch.randn(1, 640,16,24 , dtype=torch.float32, device='cuda:0')
control_6 = torch.randn(1, 640,16,24 , dtype=torch.float32, device='cuda:0')
control_7 = torch.randn(1, 640,8,12 , dtype=torch.float32, device='cuda:0')
control_8 = torch.randn(1, 1280,8,12 , dtype=torch.float32, device='cuda:0')
control_9 = torch.randn(1, 1280,8,12 , dtype=torch.float32, device='cuda:0')
control_10 = torch.randn(1, 1280,4,6 , dtype=torch.float32, device='cuda:0')
control_11 = torch.randn(1, 1280,4,6 , dtype=torch.float32, device='cuda:0')
control_12 = torch.randn(1, 1280,4,6 , dtype=torch.float32, device='cuda:0')
control_13 = torch.randn(1, 1280,4,6 , dtype=torch.float32, device='cuda:0')
# torch.onnx.export(unet_model,(x_in,t_in,c_in,controls,False),'unet.onnx',export_params=True,opset_version=17,
#                     do_constant_folding=True,
#                     input_names = ['x_in', "t_in", "c_in", "controls"],
#                     output_names=["result"],
#                     dynamic_axes={
#                     'x_in': {0 : 'B', 2 : 'H', 3 : 'W'},
#                     't_in': {0 : 'B'},
#                     'c_in': {0 : 'B'},
#                     'controls': {0 : 'B'},
#         },)
