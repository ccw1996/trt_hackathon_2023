from share import *
import config
import time
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import tensorrt as trt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img, exists, instantiate_from_config

from export_onnx import export_hackathon_onnx
from export_engine import export_engine

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml') #.cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
    
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        # if not os.path.isfile("./onnx_model/clip.onnx") or \
        #    not os.path.isfile("./onnx_model/controlnet.onnx") or \
        #    not os.path.isfile("./unet_onnx/unet.onnx") or \
        #    not os.path.isfile("./onnx_model/vae_decoder.onnx"):               
        #     export_hackathon_onnx(self.model)
            
        # if not os.path.isfile("./clip.plan") or \
        #    not os.path.isfile("./controlnet.plan") or \
        #    not os.path.isfile("./unet.plan") or \
        #    not os.path.isfile("./vae_decoder.plan"):   
        #     export_engine()

        H = 256
        W = 384
        
        self.model.clip_context = None
        self.model.control_context = None
        self.model.unet_context = None
        self.model.decoder_context = None
        self.model.controlunet_context=None
        
    
        # -------------------------------
        # load clip engine
        # -------------------------------
            
        with open("./clip.plan", 'rb') as f:
            engine_str = f.read()
            clip_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
            clip_context = clip_engine.create_execution_context()
            
            clip_context.set_binding_shape(0, (2, 77))
            self.model.clip_context = clip_context
            
        # -------------------------------
        # load controlnet engine
        # -------------------------------
        with open("./controlunet.plan", 'rb') as f:
            engine_str = f.read()
            controlunet_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
            controlunet_context = controlunet_engine.create_execution_context()

            controlunet_context.set_binding_shape(0, (2, 4, H // 8, W // 8))
            controlunet_context.set_binding_shape(1, (2, 3, H, W))
            controlunet_context.set_binding_shape(2, (2,))
            controlunet_context.set_binding_shape(3, (2, 77, 768))
            self.model.controlunet_context = controlunet_context
            
        # -------------------------------
        # load decoder engine
        # -------------------------------
        
        with open("./vae_decoder.plan", 'rb') as f:
            engine_str = f.read()
            decoder_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
            decoder_context = decoder_engine.create_execution_context()
            
            decoder_context.set_binding_shape(0, (1, 4, 32, 48))
            self.model.decoder_context = decoder_context
            
            

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        
        ddim_steps = int(ddim_steps * 0.4)
        
        with torch.no_grad():
            start = time.time_ns() // 1000 
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            preprocess = time.time_ns() // 1000
            result_clip=self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples, [n_prompt] * num_samples, self.model.clip_context)
            result_clip1=result_clip[0]
            result_clip2=result_clip[1]
            
            cond = {"c_concat": [control], "c_crossattn": [result_clip1]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [result_clip2]}  # use clip net
            shape = (4, H // 8, W // 8)
            clip = time.time_ns() // 1000

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
            ctrlnet = time.time_ns() // 1000
            x_samples = self.model.decode_first_stage(samples, decoder_context=self.model.decoder_context)
            decode = time.time_ns() // 1000
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]   
            end = time.time_ns() // 1000
            
            # print("| Module      | Cost time|")
            # print("|---          |---       |")
            # print("| Preprocess  | {:8.3f} | \ ".format(1.0 * (preprocess - start) / 1000))
            # print("| Clip        | {:8.3f} | \ ".format(1.0 * (clip - preprocess) / 1000))
            # print("| Ctrl & Unet | {:8.3f} | \ ".format(1.0 * (ctrlnet - clip) / 1000))
            # print("| Decode      | {:8.3f} | \ ".format(1.0 * (decode - ctrlnet) / 1000))
            # print("| Postprocess | {:8.3f} | \ ".format(1.0 * (end - decode) / 1000))
            # print("| Total       | {:8.3f} | \ ".format(1.0 * (end - start) / 1000))  
                    
            
        return results
    
