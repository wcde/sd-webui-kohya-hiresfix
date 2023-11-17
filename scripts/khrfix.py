### https://gist.github.com/kohya-ss/3f774da220df102548093a7abc8538ed

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from modules import scripts, script_callbacks
import gradio as gr
import torch

CONFIG_PATH = Path(__file__).parent.resolve() / '../config.yaml'


class Scaler(torch.nn.Module):
    def __init__(self, scale, block, scaler):
        super().__init__()
        self.scale = scale
        self.block = block
        self.scaler = scaler
        
    def forward(self, x, *args):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode=self.scaler)
        return self.block(x, *args)
    
    
class KohyaHiresFix(scripts.Script):
    def __init__(self):
        super().__init__()
        if not CONFIG_PATH.exists():
            open(CONFIG_PATH, 'w').close()
        self.config: DictConfig = OmegaConf.load(CONFIG_PATH)
        self.disable = False
        self.step_limit = 0

    def title(self):
        return "Kohya Hires.fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Kohya Hires.fix', open=False):
            with gr.Row():
                enable = gr.Checkbox(label='Enable extension', value=False)
            with gr.Row():
                s1 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Stop at", value=self.config.get('s1', 0.15))
                d1 = gr.Slider(minimum=1, maximum=10, step=1, label="Depth", value=self.config.get('d1', 3))
            with gr.Row():
                s2 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Stop at", value=self.config.get('s2', 0.3))
                d2 = gr.Slider(minimum=1, maximum=10, step=1, label="Depth", value=self.config.get('d2', 4))
            with gr.Row():
                scaler = gr.Dropdown(['bicubic', 'bilinear', 'nearest', 'nearest-exact'], label='Layer scaler', 
                                     value=self.config.get('scaler', 'bicubic'))
                downscale = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, label="Downsampling scale", 
                                      value=self.config.get('downscale', 0.5))
                upscale = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, label="Upsampling scale", 
                                    value=self.config.get('upscale', 4.0))
            with gr.Row():
                smooth_scaling = gr.Checkbox(label="Smooth scaling", value=self.config.get('smooth_scaling', True))
                early_out = gr.Checkbox(label="Early upsampling", value=self.config.get('early_out', False))
                only_one_pass = gr.Checkbox(label='Disable for additional passes', 
                                            value=self.config.get('only_one_pass', True))
        
        ui = [enable, only_one_pass, d1, d2, s1, s2, scaler, downscale, upscale, smooth_scaling, early_out]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)
        return ui
    

    def process(self, p, enable, only_one_pass, d1, d2, s1, s2, scaler, downscale, upscale, smooth_scaling, early_out):
        self.config = DictConfig({name: var for name, var in locals().items() if name not in ['self', 'p']})
        if not enable or self.disable:
            script_callbacks.remove_current_script_callbacks()
            return
        model = p.sd_model.model.diffusion_model
        if s1 > s2: self.config.s2 = s1
        self.p1 = (s1, d1 - 1)
        self.p2 = (s2, d2 - 1)
        self.step_limit = 0
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step < self.step_limit: return
            for s, d in [self.p1, self.p2]:
                out_d = d if self.config.early_out else -(d + 1)
                if params.sampling_step < params.total_sampling_steps * s:
                    if not isinstance(model.input_blocks[d], Scaler):
                        model.input_blocks[d] = Scaler(self.config.downscale, model.input_blocks[d], self.config.scaler)
                        model.output_blocks[out_d] = Scaler(self.config.upscale, model.output_blocks[out_d], self.config.scaler)
                    elif self.config.smooth_scaling:
                        scale_ratio = params.sampling_step / (params.total_sampling_steps * s)
                        downscale = min((1 - self.config.downscale) * scale_ratio + self.config.downscale, 1.0)
                        model.input_blocks[d].scale = downscale
                        model.output_blocks[out_d].scale = self.config.upscale * (self.config.downscale / downscale)
                    return
                elif isinstance(model.input_blocks[d], Scaler) and (self.p1[1] != self.p2[1] or s == self.p2[0]):
                    model.input_blocks[d] = model.input_blocks[d].block
                    model.output_blocks[out_d] = model.output_blocks[out_d].block
            self.config.step_limit = params.sampling_step if self.config.only_one_pass else 0
        
        script_callbacks.on_cfg_denoiser(denoiser_callback)
        
    def postprocess(self, p, processed, *args):
        for i, b in enumerate(p.sd_model.model.diffusion_model.input_blocks):
            if isinstance(b, Scaler):
                p.sd_model.model.diffusion_model.input_blocks[i] = b.block
        for i, b in enumerate(p.sd_model.model.diffusion_model.output_blocks):
            if isinstance(b, Scaler):
                p.sd_model.model.diffusion_model.output_blocks[i] = b.block
        OmegaConf.save(self.config, CONFIG_PATH)