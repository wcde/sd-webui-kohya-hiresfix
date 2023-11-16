### https://gist.github.com/kohya-ss/3f774da220df102548093a7abc8538ed

from modules import scripts, script_callbacks
import gradio as gr
import torch


class Scaler(torch.nn.Module):
    def __init__(self, scale, block):
        super().__init__()
        self.scale = scale
        self.block = block
        
    def forward(self, x, *args):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return self.block(x, *args)
    

class KohyaHiresFix(scripts.Script):
    def __init__(self):
        super().__init__()
        self.disable = False

    def title(self):
        return "Kohya Hires.fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Kohya Hires.fix', open=False):
            enable = gr.Checkbox(label='Enable extension', value=False)
            with gr.Row():
                s1 = gr.Slider(minimum=0, maximum=12, step=1, label="Stop step", value=2)
                d1 = gr.Slider(minimum=2, maximum=10, step=1, label="Depth", value=3)
            with gr.Row():
                s2 = gr.Slider(minimum=0, maximum=16, step=1, label="Stop step", value=5)
                d2 = gr.Slider(minimum=2, maximum=10, step=1, label="Depth", value=4)
            with gr.Row():
                scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.1, label="Upsampling scale", value=2.0)
                early_out = gr.Checkbox(label="Early upsampling (affects performance/quality)", value=True)
        
        ui = [enable, d1, d2, s1, s2, scale, early_out]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)
        return ui
    

    def process(self, p, enable, d1, d2, s1, s2, scale, early_out):
        if not enable or self.disable:
            script_callbacks.remove_current_script_callbacks()
            return
        model = p.sd_model.model.diffusion_model
        if s1 > s2: s2 = s1
        self.p1 = (s1, d1)
        self.p2 = (s1, d2)
        self.scale = scale
        self.early_out = early_out
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            for s, d in [self.p1, self.p2]:
                out_d = d if self.early_out else -d
                if params.sampling_step < s:
                    if not isinstance(model.input_blocks[d], Scaler):
                        model.input_blocks[d] = Scaler(0.5, model.input_blocks[d])
                        model.output_blocks[out_d] = Scaler(self.scale, model.output_blocks[out_d])
                    return
                elif isinstance(model.input_blocks[d], Scaler) and (self.p1[1] != self.p2[1] or s == self.p2[0]):
                    model.input_blocks[d] = model.input_blocks[d].block
                    model.output_blocks[out_d] = model.output_blocks[out_d].block
                
        script_callbacks.on_cfg_denoiser(denoiser_callback)
        
    def postprocess(self, p, processed, *args):
        for i, b in enumerate(p.sd_model.model.diffusion_model.input_blocks):
            if isinstance(b, Scaler):
                p.sd_model.model.diffusion_model.input_blocks[i] = b.block
        for i, b in enumerate(p.sd_model.model.diffusion_model.output_blocks):
            if isinstance(b, Scaler):
                p.sd_model.model.diffusion_model.output_blocks[i if args[-1] else -i] = b.block