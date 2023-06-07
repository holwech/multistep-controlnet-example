from PIL import Image
from diffusers import DPMSolverMultistepScheduler, ControlNetModel, StableDiffusionControlNetPipeline
import torch
from controlnet_aux import MidasDetector, LineartDetector
import PIL.ImageOps

import numpy as np
import cv2

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise ValueError("need to run on GPU")
# set mixed precision dtype
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

def resize_image(image: Image, resolution = 512):
    W, H = image.size
    if resolution < min(W, H):
        k = resolution / min(W, H)
        W *= k
        H *= k

    W_new = int(np.round(W/8) * 8)
    H_new = int(np.round(H/8) * 8)
    return image.resize((W_new, H_new))


lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
depth = MidasDetector.from_pretrained("lllyasviel/Annotators")

cn_steps = [
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=cn_steps, torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cpu").manual_seed(1)


with open("cat.jpg", "rb") as i:
    image = Image.open("cat.jpg").convert("RGB")

image = resize_image(image)

cn_images = [
    lineart(image),
    depth(image),
]

output_image = pipe(
    prompt="A cute cat sitting on marble stairs, high quality, highly-detailed, hyper-realistic, RAW, DSLR",
    negative_prompt="ugly, deformed, malformed, bad, low quality",
    image=cn_images,
    generator=generator,
    guidance_scale=7.5,
    num_inference_steps=40,
).images[0]

print("Done rendering image.")

print(output_image)