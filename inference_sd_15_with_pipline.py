import torch
from diffusers import StableDiffusionPipeline

device = "cuda:0"
prompt = "a cute cat"
seed= 42
generator = torch.Generator(device).manual_seed(seed)
pipeline = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
pil_image = pipeline(prompt=prompt, generator=generator).images[0]
pil_image.save("images/sd15_cat_with_pipline.png")

