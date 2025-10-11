import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL


device = "cuda:0"
prompt = "a cute cat"
seed= 42
generator = torch.Generator(device).manual_seed(seed)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16).to(device)
pil_image = pipeline(prompt=prompt, generator=generator, num_inference_steps=20).images[0]
pil_image.save("images/sdxl_cat_with_pipline.png")

