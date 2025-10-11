import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
with torch.no_grad():
    device = "cuda:0"
    num_inference_steps = 50
    scheduler = PNDMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    tokenizer = CLIPTokenizer.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    negative_prompt = ""
    uncond_input = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    prompt = "a cute cat"
    cond_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_embeddings = text_encoder(cond_input.input_ids.to(device))[0]
    

    prompt_embeds = torch.cat([uncond_embeddings, cond_embeddings])
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float16, device=device)
    latents = latents * scheduler.init_noise_sigma
    guidance_scale = 7.5
    unet = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to(device)
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = unet(
            latent_model_input,  
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    vae = AutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to(device)
    latents = latents / vae.config.scaling_factor  

    # tensor_image = vae.decode(latents, return_dict=False, generator=generator)[0]
    # image_processor = VaeImageProcessor(vae_scale_factor=8)
    # pil_image = image_processor.postprocess(tensor_image, output_type="pil", do_denormalize=[True])[0]

    tensor_image = vae.decode(latents, return_dict=False, generator=generator)[0][0]
    tensor_image = (tensor_image * 0.5 + 0.5).clamp(0, 1)
    numpy_image = tensor_image.cpu().permute(1, 2, 0).float().numpy()
    pil_image = Image.fromarray((numpy_image * 255).round().astype("uint8"))
    pil_image.save("images/sd15_cat_without_pipline.png")

