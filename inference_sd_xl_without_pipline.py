import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from PIL import Image
from tqdm.auto import tqdm
with torch.no_grad():
    device = "cuda:0"
    num_inference_steps = 20
    prompt = "a cute cat"
    seed= 42
    generator = torch.Generator(device).manual_seed(seed)
    scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps)
    tokenizer_1 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    tokenizers = [tokenizer_1, tokenizer_2] 
    text_encoders = [text_encoder_1,  text_encoder_2]
    prompts = [prompt, prompt]
    prompt_embeds_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        cond_input = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(cond_input.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    # negative_prompt = ""
    # uncond_tokens = [negative_prompt, negative_prompt]
    # negative_prompt_embeds_list = []
    # for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
    #     max_length = prompt_embeds.shape[1]
    #     uncond_input = tokenizer(
    #         [negative_prompt],
    #         padding="max_length",
    #         max_length=max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
    #     negative_pooled_prompt_embeds = negative_prompt_embeds[0]
    #     negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
    #     negative_prompt_embeds_list.append(negative_prompt_embeds)
    # negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, 4, 128, 128), dtype=torch.float16, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma
    guidance_scale = 5.0
    add_text_embeds = pooled_prompt_embeds
    crops_coords_top_left = (0, 0)
    text_encoder_projection_dim = 1280
    original_size = (1024, 1024)
    target_size = (1024, 1024)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float16, device=device)
    negative_add_time_ids = add_time_ids
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16).to(device)
    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        noise_pred = unet(
            latent_model_input,  
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
    latents = latents / vae.config.scaling_factor  
    tensor_image = vae.decode(latents, return_dict=False, generator=generator)[0][0]
    tensor_image = (tensor_image * 0.5 + 0.5).clamp(0, 1)
    numpy_image = tensor_image.cpu().permute(1, 2, 0).float().numpy()
    pil_image = Image.fromarray((numpy_image * 255).round().astype("uint8"))
    pil_image.save("images/sdxl_cat_without_pipline.png")
