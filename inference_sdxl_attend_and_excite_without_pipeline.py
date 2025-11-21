import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from PIL import Image
from tqdm.auto import tqdm
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.nn import functional as F

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed separately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


def compute_max_attention_per_index(
    attention_maps: torch.Tensor,
    indices: List[int],
) -> List[torch.Tensor]:
    """Computes the maximum attention value for each of the tokens we wish to alter."""
    attention_for_text = attention_maps[:, :, 1:-1]
    attention_for_text *= 100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]

    # Extract the maximum values
    max_indices_list = []
    for i in indices:
        image = attention_for_text[:, :, i]
        smoothing = GaussianSmoothing().to(attention_maps.device)
        input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        image = smoothing(input).squeeze(0).squeeze(0)
        max_indices_list.append(image.max())
    return max_indices_list

def compute_loss(max_attention_per_index: List[torch.Tensor]) -> torch.Tensor:
    """Computes the attend-and-excite loss using the maximum attention value for each token."""
    losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
    loss = max(losses)
    return loss


def update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
    """Update the latent according to the computed loss."""
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    latents = latents - step_size * grad_cond
    return latents

def perform_iterative_refinement_step(
    unet,
    attention_store,
    latents: torch.Tensor,
    indices: List[int],
    loss: torch.Tensor,
    threshold: float,
    text_embeddings: torch.Tensor,
    added_cond_kwargs,
    step_size: float,
    t: int,
    max_refinement_steps: int = 20,
):
    """
    Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
    according to our loss objective until the given threshold is reached for all tokens.
    """
    iteration = 0
    target_loss = max(0, 1.0 - threshold)
    while loss > target_loss:
        iteration += 1

        latents = latents.clone().detach().requires_grad_(True)
        unet(latents, t, encoder_hidden_states=text_embeddings,added_cond_kwargs=added_cond_kwargs).sample
        unet.zero_grad()

        attention_maps = attention_store.aggregate_attention(
                from_where=("up", "down", "mid"),
            )
        max_attention_per_index = compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices=token_indices,
        )
           

        loss = compute_loss(max_attention_per_index)

        if loss != 0:
            latents = update_latent(latents, loss, step_size)

        print(f"\t Try {iteration}. loss: {loss}")

        if iteration >= max_refinement_steps:
            print(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
            break

    # Run one more time but don't compute gradients and update the latents.
    # We just need to compute the new loss - the grad update will occur below
    latents = latents.clone().detach().requires_grad_(True)
    _ = unet(latents, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=added_cond_kwargs).sample
    unet.zero_grad()

    # Get max activation value for each subject token
    attention_maps = attention_store.aggregate_attention(
                from_where=("up", "down", "mid"),
            )
    max_attention_per_index = compute_max_attention_per_index(
        attention_maps=attention_maps,
        indices=token_indices,
    )
    loss = compute_loss(max_attention_per_index)
    print(f"\t Finished with loss of: {loss}")
    return loss, latents, max_attention_per_index

with torch.no_grad():
    device = "cuda:0"
    num_inference_steps = 50
    prompt = "a cat and a dog"
    seed= 2
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
    seed = 2
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
    new_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    new_add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    new_add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16).to(device)
    unet.enable_gradient_checkpointing()
    # unet.enable_attention_slicing()
    attn_res = (32, 32)     
    attention_store = AttentionStore(attn_res)
    original_attn_proc = unet.attn_processors  
    attn_procs = {}
    cross_att_count = 0
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)
    unet.set_attn_processor(attn_procs)
    attention_store.num_att_layers = cross_att_count

    scale_range = np.linspace(1.0, 0.5, len(scheduler.timesteps))
    scale_factor = 20
    step_size = scale_factor * np.sqrt(scale_range)
    token_indices = [2, 5]
    thresholds = {0: 0.05, 10: 0.5, 20: 0.8}
    max_iter_to_alter = 25

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            unet(
                latents,  
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            unet.zero_grad()

            attention_maps = attention_store.aggregate_attention(
                from_where=("up", "down", "mid"),
            )
            max_attention_per_index = compute_max_attention_per_index(
                attention_maps=attention_maps,
                indices=token_indices,
            )
           

            loss = compute_loss(max_attention_per_index=max_attention_per_index)

            # If this is an iterative refinement step, verify we have reached the desired threshold for all
            if i in thresholds.keys() and loss > 1.0 - thresholds[i]:
                loss, latents, max_attention_per_index = perform_iterative_refinement_step(
                    unet=unet,
                    attention_store=attention_store,
                    latents=latents,
                    indices=token_indices,
                    loss=loss,
                    threshold=thresholds[i],
                    text_embeddings=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    step_size=step_size[i],
                    t=t,
                )

            # Perform gradient update
            if i < max_iter_to_alter:
                if loss != 0:
                    latents = update_latent(
                        latents=latents,
                        loss=loss,
                        step_size=step_size[i],
                    )
                print(f"Iteration {i} | Loss: {loss:0.4f}")
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        added_cond_kwargs = {"text_embeds": new_add_text_embeds, "time_ids": new_add_time_ids}
        noise_pred = unet(
            latent_model_input,  
            t,
            encoder_hidden_states=new_prompt_embeds,
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
    pil_image.save("images/sdxl_attend_and_excite_without_pipeline.png")
