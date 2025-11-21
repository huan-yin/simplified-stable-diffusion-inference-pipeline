import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.nn import functional as F
import cv2

class AttendExciteAttnProcessor:
    def __init__(self):
        super().__init__()
        self.attn_map = None

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # only need to store attention maps during the Attend and Excite process
        if attention_probs.requires_grad:
            self.attn_map = attention_probs

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
    attn_res,
    latents: torch.Tensor,
    indices: List[int],
    loss: torch.Tensor,
    threshold: float,
    text_embeddings: torch.Tensor,
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
        unet(latents, t, encoder_hidden_states=text_embeddings).sample
        unet.zero_grad()

        attention_maps = get_net_attn_map(attn_res)
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
    _ = unet(latents, t, encoder_hidden_states=text_embeddings).sample
    unet.zero_grad()

    # Get max activation value for each subject token
    attention_maps = get_net_attn_map(attn_res)
    max_attention_per_index = compute_max_attention_per_index(
        attention_maps=attention_maps,
        indices=token_indices,
    )
    loss = compute_loss(max_attention_per_index)
    print(f"\t Finished with loss of: {loss}")
    return loss, latents, max_attention_per_index


def view_images(images,
                num_rows: int = 1,
                offset_ratio: float = 0.02) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def text_under_image(image: np.ndarray, text: str, text_color = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def show_cross_attention(prompt: str,
                         attn_res,
                         tokenizer,
                         indices_to_alter: List[int],
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = get_net_attn_map(attn_res).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((16 ** 2, 16 ** 2)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    return view_images(np.stack(images, axis=0))




attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))
    return unet

def get_net_attn_map(attn_res):
    
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        if attn_map.shape[1] == attn_res[0] * attn_res[1]:
            net_attn_maps.append(attn_map) 

    net_attn_maps = torch.stack(net_attn_maps, dim=0)
    net_attn_maps = net_attn_maps.mean(dim=0).mean(dim=0)
    net_attn_maps = net_attn_maps.reshape(attn_res[0], attn_res[1], net_attn_maps.shape[1])
    return net_attn_maps


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
    
    prompt = "an elephant with a crown"
    run_standard_sd = False
    cond_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    cond_embeddings = text_encoder(cond_input.input_ids.to(device))[0]
    prompt_embeds = torch.cat([uncond_embeddings, cond_embeddings])
    seed = 21
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float16, device=device)
    latents = latents * scheduler.init_noise_sigma
    guidance_scale = 7.5

    unet = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to(device)
    attn_res = (16, 16)

    original_attn_proc = unet.attn_processors  
    attn_procs = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is not None:
            attn_procs[name] = AttendExciteAttnProcessor()
        else:
            attn_procs[name] = original_attn_proc[name]
  
    unet.set_attn_processor(attn_procs)
    unet = register_cross_attention_hook(unet)


    scale_range = np.linspace(1.0, 0.5, len(scheduler.timesteps))
    scale_factor = 20
    step_size = scale_factor * np.sqrt(scale_range)
    token_indices = [2, 5]
    thresholds = {0: 0.05, 10: 0.5, 20: 0.8}
    max_iter_to_alter = 25
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            unet(
                latents,  
                t,
                encoder_hidden_states=cond_embeddings,
                return_dict=False,
            )[0]
            unet.zero_grad()

            attention_maps = get_net_attn_map(attn_res)
         
            if not run_standard_sd:
                max_attention_per_index = compute_max_attention_per_index(
                    attention_maps=attention_maps,
                    indices=token_indices,
                )
            

                loss = compute_loss(max_attention_per_index=max_attention_per_index)

                # If this is an iterative refinement step, verify we have reached the desired threshold for all
                if i in thresholds.keys() and loss > 1.0 - thresholds[i]:
                    loss, latents, max_attention_per_index = perform_iterative_refinement_step(
                        unet=unet,
                        attn_res=attn_res,
                        latents=latents,
                        indices=token_indices,
                        loss=loss,
                        threshold=thresholds[i],
                        text_embeddings=cond_embeddings,
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



    tensor_image = vae.decode(latents, return_dict=False, generator=generator)[0][0]
    tensor_image = (tensor_image * 0.5 + 0.5).clamp(0, 1)
    numpy_image = tensor_image.cpu().permute(1, 2, 0).float().numpy()
    pil_image = Image.fromarray((numpy_image * 255).round().astype("uint8"))
    # pil_image.save(save_path)
    visual_image = show_cross_attention(prompt, attn_res, tokenizer, token_indices, orig_image=pil_image)
    save_path = "images/ours_sd15_attend_and_excite_show_attention.png"
    visual_image.save(save_path)

