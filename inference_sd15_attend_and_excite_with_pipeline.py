import torch
from diffusers import StableDiffusionAttendAndExcitePipeline
import logging

logger_name = "diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite"


pipeline_logger = logging.getLogger(logger_name)


pipeline_logger.setLevel(logging.INFO)


pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda:0")

prompt = "a cat and a dog"

token_indices = [2, 5]

seed = 0

generator = torch.Generator("cuda:0").manual_seed(seed)

images = pipe(
    prompt=prompt,
    token_indices=token_indices,
    guidance_scale=7.5,
    generator=generator,
    num_inference_steps=50,
    max_iter_to_alter=25,
).images

image = images[0]
image.save(f"images/sd15_attend_and_excite_with_pipeline.png")