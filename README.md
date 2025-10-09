# simplified-stable-diffusion-inference-pipeline

## Why simplify

Using the StableDiffusionPipeline provided by the diffusers library, we can generate images from text with just a few lines of code. However, if we want to explore its detailed inference process, the relevant implementation code of StableDiffusionPipeline is quite cumbersome—this leaves us with a lack of understanding of how its inference works. To address this, I referred to the code implementation of StableDiffusionPipeline and simplified it as much as possible. The simplified code more clearly demonstrates the inference process of Stable Diffusion. Meanwhile, we ensure that the generation results before and after simplification are exactly consistent.


## How to use our code

### Prepare conda environment
```
conda env create -f environment.yml

conda activate simplified_sd
```

### Run the inference code with pipline
```
python inference_sd_15_with_pipline.py
```

### Run the inference code without pipline (our simplified code)
```
python inference_sd_15_without_pipline.py
```

### Verify the consistency of the generated result with/without pipeline
```
python calculate_image_difference.py
```