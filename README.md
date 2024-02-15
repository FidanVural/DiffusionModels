
## TEXT TO IMAGE STABLE DIFFUSION
By using text-to-image pretrained model, you can generate photos from prompts. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) library has pretrained models for generating images. You can use the basic code below for starting image generation.

 ```python
    import torch
    from diffusers import StableDiffusionPipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    generator = torch.Generator(device="cuda").manual_seed(30)
    
    prompt = "Black white cat with a hat, digital art"
    negative_prompt = "ugly, distorted face, poor details, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark, text, signature, missing arms, missing legs, lying down"
    
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=7.5).images[0]
    
    image
```

You can generate photos using this code. Moreover, you can get better images tweaking the hyperparameters. There are lots of hyperparameters and you can observe some results below what happens when we tweak these hyperparameters. Also, if you have a memory problem, you can use this line `pipeline.enable_sequential_cpu_offload()`.

### CODES
You can take a look [colab notebook](https://colab.research.google.com/drive/1IuiHKICugKSJogobkw100naEkcKG5mJU?usp=sharing) about text to image stable diffusion v1.5 hyperparameters, models and schedulers and try it :)

### 1) Text To Image Stable Diffusion v1.5 Hyperparameters

#### - Guidance Scale
Let's begin with the `guidance_scale` hyperparameter. The guidance_scale determines the influence of prompt on image generation. If guidance_scale is set to lower values, the model tends to be more creative to generate image. Conversely, the model tends to be stricter to follow the prompt if guidance_scale is set to be higher values. Default value of it is 7.5. You can observe the changes of created images based on guidance_scale.

<p align="center">
  <img width="1000" height="180" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/006a35f0-3db6-4c2a-9585-0fa36fa9aab0">
</p> 
 
#### - Negative Prompt
By using this hyperparameter, you will be able to remove what you do not want to see in generated image. Additionally, you can change both style and content of the image using negative_prompt. You use the same prompt and seed but you add `negative_prompt` to get the imgae you want. Let's take a look to examples of prompts and images below.

Firsty, I used "ugly, poor details" negative prompt and obtained this image.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/3380d22a-8c7c-44eb-9da8-8b1d6cafa054">
</p> 

Then, I tried "ugly, poor details, distorted face, deformed, big nose, bad art, poorly drawn feet, poorly drawn face" negative prompt and obtained this image.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/f7200f5b-5a1d-4059-ac3e-add9aa44f798">
</p> 

I noticed that I don't want the image with signature or some text on it. That's why, I tried this prompt "ugly, poor details, distorted face, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark, signature, text" and got this image.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/0b6b397b-d3f1-4926-8c24-7546688b6af6">
</p> 

Then I tried "ugly, distorted face, poor details, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark++, text++, signature++, missing arms, missing legs, lying down" this prompt. I used "++" for giving more importance to these words. We called this `prompt weight`. You can take a look https://dev.dezgo.com/guides/prompt-weighting/ and https://getimg.ai/guides/guide-to-stable-diffusion-prompt-weights. You can see the result below. 

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/7dc22de3-4ff7-4a1e-9fd4-6200cb724397">
</p>

#### - Generator
If you want to generate same image every time, you can use generator with a seed. You can set seed like this `generator = torch.Generator(device="cuda").manual_seed(30)` in the generator.

#### - Height & Width
By changing height and width, you can change the size of the image. 

#### - Number of Inference Steps
`num_inference_steps` represents the number of denosing steps. If you choose bigger step number, you can obtain a higher-quality image but the process will be slower. The default value of num_inference_steps is 50. You can observe the results.

<p align="center">
  <img width="800" height="300" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/f472ac6f-6bae-4175-988b-2b6a4e4ed265">
</p> 

#### - Number of Images per Prompt
It determines the number of images for every run. Default value of `num_images_per_prompt` is 1.

You can take a look to https://huggingface.co/docs/diffusers/v0.13.0/en/api/pipelines/stable_diffusion/text2img for more information. 


### 2) Text To Image Stable Diffusion v1.5 Models
You can try lots of different Stable Diffusion v1.5 models in https://huggingface.co/models?other=diffusers%3AStableDiffusionPipeline. Additionally, some outputs of the models below are shown. For instance, the first image includes more realistic effects while the second image looks like Vincent Van Gogh's paintings.

Prompt: "portrait of an pretty ancient woman warrior with tribal makeup detailed, dramatic lighting, mountainous backgrounds, high resolution"

Negative prompt: "ugly, distorted face, deformed, bad art, poorly drawn face, amateur, beginner, blurry, signature, watermark"

<p align="center">
  <img width="900" height="300" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/222cbcf9-012a-4ae3-8a4d-5e8bc74ceead">
</p> 


### 3) Text To Image Stable Diffusion v1.5 Schedulers
Schedulers are used for denoising process of stable diffusion. Scheduler plays an important role in the denoising process because every step's noise level is different from each other. You can try different scheduler algorithms which exist in the diffuser library. Generally, there is a trade-off between speed and quality.

### 4) Text To Image Stable Diffusion v1.5 Prompt 
Providing a good and detailed prompt to models is important to obtain the desired image. I used Dream Shaper model to generate images. Now let's look some prompts.

If you give just subject as a prompt such as `prompt = "A witch"`, you can obtain below image. 

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/99d04a46-48e2-4fc2-9d92-1d16433b3bab">
</p> 

Then you can extend your prompt by adding more detail and a style. You can see the result below of this prompt: `prompt = "A beautiful and powerful witch, futuristic"`

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/5d18736d-a8f9-4468-8f80-db6d6ecd6e86">
</p> 

You can add some resolution keywords like `prompt = "A beautiful and powerful witch, highly detailed, futuristic"`.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/dbacbf4d-0db4-4ede-b117-e52aadd60328">
</p> 

It can be added lighting to image such as `prompt = "A beautiful and powerful witch, highly detailed, surrounded by clouds at night, futuristic"`.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/001c9d46-8d65-4c42-955d-ca4756ad07a7">
</p> 

Additionally, you can give more detail about the image that you want. In my prompt, I added some details about hair, dress and background. Finally, my prompt is `prompt = "Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, witch hat, bats, mountain background, highly detailed, surrounded by clouds at night, futuristic"`.

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/7ac4c81b-10ef-4dc9-95b2-34b8dbad5d79">
</p> 

Finally, you can take a look at the final image, which includes a negative prompt.

`prompt = "Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, windy, witch hat, bats, mountain background, highly detailed, surrounded by clouds at night, futuristic"`

`negative_prompt = "ugly, distorted, deformed, mutation, out of frame"`

<p align="center">
  <img width="350" height="350" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/6faf67d9-37d5-4fd6-a773-e4fd34b0302d">
</p> 

You can find a notebook about techniques of prompt in the CODES section of readme and try other styles, lightings, additional keywords, etc.


## IMAGE TO IMAGE STABLE DIFFUSION

Image-to-image is quite similar to text to image, with the main difference being the addition of an initial image alongside the prompt. You can take a look the code below.

 ```python
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    
    import requests
    from PIL import Image
    from io import BytesIO

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(30)
    
    url = "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg" # random cat image
    
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    STYLE = "dark art"
    RESOLUTION = "highly detailed, "
    LIGHTING = "surrounded by clouds at night, "
    
    prompt = "Portrait of a cat wizard, white color, serious face, putting on a black cloak" + RESOLUTION + LIGHTING + STYLE
    negative_prompt = "ugly, poorly drawn, bad anatomy, mutation, signature, text, watermark"
    
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, generator=generator).images[0]

    image
```

Additionally, you can check the [img2img](https://github.com/FidanVural/DiffusionModels/tree/master/img2img) directory to see results and explore various hyperparameters.

## STABLE DIFFUSION INPAINTING
If you want to modify certain portions of an image, you can use inpainting models. These models realize inpainting of images by using a mask.

 ```python
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    
    import requests
    from PIL import Image
    from io import BytesIO
   
    # Download the model
    pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    generator = torch.Generator("cuda").manual_seed(92)

    image_path = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    
    response_i = requests.get(image_path)
    init_image = Image.open(BytesIO(response_i.content)).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    mask_path = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    
    response_m = requests.get(mask_path)
    mask_image = Image.open(BytesIO(response_m.content)).convert("RGB")
    mask_image = mask_image.resize((512, 512))
    
    prompt = "a white cat sitting on a bench"
    
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
    image_grid([init_image, mask_image, image], rows=1, cols=3)
```

You can see the initial image, the mask and the generated image produced by a stable diffusion inpainting model below. Also, you can check the [img2img](https://github.com/FidanVural/DiffusionModels/tree/master/inpainting) to explore other inpainting models.

<p align="center">
  <img width="900" height="300" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/39a5180f-2076-475e-92df-64b243a61668">
</p> 


## STABLE DIFFUSION XL (SDXL)
The Stable Diffusion XL model ia larger model than v1.5 that it can be used for text to image, image to image and inpatinting tasks. The SDXL model generates images of size 1024x1024. 

 ```python
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(286773456)

    STYLE = "futuristic, fantasy"
    RESOLUTION = "highly detailed, "
    LIGHTING = "surrounded by clouds at night, "
    prompt = "Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, windy, witch hat, bats, mountain background" + RESOLUTION + LIGHTING + STYLE
    
    negative_prompt = "ugly, distorted, deformed, mutation, out of frame"
    
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator).images[0]
    
    image
```

First of all, we'll compare the results of SDXL and stable diffusion v1.5 models. You can see the comparison results below.

prompt: `"Black white cat with a hat, digital art"`

negative_prompt: `"ugly, distorted face, poor details, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark, text, signature, missing arms, missing legs, lying down"`


![result_neg_6](https://github.com/FidanVural/DiffusionModels/assets/56233156/27617450-2759-435e-9aae-1caf19fbf44a) | ![cat_xl](https://github.com/FidanVural/DiffusionModels/assets/56233156/2a7b9c75-fe7c-4be9-8961-2c4b68df2b65)
:------------------------:|:-------------------------:
Stable Diffusion v1.5              |  SDXL

prompt: `"Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, windy, witch hat, bats, mountain background" + RESOLUTION + LIGHTING + STYLE`

negative_prompt: `"ugly, distorted, deformed, mutation, out of frame"`

![last](https://github.com/FidanVural/DiffusionModels/assets/56233156/bf380248-9f73-4375-80e3-0ae479902d3e) | ![witch_xl](https://github.com/FidanVural/DiffusionModels/assets/56233156/614702e1-8faa-484e-adb4-dc0fda7b292d)
:------------------------:|:-------------------------:
Stable Diffusion v1.5              |  SDXL

Now we can explore some hyperparameters. The hyperparameters that I tried first are **prompt_2** and **negative_prompt_2**. SDXL model have two text-encoders, so we can pass different prompts each of them. Let's take a look at the results 🚀 Also, I changed the prompt and the prompt_2 to see the effects of prompts' order to the image :)

prompt: `"Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, windy, witch hat, bats, mountain background, highly detailed, surrounded by clouds at night, futuristic, fantasy"`

prompt_2: `"Portrait of a beautiful and powerful witch, highly detailed, surrounded by clouds at night, cyberpunk"`

<p align="center">
  <img width="512" height="512" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/934ccd3c-25e7-4360-acf4-6d8a4a676000">
</p> 

prompt: `"Portrait of a beautiful and powerful witch, highly detailed, surrounded by clouds at night, cyberpunk"`

prompt_2: `"Portrait of a beautiful and powerful witch, wearing a black dress with gemstones, serious eyes, small face, white with highlighted purple hair, windy, witch hat, bats, mountain background, highly detailed, surrounded by clouds at night, futuristic, fantasy"`

<p align="center">
  <img width="512" height="512" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/3275dcfc-d7c8-4937-9184-44584a1d1d74">
</p> 

Another hyperparameter is **negative_original_size**. You can see the effects of negative_original_size to below images. 

![512](https://github.com/FidanVural/DiffusionModels/assets/56233156/323cc7fe-5910-4b35-9c63-815137693698) | ![256](https://github.com/FidanVural/DiffusionModels/assets/56233156/bb8ff9c0-aad6-460a-9bbe-e0243dc4ba2c)
:------------------------:|:-------------------------:|
negative_original_size: 512x512   |  negative_original_size: 256x256

If you want, you can obtain more information from https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl.

### 1) Stable Diffusion XL Models

There are lots of Stable Diffusion XL models that you can try. You can visit https://huggingface.co/models?other=diffusers:StableDiffusionXLPipeline&sort=downloads to see various models. I tried two different models, one being `RealVisXL_V3.0` and the other being `OpenDalleV1.1`. You can see below the images generated by these models, respectively.

![kid](https://github.com/FidanVural/DiffusionModels/assets/56233156/df75b6d7-043e-4636-bb0c-050b98d97079) | ![vatikan](https://github.com/FidanVural/DiffusionModels/assets/56233156/462cc0c0-5a64-4229-87cf-49c5aa99a2c1)


