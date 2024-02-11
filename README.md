
## TEXT TO IMAGE STABLE DIFFUSION v1.5
By using text-to-image pretrained model, you can generate photos from prompts. [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) library has pretrained models for generating images. 

 ```python
    iimport torch
    from diffusers import StableDiffusionPipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
 
    generator = torch.Generator(device="cuda").manual_seed(30)
    
    prompt = "Black white cat with a hat, digital art"
    
    image = pipeline(prompt=prompt, generator=generator).images[0]
    image
```

You can generate photos using this code. Moreover, you can get better images tweaking the hyperparameters. There are lots of hyperparameters and you can observe some results below what happens when we tweak these hyperparameters. Also, if you have a memory problem, you can use this line `pipeline.enable_sequential_cpu_offload()`.

### CODES
Text to image Stable diffusion v1.5 hyperparameters, models and schedulers: https://colab.research.google.com/drive/1IuiHKICugKSJogobkw100naEkcKG5mJU?usp=sharing 
Text to image stable diffusion v1.5 prompt: https://colab.research.google.com/drive/1WCuR04uKpVKZaao3nbmViwj0Dhz80tyu?authuser=1#scrollTo=wPhimm4pDjCI
Image to image stable diffusion v1.5 hyperparameters: https://colab.research.google.com/drive/1Uh_X5b2XQvsZSLdyAHEVxAuXQ3LBt_HG#scrollTo=kSsXEmcHWVze

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
