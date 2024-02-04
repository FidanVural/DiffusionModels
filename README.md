
 ### Stable Diffusion v1.5 Text-to-image Model
By using text-to-image pretrained model, you can generate photos from prompts. 

 ```python
    import torch
    from diffusers import StableDiffusionPipeline
    from compel import Compel
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    #pipeline.enable_sequential_cpu_offload()
    generator = torch.Generator(device="cuda").manual_seed(30)
    
    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    
    prompt = "Black white cat with a hat, digital art"
    
    image = pipeline(prompt=prompt, generator=generator).images[0]
    
    #image.save("result.png")

image
```

You can generate photos using this code. Moreover, you can get better images tweaking the hyperparameters. There are lots of hyperparameters and you can observe some results below what happens when we tweak these hyperparameters. Also, if you don't have any memory problem, you can remove this line `pipeline.enable_sequential_cpu_offload()`.

#### guidance_scale
Let's begin with the `guidance_scale` hyperparameter. The guidance_scale determines the influence of prompt on image generation. If guidance_scale is set to lower values, the model tends to be more creative to generate image. Conversely, the model tends to be stricter to follow the prompt if guidance_scale is set to be higher values. Default value of it is 7.5. You can observe the changes of created images based on guidance_scale.

<p align="center">
  <img width="1000" height="180" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/006a35f0-3db6-4c2a-9585-0fa36fa9aab0">
</p> 
 
#### negative_prompt
By using this hyperparameter, you will be able to remove what you do not want to see in generated image. Additionally, you can change both style and content of the image using negative_prompt. You use the same prompt and seed but you add `negative_prompt` to get the imgae you want. Let's take a look to examples of prompts and images below.

Firsty, I used "ugly, poor details" negative prompt and obtained this image.

<p align="center">
  <img width="450" height="450" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/3380d22a-8c7c-44eb-9da8-8b1d6cafa054">
</p> 

Then, I tried "ugly, poor details, distorted face, deformed, big nose, bad art, poorly drawn feet, poorly drawn face" negative prompt and obtained this image.

<p align="center">
  <img width="450" height="450" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/f7200f5b-5a1d-4059-ac3e-add9aa44f798">
</p> 

I noticed that I don't want the image with signature or some text on it. That's why, I tried this prompt "ugly, poor details, distorted face, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark, signature, text" and got this image.

<p align="center">
  <img width="450" height="450" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/0b6b397b-d3f1-4926-8c24-7546688b6af6">
</p> 

Then I tried "ugly, distorted face, poor details, deformed, big nose, bad art, poorly drawn feet, poorly drawn face, watermark++, text++, signature++, missing arms, missing legs, lying down" this prompt. I used "++" for giving more importance to these words. We called this `prompt weight`. You can take a look https://dev.dezgo.com/guides/prompt-weighting/ and https://getimg.ai/guides/guide-to-stable-diffusion-prompt-weights. You can see the result below. 

<p align="center">
  <img width="450" height="450" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/7dc22de3-4ff7-4a1e-9fd4-6200cb724397">
</p>

#### generator
If you want to generate same image every time, you can use generator with a seed. You can set seed like this `generator = torch.Generator(device="cuda").manual_seed(30)` in the generator.

#### height & width
By changing height and width, you can change the size of the image. 

#### num_inference_steps

`num_inference_steps` represents the number of denosing steps. If you choose bigger step number, you can obtain a higher-quality image but the process will be slower. The default value of num_inference_steps is 50. You can observe the results.

<p align="center">
  <img width="800" height="300" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/f472ac6f-6bae-4175-988b-2b6a4e4ed265">
</p> 

#### num_images_per_prompt
It determines the number of images for every run. Default value of `num_images_per_prompt` is 1.

You can take a look to https://huggingface.co/docs/diffusers/v0.13.0/en/api/pipelines/stable_diffusion/text2img for more information. 


