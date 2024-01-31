
 ### Stable diffusion v1.5 Text-to-image model

 ```bash
    from diffusers import AutoPipelineForText2Image
    import torch
    
    pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    pipeline.enable_sequential_cpu_offload() # to reduce memory usage

    generator = torch.Generator(device="cuda").manual_seed(30)
    
    prompt = "a woman with black hat, vivid colors"
    
    image = pipeline(prompt, generator=generator).images[0] 
    
    image.save("result.png")
```
You can generate photos using this code. Moreover, you can get better images tweaking the hyperparameters. There are lots of hyperparameters and you can observe some results below what happens when we tweak these hyperparameters. 

#### guidance_scale
Let's begin with the `guidance_scale` hyperparameter. The guidance_scale determines the influence of prompt on image generation. If guidance_scale is set to lower values, the model tends to be more creative to generate image. Conversely, the model tends to be stricter to follow the prompt if guidance_scale is set to be higher values. Default value of it is 7.5. You can observe the changes of created images based on guidance_scale.

<p align="center">
  <img width="1000" height="200" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/006a35f0-3db6-4c2a-9585-0fa36fa9aab0">
</p> 

#### negative_prompt
By using this hyperparameter, you will be able to remove what you do not want to see in generated image. Additionally, you can change both style and content of the image using negative_prompt. You use the same prompt and seed but you add `negative_prompt` to get the imgae you want. Let's take a look to examples of prompts and images below.

Firsty, I used "ugly, poor details" negative prompt and obtained this image.

<p align="center">
  <img src="https://github.com/FidanVural/DiffusionModels/assets/56233156/3380d22a-8c7c-44eb-9da8-8b1d6cafa054">
</p> 

Then, I tried "ugly, poor details, distorted face, deformed, big nose, bad art, poorly drawn feet, poorly drawn face" negative prompt and obtained this image.

<p align="center">
  <img src="https://github.com/FidanVural/DiffusionModels/assets/56233156/f7200f5b-5a1d-4059-ac3e-add9aa44f798">
</p> 
