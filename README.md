
 ### Stable diffusion v1.5 Text-to-image model

 ```bash
    from diffusers import AutoPipelineForText2Image
    import torch
    
    pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    pipeline.enable_sequential_cpu_offload()
    generator = torch.Generator(device="cuda").manual_seed(30)
    
    prompt = "a woman with black hat, vivid colors"
    
    image = pipeline(prompt, generator=generator).images[0]
    
    image.save("result.png")
```
You can generate photos using this code. Moreover, you can get better images tweaking the hyperparameters. There are lots of hyperparameters and you can observe some results below what happens when we tweak these hyperparameters. 

Let's begin with the 'guidance_scale' hyperparameter. The guidance_scale determines the influence of prompt on image generation. If guidance_scale is set to lower values, the model tends to be more creative to generate image. Conversely, the model tends to be stricter to follow the prompt if guidance_scale is set to be higher values. You can observe the changes of created images based on guidance_scale.

<p align="center">
  <img width="500" height="200" src="https://github.com/FidanVural/DiffusionModels/assets/56233156/006a35f0-3db6-4c2a-9585-0fa36fa9aab0">
</p> 



