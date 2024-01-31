
 ### Stable Diffusion v1.5 Text-to-image model

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
