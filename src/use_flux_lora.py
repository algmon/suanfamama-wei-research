from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('seawolf2357/test-fash2', weight_name='test-fash2.safetensors')
image = pipeline('a model wearing a magenta satin blouse and jeans, standing against a grey background. The blouse has a glossy finish, giving it a luxurious look. [trigger]').images[0]
image.save("magenta_blouse.png")

