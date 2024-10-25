from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('seawolf2357/test-fash2', weight_name='test-fash2.safetensors')
#image = pipeline('a model wearing a magenta satin blouse and jeans, standing against a grey background. The blouse has a glossy finish, giving it a luxurious look. [trigger]').images[0]
#image.save("magenta_blouse.png")

image = pipeline('a man model wearing a grey 100% organic cotton polo tshirt and jeans, standing against a grey background. The polo tshirt should have a formal style, giving it a formal business look. [trigger]').images[0]
image.save("grey_polo.png")
