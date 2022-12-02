# make sure you're logged in with `huggingface-cli login`
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
YOUR_TOKEN=""
#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN,revision="fp16", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4",
	revision="fp16", torch_dtype=torch.float16, 
	use_auth_token=True
).to("cuda")


code = 1
num_steps=50 #Default value
generator = torch.Generator("cuda:2").manual_seed(1024)

if code == 1:
	path = "images/qualitative/"
	
	captionfile = path + "captions.txt"
	
	count = 0
	
	with open(captionfile, 'r') as readfile:
	
		for caption in readfile:
		
			prompt = caption.rstrip()
			print(prompt)
	
			#The other parameter in the pipeline call is guidance_scale. It is a way to increase the adherence to the conditional signal which in this case is text as well as overall sample quality. In simple terms classifier free guidance forces the generation to better match with the prompt. Numbers like 7 or 8.5 give good results, if you use a very large number the images might look good, but will be less diverse. 
			start = time.time()
			
			for i in range(5):
				with autocast("cuda"):
					#image = pipe(prompt,guidance_scale=5.5, generator=generator,num_inference_steps=num_steps).images[0]
					image = pipe(prompt,guidance_scale=7.5,num_inference_steps=num_steps).images[0]  
		
				image.save(f"images/qualitative/bias/image{count}_bias.jpg")
				count += 1
			
			end = time.time()
			
			ttime = float("{:.2f}".format(end - start))
			print(f"Time taken = {ttime}")
			count += 1
			
			#if count == 1:
			#	break
else:
	num_rows = 1
	num_cols = 2

	prompt = ["a photograph of an astronaut riding a horse"] * num_cols

	all_images = []
	for i in range(num_rows):
	  images = pipe(prompt,guidance_scale=7.5,num_inference_steps=num_steps).images
	  all_images.extend(images)

	grid = image_grid(all_images, rows=num_rows, cols=num_cols)
	grid.save("images/grid_astronaut_rides_horse.png")

### Generate non-square images

#Stable Diffusion produces images of `512 × 512` pixels by default. But it's very easy to override the default using the `height` and `width` arguments, so you can create rectangular images in portrait or landscape ratios.

#The best way to create non-square images is to use `512` in one dimension, and a value larger than that in the other one.

#prompt = "a photograph of an astronaut riding a horse"

#image = pipe(prompt, height=512, width=768).images[0]
#image.save("images/astronaut_rides_horse.png")



