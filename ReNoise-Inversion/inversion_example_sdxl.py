import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("example_images/image.jpg").convert("RGB")#.resize((1024, 1024))
prompt = "a woman with umbrella"

config = RunConfig(model_type = model_type,
                    num_inference_steps = 20,
                    num_inversion_steps = 20,
                    num_renoise_steps = 1,
                    scheduler_type = scheduler_type,
                    perform_noise_correction = False,
                    seed = 7865)

_, inv_latent, _, all_latents = invert(input_image,
                                       prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                       do_reconstruction=False)
print(f"inv_latent.shape: {inv_latent.shape}")
import ipdb; ipdb.set_trace()
rec_images = pipe_inference(image = inv_latent,
                           prompt = prompt,
                        #    denoising_start=0.0,
                           num_inference_steps = config.num_inference_steps,
                           guidance_scale = 0.0)#.images[0]

for i, image in enumerate(rec_images.images) : image.save(f"{i}_recon.jpg")
