import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
import torch
import torch.distributed as dist
import PIL

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import argparse
import warnings

from tqdm import tqdm

from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes, get_pipe_inversion
from src.config import RunConfig

from main import run as invert

import numpy as np

from utils import resize_image
import json

def main(args):
    # Config
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/'
    save_dir_initial = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/initial/'
    save_dir_inversion = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/inversion/'
    os.makedirs(save_dir_initial, exist_ok=True)
    os.makedirs(save_dir_inversion, exist_ok=True)

    # CoCo Dataset Load
    cap = datasets.CocoCaptions(root = '/media/dataset1/COCO2014/images/train2014',
                        annFile = '/media/dataset1/COCO2014/annotations/captions_train2014.json',
                        transform=transforms.PILToTensor()) # TODO : CenterCrop, Resize to 1024x1024
    cap = torch.utils.data.Subset(cap, range(0, 1000))

    # Model Load
    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)


    config = RunConfig(model_type = model_type,
                        num_inference_steps = args.steps,
                        num_inversion_steps = args.steps,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 7865)

    # COCO Image, COCO Caption 사용, CFG 없이 Inversion

    ##### Create 50 Random seeds
    import random
    import time
    # Set random seed depending on current time
    # Get current time in seconds (or milliseconds)
    current_time = int(time.time())  # or int(time.time() * 1000) for milliseconds

    # Set the seed using the current time
    random.seed(current_time)
    random_seeds = [random.randint(0, 2**32 - 1) for _ in range(args.num_random_seeds)]


    # Training loop
    for i, (input_image, captions) in tqdm(enumerate(cap)):
        for j, seed in enumerate(random_seeds):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            idx = i * len(random_seeds) + j

            prompt = captions[0] # 첫번째 Caption 사용
            # save prompt
            with open(os.path.join(save_dir_initial, f"{idx}_prompt.txt"), 'w') as f:
                f.write(prompt)

            # Generate initial noise
            initial_noise = torch.randn(1,4,128,128).to(device).to(torch.float16)
            # Save initial noise
            np.save(os.path.join(save_dir_initial, f"{idx}.npy"), initial_noise.cpu().numpy())


            # Generate image using prompt
            img_pil = pipe_inference(image = initial_noise,
                                    prompt = prompt,
                                    denoising_start=0.0,
                                    num_inference_steps = config.num_inference_steps,
                                    guidance_scale = args.guidance_scale).images[0]

            # Save image
            img_pil.save(os.path.join(save_dir_initial, f"{idx}_img.jpg"))

            # Inversion
            img, inv_latent, noise, all_latents = invert(img_pil,
                                        prompt,
                                        config,
                                        pipe_inversion=pipe_inversion,
                                        pipe_inference=pipe_inference,
                                        #    original_size=original_size,
                                            # crops_coords_top_left=crops_coords_top_left,
                                        do_reconstruction=False)

            np.save(os.path.join(save_dir_inversion, f"{idx}.npy"), inv_latent.cpu().numpy())


            # Save reconstructed image
            rec_image = pipe_inference(image = inv_latent,
                                    prompt = prompt,
                                    denoising_start=0.0,
                                    num_inference_steps = config.num_inference_steps,
                                    guidance_scale = 0.0).images[0]

            rec_image.save(os.path.join(save_dir_inversion, f"{idx}_img.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="coco_latents")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_random_seeds", type=int, default=50)
    args = parser.parse_args()
    main(args)