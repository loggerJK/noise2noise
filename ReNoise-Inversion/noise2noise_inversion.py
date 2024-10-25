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
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

import numpy as np

from utils import resize_image
import json

def main():
    # Config
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

    for i, (input_image, captions) in tqdm(enumerate(cap)):
        prompt = captions[0] # 첫번째 Caption 사용

        # Image processing
        image, original_h, original_w, c_top, c_left = resize_image(input_image, 1024)
        original_size = (original_h, original_w)
        crops_coords_top_left = (c_top, c_left)
        input_image = image.to(device)
        print(f"original_size : {original_size}, crops_coords_top_left : {crops_coords_top_left}")
        print(f"input_image.shape : {input_image.shape}")

        # Save resized image
        input_image_np = input_image.squeeze().cpu().numpy()
        print(f"input_image_np.shape : {input_image_np.shape}")
        input_image_np = np.transpose(input_image_np, (1, 2, 0))
        input_image_pil = Image.fromarray(input_image_np)
        input_image_pil.save(os.path.join(save_dir_initial, f"{i}_img.jpg"))

        # Save original image and crop_coords_top_left as json
        json_dict = {
            "original_size": original_size,
            "crops_coords_top_left": crops_coords_top_left
        }
        with open(os.path.join(save_dir_initial, f"{i}.json"), 'w') as f:
            json.dump(json_dict, f, indent=4)



        # Save perturbed image
        input_image_transformed = 2.0 * (input_image / 255.0) - 1.0
        noise = torch.randn(input_image.shape).to(device)
        perturbed_noise = pipe_inference.scheduler.add_noise(input_image_transformed, noise, pipe_inference.scheduler.timesteps[0])
        np.save(os.path.join(save_dir_initial, f"{i}.npy"), perturbed_noise.cpu().numpy())


        img, inv_latent, noise, all_latents = invert(input_image_pil,
                                       prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                    #    original_size=original_size,
                                        # crops_coords_top_left=crops_coords_top_left,
                                       do_reconstruction=False)

        print(f"inv_latent.shape: {inv_latent.shape}")
        np.save(os.path.join(save_dir_inversion, f"{i}.npy"), inv_latent.cpu().numpy())


        # # Save reconstructed image
        # rec_image = pipe_inference(image = inv_latent,
        #                         prompt = prompt,
        #                         denoising_start=0.0,
        #                         num_inference_steps = config.num_inference_steps,
        #                         guidance_scale = 0.0).images[0]

        # rec_image.save(os.path.join(save_dir_inversion, f"{i}_img.jpg"))





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, default="coco_latents")
    args.add_argument("--steps", type=int, default=20)
    main()