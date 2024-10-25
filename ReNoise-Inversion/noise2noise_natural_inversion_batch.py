import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from src.schedulers.ddim_scheduler import MyDDIMScheduler

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

from utils import resize_image, my_collate_fn
import json

def main():
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/'
    save_dir_initial = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/initial/'
    save_dir_inversion = f'/media/dataset1/project/jiwon/ReNoise-Inversion/{args.name}/inversion/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_initial, exist_ok=True)
    os.makedirs(save_dir_inversion, exist_ok=True)
    batch_size = args.batch_size
    # Save args
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    # CoCo Dataset Load
    cap = datasets.CocoCaptions(root = '/media/dataset1/COCO2014/images/train2014',
                        annFile = '/media/dataset1/COCO2014/annotations/captions_train2014.json',
                        transform=transforms.PILToTensor()) # TODO : CenterCrop, Resize to 1024x1024
    dataloader = torch.utils.data.DataLoader(cap, batch_size=batch_size, shuffle=False, collate_fn = my_collate_fn())

    # Model Load

    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
    # pipe_inversion = get_pipe_inversion(model_type, scheduler_type, device=device)
    # pipe_inference = None
    config = RunConfig(model_type = model_type,
                        num_inference_steps = args.steps,
                        num_inversion_steps = args.steps,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 7865)

    if args.trailing:
        inf_scheduler_config = pipe_inversion.scheduler.config
        inf_scheduler_config['timestep_spacing'] = 'trailing'
        inf_scheduler_config['_use_default_values'] = [x for x in inf_scheduler_config['_use_default_values'] if x != 'rescale_betas_zero_snr']
        new_scheduler = MyDDIMScheduler.from_config(inf_scheduler_config )

        pipe_inference.scheduler = new_scheduler
        pipe_inversion.scheduler = new_scheduler


    # COCO Image, COCO Caption 사용, CFG 없이 Inversion
    for i, (imgs_pil, txts, hw_list, c_top_left_list) in tqdm(enumerate(dataloader)):

        # Save resized image
        for j, img_pil in enumerate(imgs_pil):
            idx = i * batch_size + j
            img_pil.save(os.path.join(save_dir_initial, f"{idx}_img.jpg"))

        # Save coordinate information
        for j, (hw, c_top_left) in enumerate(zip(hw_list, c_top_left_list)):
            idx = i * batch_size + j
            json_dict = {
                "original_size": hw,
                "crops_coords_top_left": c_top_left
            }
            with open(os.path.join(save_dir_initial, f"{idx}.json"), 'w') as f:
                json.dump(json_dict, f, indent=4)

        # Save prompt
        for j, txt in enumerate(txts):
            idx = i * batch_size + j
            with open(os.path.join(save_dir_initial, f"{idx}.txt"), 'w') as f:
                f.write(txt)

        # Save perturbed noise for each image
        input_images = []
        for input_image in imgs_pil:
            input_image = transforms.PILToTensor()(input_image).to(device)
            input_image_transformed = 2.0 * (input_image / 255.0) - 1.0 # Normalize to [-1, 1]
            noise = torch.randn(input_image.shape).to(device)
            perturbed_noise = pipe_inversion.scheduler.add_noise(input_image_transformed, noise, pipe_inversion.scheduler.timesteps[0]) # pipe_inversion.scheduler.timesteps[0] : 981
            np.save(os.path.join(save_dir_initial, f"{i}.npy"), perturbed_noise.cpu().numpy())

        # Input은 다음 두가지 중 하나이어야 함 (1) PIL Image (2) [0,1] Normalized Tensor
        for input_image in imgs_pil:
            input_image = transforms.PILToTensor()(input_image).to(device)
            input_image_transformed = input_image / 255.0 # Normalize to [0, 1]
            input_images.append(input_image_transformed)
        input_images = torch.stack(input_images)

        img, inv_latent, noise, all_latents = invert(input_images,
                                       txts,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference if pipe_inference is not None else None,
                                       original_size=hw_list[0],
                                        crops_coords_top_left=c_top_left_list[0],
                                       do_reconstruction=False)

        print(f"inv_latent.shape: {inv_latent.shape}")
        for j, latent in enumerate(inv_latent):
            idx = i * batch_size + j
            np.save(os.path.join(save_dir_inversion, f"{idx}.npy"), latent.cpu().numpy())


        # Save reconstructed image
        test_prompt = 'a photo of a cat'
        rec_images = pipe_inference(image = inv_latent,
                                # prompt = [test_prompt] * len(txts),
                                prompt = txts,
                                denoising_start=0.0,
                                num_inference_steps = config.num_inference_steps,
                                guidance_scale = 0.0,
                                original_size=hw_list[0],
                                crops_coords_top_left=c_top_left_list[0],
                                )# .images[0]

        for j, image in enumerate(rec_images.images):
            idx = i * batch_size + j
            image.save(os.path.join(save_dir_inversion, f"{idx}_recon.jpg"))





if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--name", type=str, default="coco_latents")
    args.add_argument("--steps", type=int, default=20)
    args.add_argument("--trailing", action='store_true', default=False, help="Use trailing noise")
    args = args.parse_args()
    main()