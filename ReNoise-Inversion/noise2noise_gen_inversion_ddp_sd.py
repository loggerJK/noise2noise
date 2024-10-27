import os
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
    save_dir = f'./{args.name}/'
    save_dir_initial = os.path.join(save_dir, 'initial/')
    save_dir_inversion = os.path.join(save_dir, 'inversion/')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_initial, exist_ok=True)
    os.makedirs(save_dir_inversion, exist_ok=True)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(f"Device: {device}")

    # CoCo Dataset Load
    dataset_path = '/media/dataset1/COCO2014'
    cap = datasets.CocoCaptions(root = os.path.join(dataset_path, 'images/train2014'),
                        annFile = os.path.join(dataset_path, 'annotations/captions_train2014.json'),
                        transform=transforms.PILToTensor()) # TODO : CenterCrop, Resize to 1024x1024
    
    # if args.num_prompts % dist.get_world_size() != 0:
    #     raise ValueError("num_prompts must be divisible by world_size (Total GPU Number).")
    
    # not_found = np.load('./analysis/not_found_or_error.npy')
    # not_found = not_found.astype(int).tolist()
    # cap = torch.utils.data.Subset(cap, not_found)
    # print(f"Processing {len(not_found)} prompts")
    
    args.num_prompts = len(cap)
    print(f"Total number of prompts: {args.num_prompts}")
    start_idx = args.num_prompts // dist.get_world_size() * rank
    num_prompt_per_gpu = args.num_prompts // dist.get_world_size()
    if rank == dist.get_world_size() - 1:
        if args.num_prompts % dist.get_world_size() != 0:
            num_prompt_per_gpu += args.num_prompts % dist.get_world_size()
    add = 0
    
    cap = torch.utils.data.Subset(cap, range(start_idx + add, start_idx + num_prompt_per_gpu))
    print(f"GPU {rank} processing {start_idx + add} to {start_idx + add + num_prompt_per_gpu - 1}")

    # Model Load
    model_type = Model_Type.SD21
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
    pipe_inversion.set_progress_bar_config(disable=True)
    pipe_inference.set_progress_bar_config(disable=True)


    config = RunConfig(model_type = model_type,
                        num_inference_steps = args.steps,
                        num_inversion_steps = args.steps,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = args.global_seed)

    # COCO Image, COCO Caption 사용, CFG 없이 Inversion

    ##### Create 50 Random seeds
    import random
    import time
    # Set random seed depending on current time
    # Get current time in seconds (or milliseconds)
    current_time = int(time.time())  # or int(time.time() * 1000) for milliseconds


    # Training loop
    progress_bar = tqdm(total=len(cap) * args.num_random_seeds, leave = True if rank == 0 else False)
    for i, (input_image, captions) in tqdm(enumerate(cap)):
        for j in range(args.num_random_seeds):
            
            # seed = int(time.time())
            idx = (start_idx + add + i) * args.num_random_seeds + j
            
            seed = idx
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)


            prompt = captions[0] # 첫번째 Caption 사용
            # save prompt
            with open(os.path.join(save_dir_initial, f"{idx}_prompt.txt"), 'w') as f:
                f.write(prompt)

            # Generate initial noise
            initial_noise = torch.randn(1,4,64,64).to(device).to(torch.float16)
            # Save initial noise
            np.save(os.path.join(save_dir_initial, f"{idx}.npy"), initial_noise.cpu().numpy())


            # Generate image using prompt
            inference_result = pipe_inference(image = initial_noise,
                                    prompt = prompt,
                                    denoising_start=0.0,
                                    strength=1.0,
                                    num_inference_steps = config.num_inference_steps,
                                    guidance_scale = args.guidance_scale,
                                    output_type = "latent",
                                    return_dict = False,
                                    )
            
            img_latent = inference_result[0]
            all_latents = inference_result[1]
            
            # Save latent
            np.save(os.path.join(save_dir_initial, f"{idx}_imglatent.npy"), img_latent.cpu().numpy())
            np.save(os.path.join(save_dir_initial, f"{idx}_alllatents.npy"), all_latents)
            
            # Save image
            img_pil = pipe_inference.vae.decode(img_latent / pipe_inference.vae.config.scaling_factor, return_dict=False)[0].detach().cpu()
            img_pil = pipe_inference.image_processor.postprocess(img_pil, output_type="pil")[0]
            img_pil.save(os.path.join(save_dir_initial, f"{idx}_img.png"))

            # Inversion
            # img, inv_latent, noise, all_latents = invert(img_pil,
            #                             prompt,
            #                             config,
            #                             pipe_inversion=pipe_inversion,
            #                             pipe_inference=pipe_inference,
            #                             #    original_size=original_size,
            #                                 # crops_coords_top_left=crops_coords_top_left,
            #                             do_reconstruction=False)

            # np.save(os.path.join(save_dir_inversion, f"{idx}.npy"), inv_latent.cpu().numpy())


            # Save reconstructed image
            # rec_image = pipe_inference(image = inv_latent,
            #                         prompt = prompt,
            #                         denoising_start=0.0,
            #                         strength=1.0,
            #                         num_inference_steps = config.num_inference_steps,
            #                         guidance_scale = 0.0).images[0]

            # rec_image.save(os.path.join(save_dir_inversion, f"{idx}_img.jpg"))

            progress_bar.update(1)
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="coco_latents_sd")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_random_seeds", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=82783)
    parser.add_argument("--global_seed", type=int, default=7865)
    args = parser.parse_args()
    main(args)