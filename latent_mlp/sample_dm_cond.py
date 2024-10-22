import torch
from argparse import ArgumentParser
from model import UNet, residualMLP
import numpy as np
import os
from model import create_unet_dm
from diffusers import DDIMScheduler, DDIMPipeline
from diffusers.pipelines.ddim.pipeline_ddim_condition import DDIMCondPipeline

def main(args):

    # Config
    # 실행하는 시간에 따라 다르게 설정해야함
    # /media/dataset1/project/jiwon/noise2noise/latent_mlp/results/unet_dm/unet/diffusion_pytorch_model.safetensors
    exp_name = args.model_path.split('/')[-1]
    # model_name = args.model_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join('./sample/unet_dm_cond4', exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")


    # Scheduler
    pipe = DDIMCondPipeline.from_pretrained(args.model_path)
    prompt = "A corgi riding a skateboard"
    # prompt = "a few small loaves of bread in a basket"
    pipe = pipe.to('cuda')


    for i in range(args.target_size):
        noise = torch.randn(1, 4, 128, 128).to(pipe.device)
        with torch.no_grad():
            pred_noise = pipe(batch_size=noise.shape[0],
                              latents=noise,
                              prompt=prompt,
                              num_inference_steps=50,
            )[0]

        np.save(f"{save_dir}/{i}_noise.npy", noise.cpu().numpy())
        np.save(f"{save_dir}/{i}_pred.npy", pred_noise.cpu().numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target_size', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='./sample/model_99epoch.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--validation_every", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet_dm"])
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args)