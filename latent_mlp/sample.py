import torch
from argparse import ArgumentParser
from model import UNet, residualMLP
import numpy as np
import os

def main(args):

    # Config
    # 실행하는 시간에 따라 다르게 설정해야함
    # e.g. ) /media/dataset1/project/jiwon/noise2noise/latent_mlp/results/upscaling/model_0epoch.pt
    exp_name = args.model_path.split('/')[-2]
    model_name = args.model_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join('./sample/', exp_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load Model
    model = UNet(4,4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    model_path = args.model_path
    model.load_state_dict(torch.load(model_path))


    for i in range(args.target_size):
        noise = torch.randn(1, 4, 128, 128).to(device)
        with torch.no_grad():
            pred_noise = model(noise)

        np.save(f"{save_dir}/{i}_noise.npy", noise.cpu().numpy())
        np.save(f"{save_dir}/{i}_pred.npy", pred_noise.cpu().numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target_size', type=int, default=100)
    parser.add_argument('--model_path', type=str, default='./sample/model_99epoch.pt')

    args = parser.parse_args()
    main(args)