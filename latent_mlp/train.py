import torch
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import residualMLP, UNet
import os
from tqdm.auto import tqdm
from argparse import ArgumentParser
from utils import my_collate_fn, CustomDataset
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from dataclasses import dataclass, asdict, field
from torch.utils.data import Subset

from train_unet_dm import train_unet_dm
from train_unet_dm_gan import train_unet_dm_gan


def chi2_neg_log_prob(x:torch.Tensor):
    x = x.reshape(x.shape[0], -1) # (N, C, H, W) -> (N, C*H*W)
    d = x.shape[1]
    norm = torch.norm(x, dim=1, p=2) # (N, )
    reg = (d-2) * torch.log(norm) - (norm **2) / 2 # (N, )
    const = -d/2 * torch.log(torch.tensor(2)) - torch.lgamma(torch.tensor(d/2)) # scalar
    const = const.to(x.device).detach()
    reg = const + reg
    # print(f"norm.shape : {norm.shape}, neg-reg.shape: {reg.shape}")
    # print(f"norm : {norm.mean()}, neg-reg: {- reg.mean()}")
    return  -(reg.mean()), norm.mean()

def train_unet(model, train_dataloader, val_dataloader, save_dir, writer, device, args):
    # train dataset
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for epoch in tqdm(range(args.start_from_epoch, args.num_epochs)):
        for i, (x, y) in enumerate(train_dataloader):
            step = i + epoch * len(train_dataloader)
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            if args.upscaling:
                y = y *  (torch.norm(x, dim=[1,2,3], p=2) / torch.norm(y, dim=[1,2,3], p=2)).reshape(-1, 1, 1, 1)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            if args.chi2_reg > 0:
                reg = chi2_neg_log_prob(y_pred)
                writer.add_scalar('Loss/chi2_reg', reg.item(), step)
                loss += args.chi2_reg * reg

            writer.add_scalar('Loss/train', loss.item(), step)

            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in tqdm(val_dataloader):
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_y_pred = model(val_x)
                val_loss += criterion(val_y_pred, val_y).item()

        val_loss /= len(val_dataloader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"epoch: {epoch}, val_loss: {val_loss}")
        model.train()


        # Save Model
        if (epoch ) % args.save_every_epochs == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}epoch.pt"))



def main(args):
    #### Config
    save_dir = './results/' + args.run_name
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.model == "unet":
        writer = SummaryWriter(save_dir)
    else :
        writer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    # Save args
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))


    ##### Load Dataset
    import glob
    from pprint import pprint
    folder_path = '/media/dataset1/project/jiwon/noise2noise_dataset/coco_latents/'
    initial_folder_path = os.path.join(folder_path, 'initial')
    inversion_folder_path = os.path.join(folder_path, 'inversion')

    initial_latents = sorted(glob.glob(os.path.join(initial_folder_path , '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    inversion_latents = sorted(glob.glob(os.path.join(inversion_folder_path , '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    prompts = sorted(glob.glob(os.path.join(initial_folder_path , '*_prompt.txt')), key=lambda x: int(x.split('/')[-1].split('_')[0]))

    if args.debug:
        initial_latents = initial_latents[:100]
        inversion_latents = inversion_latents[:100]
        prompts = prompts[:100]

    ####### File loading #######
    if not args.lazy_load:
        initial_latents_loaded = []
        inversion_latents_loaded = []

        ## inversion latents 중에서, 대응하는 inital latent가 있는지 확인하고, 있는 경우에만 로드한다
        # for i, latent in tqdm(enumerate(inversion_latents), total=len(inversion_latents)):
        #     initial_latent = latent.replace('_inversion_latent.pt', '_initial_latent.pt')
        #     if initial_latent in initial_latents:
        #         initial_latents_loaded.append(torch.load(initial_latent).cpu().numpy())
        #         inversion_latents_loaded.append(torch.load(latent).cpu().numpy())

        for i, (initial_latent, inversion_latent) in tqdm(enumerate(zip(initial_latents, inversion_latents)), total=len(inversion_latents)):
            initial_latents_loaded.append(np.load(initial_latent))
            inversion_latents_loaded.append(np.load(inversion_latent))

        
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            with open(prompt, 'r') as f:
                prompts[i] = f.read()

        initial_latents = torch.Tensor(np.array(initial_latents_loaded)).squeeze(1) # (N, C, H, W)
        inversion_latents = torch.Tensor(np.array(inversion_latents_loaded)).squeeze(1) # (N, C, H, W)
        prompts = prompts

    x = initial_latents
    y = inversion_latents
    # print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    # latent_dim = x.shape[1]

    latent_dim = 128

    # train dataset
    percent = 0.99
    def choice(N, percent):
        import random
        tmp = list(range(N))
        random.shuffle(tmp)
        cut = int(N * percent)
        return tmp[:cut], tmp[cut:]
    train_idx, val_idx = choice(len(x), percent)

    train_x = x #[:int(len(x) * percent)]
    train_y = y #[:int(len(y) * percent)]
    train_prompt = prompts #[:int(len(prompts) * percent)]

    val_x = x
    val_y = y
    val_prompt = prompts

    train_dataset = CustomDataset(train_x, train_y, train_prompt)
    val_dataset = CustomDataset(val_x, val_y, val_prompt)

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    if args.lazy_load:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=4)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load Model
    if args.model == "unet":
        model = UNet(latent_dim, latent_dim).to(device)
        train_unet(model, train_dataloader, val_dataloader, save_dir, writer, device, args)
    elif args.model == "unet_dm" or args.model == "unet_dm_cond":
        model = None
        train_unet_dm(model, train_dataloader, val_dataloader, save_dir, writer, device, args)
    elif args.model == "unet_dm_gan":
        model = None
        train_unet_dm_gan(model, train_dataloader, val_dataloader, save_dir, writer, device, args)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument("--val_every_epochs", type=int, default=2)
    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--upscaling", action="store_true", default=False)
    parser.add_argument("--chi2-reg", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--restore_config", type=str, default=None)

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet_dm", "unet_dm_cond", "unet_dm_gan", "unet_dm_cond_gan"])
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    parser.add_argument("--lazy_load", action="store_true", default=True, help="if True, does not load the entire dataset into memory. Useful for large datasets.")

    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--start_from_epoch", type=int, default=0)

    args = parser.parse_args()
    main(args)