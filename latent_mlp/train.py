import torch
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import residualMLP, UNet
import os
from tqdm import tqdm
from argparse import ArgumentParser

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
    return  -(reg.mean())

def train_unet(model, train_dataloader, val_dataloader, save_dir, writer, device, args):
    # train dataset
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for epoch in tqdm(range(20)):
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
            if (step ) % args.validation_every == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_x, val_y in tqdm(val_dataloader):
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)
                        val_y_pred = model(val_x)
                        val_loss += criterion(val_y_pred, val_y).item()

                val_loss /= len(val_dataloader)
                writer.add_scalar('Loss/val', val_loss, step)
                print(f"epoch: {epoch}, val_loss: {val_loss}")
                model.train()


        # Save Model
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch}epoch.pt"))

def train_unet_dm(model, train_dataloader, val_dataloader, save_dir, writer, device, args):
    from accelerate import Accelerator
    from huggingface_hub import create_repo, upload_folder
    from tqdm.auto import tqdm
    from pathlib import Path
    import os

    import torch
    from PIL import Image
    from diffusers import DDPMScheduler, DDIMScheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    from diffusers import DDPMPipeline, DDIMPipeline
    from diffusers.utils import make_image_grid
    import os

    from model import create_unet_dm

    import torch.nn.functional as F

    model, config = create_unet_dm(args) # Now, config contains args

    print(f"Model config: {config}")

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                    clip_sample=False,
                                    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.output_dir,
    )
    accelerator.init_trackers(
                project_name="./",
            )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (x, y) in enumerate(train_dataloader):
            if config.upscaling:
                y = y *  (torch.norm(x, dim=[1,2,3], p=2) / torch.norm(y, dim=[1,2,3], p=2)).reshape(-1, 1, 1, 1)

            clean_images = y
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                if config.chi2_reg > 0:
                    reg = chi2_neg_log_prob(noise_pred)
                    loss += config.chi2_reg * reg

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,}
            logs = {"Loss/train": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.chi2_reg > 0:
                logs["Loss/chi2_reg"] = reg.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Validation loop
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if (epoch % config.save_image_epochs == 0):
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline.set_progress_bar_config(disable=True) # Disable the progress bar for this call

            with torch.no_grad():
                print(f"Generating images for epoch {epoch}, length : {len(val_dataloader)}")
                val_progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
                val_progress_bar.set_description(f"Val Epoch {epoch}")
                for val_x, val_y in (val_dataloader):
                    pred = pipeline(
                        batch_size=config.eval_batch_size,
                        latents = val_x,
                    ).images

                    all_preds = accelerator.gather(pred)
                    all_val_y = accelerator.gather(val_y)
                    loss = F.mse_loss(all_preds, all_val_y)
                    if config.chi2_reg > 0:
                        reg = chi2_neg_log_prob(pred)
                        loss += config.chi2_reg * reg
                    logs = {"Loss/val": loss.detach().item()}
                    if args.chi2_reg > 0:
                        logs["Loss/chi2_reg"] = reg.detach().item()
                    accelerator.log(logs, step=global_step)
                    val_progress_bar.update(1)

        if accelerator.is_main_process :
            if (epoch % config.save_model_epochs == 0):
                pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(config.output_dir)




def main(args):
    # Config
    save_dir = './results/' + args.run_name
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.model == "unet":
        writer = SummaryWriter(save_dir)
    else :
        writer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # Load Dataset
    import glob
    from pprint import pprint
    folder_path = '/media/dataset1/donghoon/noise2noise/corgi-riding-skateboard-fix-prompt-error-while-inversion/sdxl/'
    # 0_initial_latent.pt
    initial_latents = sorted(glob.glob(folder_path + '*_initial_latent.pt'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
    # 0_latent.pt
    inversion_latents = sorted(glob.glob(folder_path + '*[0-9]_inversion_latent.pt'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
    # 0_recon.png

    if args.debug:
        initial_latents = initial_latents[:100]
        inversion_latents = inversion_latents[:100]

    initial_latents_loaded = []
    inversion_latents_loaded = []

    ## inversion latents 중에서, 대응하는 inital latent가 있는지 확인하고, 있는 경우에만 로드한다
    for i, latent in tqdm(enumerate(inversion_latents)):
        initial_latent = latent.replace('_inversion_latent.pt', '_initial_latent.pt')
        if initial_latent in initial_latents:
            initial_latents_loaded.append(torch.load(initial_latent).cpu().numpy())
            inversion_latents_loaded.append(torch.load(latent).cpu().numpy())

    initial_latents = torch.Tensor(np.array(initial_latents_loaded)).squeeze(1) # (N, C, H, W)
    inversion_latents = torch.Tensor(np.array(inversion_latents_loaded)).squeeze(1) # (N, C, H, W)

    x = initial_latents
    y = inversion_latents

    latent_dim = x.shape[1]
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")

    # train dataset
    percent = 0.99
    train_x = x[:int(len(x) * percent)]
    train_y = y[:int(len(y) * percent)]
    val_x = x[int(len(x) * percent):]
    val_y = y[int(len(y) * percent):]
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Model
    if args.model == "unet":
        model = UNet(latent_dim, latent_dim).to(device)
        train_unet(model, train_dataloader, val_dataloader, save_dir, writer, device, args)
    elif args.model == "unet_dm":
        model = UNet(latent_dim, latent_dim).to(device)
        train_unet_dm(model, train_dataloader, val_dataloader, save_dir, writer, device, args)






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--validation_every", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--upscaling", action="store_true", default=False)
    parser.add_argument("--chi2-reg", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet_dm"])
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    main(args)