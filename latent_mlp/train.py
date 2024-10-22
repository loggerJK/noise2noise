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
    from diffusers.pipelines.ddim.pipeline_ddim_condition import DDIMCondPipeline
    from diffusers.utils import make_image_grid
    import os

    from model import create_unet_dm, create_unet_dm_cond

    import torch.nn.functional as F

    if args.pretrained is None:
        # Initialize
        if args.model == "unet_dm":
            model, config = create_unet_dm(args) # Now, config contains args
        elif args.model == "unet_dm_cond":
            model, config = create_unet_dm_cond(args)
            tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer", variant="fp16")
            text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", variant="fp16")
        else:
            raise ValueError(f"Invalid model type: {args.model}")

        noise_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                        clip_sample=False,
                                        )

    else :
        # Restore from pretrained

        if args.model == "unet_dm":
            _, config = create_unet_dm(args) # Now, config contains args
            pipe = DDIMPipeline.from_pretrained(args.pretrained)
        elif args.model == "unet_dm_cond":
            _, config = create_unet_dm_cond(args)
            pipe = DDIMCondPipeline.from_pretrained(args.pretrained)
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder

        model = pipe.unet
        noise_scheduler = pipe.scheduler
        # if args.restore_config : config = pipe.config

    print(f"Model config: {config}")

    # Save config
    config_dict = config.asdict()
    import json
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # Requires grad off for text encoder
    if args.model == 'unet_dm_cond' : text_encoder.requires_grad_(False)

    # Initialize accelerator and tensorboard logging
    logger = args.logger
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=logger,
        project_dir=config.output_dir,
    )
    if logger == "tensorboard":
        accelerator.init_trackers(
                    project_name="./",
                )
    elif logger == "wandb":
        accelerator.init_trackers(
            project_name="noise2noise",
            config=config,
            init_kwargs={"name": args.run_name},
        )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if args.model == "unet_dm_cond":
        tokenizer, text_encoder = accelerator.prepare(tokenizer, text_encoder)
    device = accelerator.device

    global_step = 0

    # Now you train the model
    for epoch in range(config.start_from_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (x, y, prompts) in enumerate(train_dataloader):
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

            # Process the prompt
            if args.model == "unet_dm_cond":
                with torch.no_grad():
                    text_input = tokenizer(
                        prompts, padding="max_length", max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt"
                    )
                    text_embeddings = text_encoder(text_input.input_ids.to(y.device))[0]

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.model == "unet_dm":
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                elif args.model == "unet_dm_cond":
                    noise_pred = model(noisy_images, timesteps, text_embeddings, return_dict=False)[0]
                mse_loss = F.mse_loss(noise_pred, noise)

                # Regularization
                reg_loss = torch.tensor(0.0).to(noise_pred.device)
                if args.chi2_reg > 0:
                    pred_original_sample = []
                    for noise_pred_, t, noisy_image in zip(noise_pred, timesteps, noisy_images):
                        if t < 100:
                            pred_original_sample_ = noise_scheduler.step(noise_pred_, t, noisy_image).pred_original_sample
                            pred_original_sample.append(pred_original_sample_)
                    if len(pred_original_sample ) > 0:
                        pred_original_sample = torch.stack(pred_original_sample)
                        reg = chi2_neg_log_prob(pred_original_sample)
                    else:
                        reg = torch.tensor(0.0).to(noise_pred.device)
                    reg_loss = args.chi2_reg * reg
                loss = mse_loss + reg_loss


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,}
            logs = {"Loss/train": loss.detach().item(), "Loss/train/mse" : mse_loss.detach().item(),  "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch}
            if args.chi2_reg > 0:
                logs["Loss/chi2_reg"] = reg_loss.detach().item()
                logs["chi2"] = reg.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Validation loop
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if args.model == "unet_dm":
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        elif args.model == "unet_dm_cond":
            pipeline = DDIMCondPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, tokenizer=accelerator.unwrap_model(tokenizer), text_encoder=accelerator.unwrap_model(text_encoder))

        if (epoch % config.val_every_epochs == 0):
            pipeline.set_progress_bar_config(disable=True) # Disable the progress bar for this call

            # Validation loop
            with torch.no_grad():
                print(f"Generating images for epoch {epoch}, length : {len(val_dataloader)}")
                val_progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process)
                val_progress_bar.set_description(f"Val Epoch {epoch}")
                val_loss = 0.0
                total_len = 0
                mean_norm = 0.0
                mean_chi2 = 0.0

                custom_mean_norm = 0.0
                custom_mean_chi2 = 0.0
                for val_x, val_y, prompts in (val_dataloader):
                    val_x = torch.randn(val_x.shape, device=val_x.device)
                    if args.model == "unet_dm":
                        pred = pipeline(
                            batch_size=config.eval_batch_size,
                            latents = val_x,
                            num_inference_steps=config.num_inference_steps,
                        ).images
                    elif args.model == "unet_dm_cond":
                        pred = pipeline(
                            batch_size=config.eval_batch_size,
                            latents = val_x,
                            prompt = prompts,
                            num_inference_steps=config.num_inference_steps,
                        ).images
                        prompt_ = "A photo of a corgi riding a skateboard"
                        pred_ = pipeline(
                            batch_size=config.eval_batch_size,
                            latents = torch.randn(val_x.shape, device=val_x.device),
                            prompt = [prompt_] * val_x.shape[0],
                            num_inference_steps=config.num_inference_steps,
                        ).images

                    all_preds = accelerator.gather(pred)
                    all_val_y = accelerator.gather(val_y)
                    loss = F.mse_loss(all_preds, all_val_y)
                    # if config.chi2_reg > 0:
                    #     reg = chi2_neg_log_prob(pred)
                    #     loss += config.chi2_reg * reg
                    reg, norm = chi2_neg_log_prob(all_preds)
                    
                    val_loss += loss.detach().item()
                    mean_norm += norm.detach().item()
                    mean_chi2 += reg.detach().item()

                    # Custom prompt
                    if args.model == "unet_dm_cond":
                        all_preds_ = accelerator.gather(pred_)
                        custom_reg, custom_norm = chi2_neg_log_prob(all_preds_)
                        custom_mean_norm += custom_norm.detach().item()
                        custom_mean_chi2 += custom_reg.detach().item()

                    val_progress_bar.update(1)
                    postfix = {
                        "val_loss": loss.detach().item(), 
                        "val_norm": norm.detach().item(), 
                        "chi2" : reg.detach().item()}
                    if args.model == "unet_dm_cond":
                        postfix.update({
                        "val_custom_norm": custom_norm.detach().item(),
                        "val_custom_chi2" : custom_reg.detach().item()})
                    val_progress_bar.set_postfix(postfix)
                
                val_loss /= len(val_dataloader)
                mean_norm /= len(val_dataloader)
                mean_chi2 /= len(val_dataloader)
                custom_mean_norm /= len(val_dataloader)
                custom_mean_chi2 /= len(val_dataloader)

                logs = {"Loss/val": val_loss, "val_norm": mean_norm, "val_chi2": mean_chi2, "val_custom_norm": custom_mean_norm, "val_custom_chi2": custom_mean_chi2}
                # if args.chi2_reg > 0:
                #     logs["Loss/chi2_reg"] = reg.detach().item()
                accelerator.log(logs, step=global_step) # Log the validation loss every epoch

        if accelerator.is_main_process :
            pipeline.save_pretrained(config.output_dir)
            if (epoch % config.save_model_epochs == 0):
                pipeline.save_pretrained(os.path.join(config.output_dir, f"model_{epoch}epoch"))




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

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "unet_dm", "unet_dm_cond"])
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    parser.add_argument("--lazy_load", action="store_true", default=True, help="if True, does not load the entire dataset into memory. Useful for large datasets.")

    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--start_from_epoch", type=int, default=0)

    args = parser.parse_args()
    main(args)