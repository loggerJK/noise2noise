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
from utils import my_collate_fn, CustomDataset
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from dataclasses import dataclass, asdict, field
from torch.utils.data import Subset
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))
from stylegan2.training.networks import Discriminator



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

def train_unet_dm_gan(model, train_dataloader, val_dataloader, save_dir, writer, device, args):
    if args.pretrained is None:
        # Initialize
        if args.model == "unet_dm_gan":
            model, config = create_unet_dm(args) # Now, config contains args
        elif args.model == "unet_dm_cond_gan":
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
        if args.model == "unet_dm_gan":
            _, config = create_unet_dm(args) # Now, config contains args
            pipe = DDIMPipeline.from_pretrained(args.pretrained)
        elif args.model == "unet_dm_cond_gan":
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

    # Initialize discriminator
    disc = Discriminator(c_dim=0, 
                     img_resolution=128,
                     img_channels=4,)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.add_param_group({"params": disc.parameters(), "lr": config.learning_rate})
    
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
    model, disc, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, disc, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if args.model == "unet_dm_cond_gan":
        tokenizer, text_encoder = accelerator.prepare(tokenizer, text_encoder)
    device = accelerator.device

    global_step = 0

    # Now you train the model
    for epoch in range(config.start_from_epoch, config.num_epochs):
        # Initialize the pipeline for each epoch - GAN EMA
        if args.model == "unet_dm_gan":
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        elif args.model == "unet_dm_cond_gan":
            pipeline = DDIMCondPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, tokenizer=accelerator.unwrap_model(tokenizer), text_encoder=accelerator.unwrap_model(text_encoder))
        pipeline.set_progress_bar_config(disable=True)
            
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        mode = "gen"
        for step, (x, y, prompts) in enumerate(train_dataloader):
            
            # 100 Step 단위로 모드를 바꿈
            if step % 100 == 0:
                if mode == "gen":
                    mode = "disc"
                    print("Switching to disc mode")
                    model.requires_grad_(False)
                    disc.requires_grad_(True)
                    # Update the pipeline for each epoch - GAN EMA
                    if args.model == "unet_dm_gan":
                        pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                    elif args.model == "unet_dm_cond_gan":
                        pipeline = DDIMCondPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, tokenizer=accelerator.unwrap_model(tokenizer), text_encoder=accelerator.unwrap_model(text_encoder))
                    pipeline.set_progress_bar_config(disable=True)
                else:
                    mode = "gen"
                    print("Switching to gen mode")
                    model.requires_grad_(True)
                    disc.requires_grad_(False)
            
            
            if config.upscaling:
                y = y *  (torch.norm(x, dim=[1,2,3], p=2) / torch.norm(y, dim=[1,2,3], p=2)).reshape(-1, 1, 1, 1)

            real = clean_images = y
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            # timesteps_T = torch.tensor(noise_scheduler.config.num_train_timesteps-1, device=clean_images.device).repeat(bs)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            # noisy_images_T = noise_scheduler.add_noise(clean_images, noise, timesteps_T)

            # Process the prompt
            if args.model == "unet_dm_cond_gan":
                with torch.no_grad():
                    text_input = tokenizer(
                        prompts, padding="max_length", max_length = tokenizer.model_max_length, truncation=True, return_tensors="pt"
                    )
                    text_embeddings = text_encoder(text_input.input_ids.to(y.device))[0]

            with accelerator.accumulate(model):

                # GAN Loss
                latents = torch.randn(noisy_images.shape, device=noisy_images.device)
                fake = pipeline(
                    batch_size=latents.shape[0],
                    latents = latents,
                    num_inference_steps=config.num_inference_steps,
                ).images

                if mode == "disc":
                    real_logits = disc(real, None)
                    fake_logits = disc(fake.detach(), None)
                    disc_loss = F.softplus(real_logits).mean() + F.softplus(-fake_logits).mean()
                    loss = disc_loss
                
                elif mode == "gen":
                    # MSE Loss
                    if args.model == "unet_dm_gan":
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    elif args.model == "unet_dm_cond_gan":
                        noise_pred = model(noisy_images, timesteps, text_embeddings, return_dict=False)[0]
                    mse_loss = F.mse_loss(noise_pred, noise)
                    
                    fake_logits = disc(fake, None)
                    # Maximize the probability of the fake images
                    gen_loss = F.softplus(fake_logits).mean()
                    loss = mse_loss + gen_loss
                    

                # Regularization
                reg_loss = torch.tensor(0.0).to(device)
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
                        reg = torch.tensor(0.0).to(device)
                    reg_loss = args.chi2_reg * reg
                
                loss = loss + reg_loss


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,}
            logs = {"Loss/train": loss.detach().item(),  "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch}
            
            if mode == "disc":
                logs["Loss/train/disc"] = disc_loss.detach().item()
            else:
                logs["Loss/train/gen"] = gen_loss.detach().item()
                logs["Loss/train/mse"] = mse_loss.detach().item()
            
            if args.chi2_reg > 0:
                logs["Loss/chi2_reg"] = reg_loss.detach().item()
                logs["chi2"] = reg.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Validation loop
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if args.model == "unet_dm_gan":
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        elif args.model == "unet_dm_cond_gan":
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
                    if len(val_x.shape) == 3: val_x = val_x.unsqueeze(0)
                    if args.model == "unet_dm_gan":
                        pred = pipeline(
                            batch_size=config.eval_batch_size,
                            latents = val_x,
                            num_inference_steps=config.num_inference_steps,
                        ).images
                    elif args.model == "unet_dm_cond_gan":
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
                    reg, norm = chi2_neg_log_prob(all_preds)
                    
                    val_loss += loss.detach().item()
                    mean_norm += norm.detach().item()
                    mean_chi2 += reg.detach().item()

                    # Custom prompt
                    if args.model == "unet_dm_cond_gan":
                        all_preds_ = accelerator.gather(pred_)
                        custom_reg, custom_norm = chi2_neg_log_prob(all_preds_)
                        custom_mean_norm += custom_norm.detach().item()
                        custom_mean_chi2 += custom_reg.detach().item()

                    val_progress_bar.update(1)
                    postfix = {
                        "val_loss": loss.detach().item(), 
                        "val_norm": norm.detach().item(), 
                        "chi2" : reg.detach().item()}
                    if args.model == "unet_dm_cond_gan":
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


