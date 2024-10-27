import os
import torch
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer, Reflow, ReflowTrainer
from utils import CustomDatasetV2
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

model = Unet(dim = 64, channels=4)
# print(model)

checkpoint_path = '/media/dataset1/project/jiwon/noise2noise/latent_mlp/checkpoints/checkpoint.100000.pt'

tmp = torch.load(checkpoint_path)['model']
# Change keys : model.blah.blah -> blah.blah
new_dict = {}
for k, v in tmp.items():
    new_dict[k.split('model.')[1]] = v
model.load_state_dict(new_dict)
model.to('cuda')

accelerator = Accelerator(log_with='wandb')
accelerator.init_trackers(
            project_name="noise2noise",
        )

rectified_flow = RectifiedFlow(model,
                               data_shape=(4, 128, 128),)
reflow = Reflow(rectified_flow)
optimizer = torch.optim.Adam(reflow.parameters(), lr=1e-4)

reflow, optimizer = accelerator.prepare(reflow, optimizer)

num_train_steps = 100_000

progress = tqdm(total=num_train_steps, disable=not accelerator.is_local_main_process)
reflow.train()
for step in range(num_train_steps):
    
    dummy = None
    reflow_loss = reflow(dummy)
    accelerator.backward(reflow_loss)
    optimizer.step()
    optimizer.zero_grad()
    
    progress.update(1)
    progress.set_postfix({'loss': reflow_loss.item()})
    
    


    