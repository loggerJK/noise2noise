import os, sys
from os import path
print(( path.dirname( path.abspath(__file__) ) ))
sys.path.append(( path.dirname( path.abspath(__file__) ) ))

import torch
from rectified_flow import RectifiedFlow, ImageDataset, Unet, Trainer
from utils import CustomDatasetV2

model = Unet(dim = 64, channels=4)

rectified_flow = RectifiedFlow(model)

##### Load Dataset
import glob
from pprint import pprint
folder_path = '/media/dataset1/project/jiwon/noise2noise_dataset/coco_latents/'
initial_folder_path = os.path.join(folder_path, 'initial')
inversion_folder_path = os.path.join(folder_path, 'inversion')

initial_latents = sorted(glob.glob(os.path.join(initial_folder_path , '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
inversion_latents = sorted(glob.glob(os.path.join(inversion_folder_path , '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
prompts = sorted(glob.glob(os.path.join(initial_folder_path , '*_prompt.txt')), key=lambda x: int(x.split('/')[-1].split('_')[0]))
x = initial_latents
y = inversion_latents

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

train_dataset = CustomDatasetV2(train_x, train_y, train_prompt)
val_dataset = CustomDatasetV2(val_x, val_y, val_prompt)


trainer = Trainer(
    rectified_flow,
    dataset = train_dataset,
    num_train_steps = 200_000,
    save_results_every=10_000,
    checkpoint_every=10_000,
    checkpoints_folder = './checkpoints/mapping2',
    batch_size=32,
    results_folder = './results_flow',   # samples will be saved periodically to this folder
    accelerate_kwargs = {'log_with' : 'wandb'}
)

##### Load pretrained
# checkpoint_path = '/media/dataset1/project/jiwon/noise2noise/latent_mlp/checkpoints/checkpoint.100000.pt'
# trainer.load(checkpoint_path)

trainer()