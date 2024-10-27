import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from rectified_flow_pytorch import RectifiedFlow, ImageDataset, Unet, Trainer
from utils import CustomDatasetV2
import numpy as np
from tqdm import tqdm

model = Unet(dim = 64, channels=4)
# print(model)

checkpoint_path = '/media/dataset1/project/jiwon/noise2noise/latent_mlp/checkpoints/checkpoint.90000.pt'
tmp = torch.load(checkpoint_path)['model']
# Change keys : model.blah.blah -> blah.blah
new_dict = {}
for k, v in tmp.items():
    new_dict[k.split('model.')[1]] = v
model.load_state_dict(new_dict)
model.to('cuda')

rectified_flow = RectifiedFlow(model)

save_path = './sample/flow/'
os.makedirs(save_path, exist_ok=True)

batch_size = 1
shape = (4, 128, 128)
for i in tqdm(range(30)):
    initial_latent = torch.randn(1,4,128,128).to('cuda')
    pred = rectified_flow.sample(batch_size=batch_size, data_shape=shape, initial_latent=initial_latent)
    pred = pred.detach().cpu().numpy()
    np.save(
        os.path.join(save_path, f'{i}_noise.npy'),
        initial_latent.detach().cpu().numpy()
    )
    print(f"pred shape: {pred.shape}")
    np.save(
        os.path.join(save_path, f'{i}_pred.npy'),
        pred
    )
    
    