export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2

# python train.py --batch_size 8 --run_name unet_dm_cond --model unet_dm_cond --logger tensorboard --debug
accelerate launch  --main_process_port 29500 train.py --batch_size 8 --run_name unet_dm_cond3 --model unet_dm_cond --logger wandb --save_every_epochs 5
