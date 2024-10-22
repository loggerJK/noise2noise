export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# python train.py --batch_size 8 --run_name unet_dm_cond --model unet_dm_cond --logger tensorboard --debug
accelerate launch --config-file ./default_config.yaml --main_process_port 29502 train.py --batch_size 16 --run_name unet_dm_TEST --model unet_dm --logger tensorboard --debug --save_every_epochs 5 --val_every_epochs 2
