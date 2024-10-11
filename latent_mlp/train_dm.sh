export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

# python train.py --batch_size 8 --run_name unet_dm_test --model unet_dm --debug
accelerate launch train.py --batch_size 16 --run_name unet_dm --model unet_dm
