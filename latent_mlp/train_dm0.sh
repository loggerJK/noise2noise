export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# python train.py --batch_size 8 --run_name unet_dm_test --model unet_dm
# accelerate launch --main_process_port 0 train.py --batch_size 16 --run_name unet_dm --model unet_dm
accelerate launch  --main_process_port 29500 train.py --batch_size 16 --run_name unet_dm_upscaling_reg --model unet_dm --upscaling --chi2-reg 0.01 

