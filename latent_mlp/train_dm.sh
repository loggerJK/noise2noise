export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

python train.py --batch_size 16 --run_name unet_dm_upscaling_reg --model unet_dm --upscaling --chi2-reg 0.01
# accelerate launch train.py --batch_size 4 --run_name unet_dm_upscaling_reg --model unet_dm --upscaling --chi2-reg 0.001 --debug
