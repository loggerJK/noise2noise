export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python train.py --batch_size 128 --run_name unet_upscaling_reg_full --upscaling --chi2-reg 0.01 --save_every_epochs 5 --num_epochs 500