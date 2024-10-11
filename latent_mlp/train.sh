export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

python train.py --batch_size 128 --run_name upscaling_reg_full --upscaling --chi2-reg 0.001