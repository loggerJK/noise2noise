export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE=1

accelerate launch --config-file ./default_config.yaml --main_process_port 29502 rectified_flow.py