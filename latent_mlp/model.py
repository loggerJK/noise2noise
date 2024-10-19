import torch.nn as nn

from unet_parts import DoubleConv, Down, Up, OutConv
from dataclasses import dataclass, asdict, field

class residualMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(residualMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ),

        )

    def forward(self, x):
        res = x.clone()
        return res + self.model(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128,))
        self.up4 = (Up(128, 64,))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        res = x.clone()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}, x3.shape: {x3.shape}, x4.shape: {x4.shape}")
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#################### U-Net DM ###########################
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class TrainingConfig():
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results/DDPM"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    def asdict(self):
        return asdict(self)


def create_unet_dm(args):

    config = TrainingConfig()
    for key, value in vars(args).items():
        setattr(config, key, value)
    setattr(config, "output_dir", args.save_dir)
    setattr(config, "train_batch_size", args.batch_size)
    setattr(config, "eval_batch_size", args.batch_size)
    setattr(config, "save_model_epochs", args.save_every_epochs)
    setattr(config, "save_image_epochs", args.save_every_epochs)

    from diffusers import UNet2DModel

    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=4,  # the number of input channels, 3 for RGB images
        out_channels=4,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model, config

def create_unet_dm_cond(args):

    config = TrainingConfig()
    for key, value in vars(args).items():
        setattr(config, key, value)
    setattr(config, "output_dir", args.save_dir)
    setattr(config, "train_batch_size", args.batch_size)
    setattr(config, "eval_batch_size", args.batch_size)
    setattr(config, "save_model_epochs", args.save_every_epochs)
    setattr(config, "save_image_epochs", args.save_every_epochs)
    setattr(config, "learning_rate", args.lr)

    from diffusers import UNet2DConditionModel

    model = UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=4,  # the number of input channels, 3 for RGB images
        out_channels=4,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=768, # CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    )

    return model, config