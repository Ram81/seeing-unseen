from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.models.base import SPModel
from seeing_unseen.models.encoders.clip_encoder import ResNetCLIPEncoder
from seeing_unseen.models.encoders.fusion import FusionConvLat, FusionMult
from seeing_unseen.models.encoders.resnet import ConvBlock, IdentityBlock
from seeing_unseen.models.encoders.unet import Up


@registry.register_affordance_model(name="clip_unet_img_query")
class CLIPUNetImgQuery(SPModel):
    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.target_input_shape = target_input_shape
        self.output_dim = output_dim
        self.upsample_factor = upsample_factor
        self.bilinear = bilinear
        self.batchnorm = batchnorm

        self.init_clip()
        self.init_target_encoder()
        self.init_decoder()
        self.train()

        self.activation = nn.Sigmoid()

    def init_clip(self):
        self.clip = ResNetCLIPEncoder(
            input_shape=self.input_shape,
            backbone_type="prepool",
            clip_model="RN50",
        )
        self.clip_out_dim = self.clip.output_shape[0]

    def init_target_encoder(self):
        self.target_encoder = ResNetCLIPEncoder(
            input_shape=self.target_input_shape,
            backbone_type="none",
            clip_model="RN50",
        )
        self.target_encoder_out_dim = self.target_encoder.output_shape[0]

    def init_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.clip_out_dim, 1024, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

        self.up1 = Up(
            self.clip_out_dim, 1024 // self.upsample_factor, self.bilinear
        )
        self.up2 = Up(1024, 512 // self.upsample_factor, self.bilinear)

        self.lin_proj = nn.Linear(self.target_encoder_out_dim, 256)
        self.target_fuser = FusionConvLat(input_dim=256 + 256, output_dim=256)
        self.up3 = Up(512, 256 // self.upsample_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(
                128,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                64,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                64,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                32,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                32,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                16,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward(self, **kwargs) -> torch.Tensor:
        # receptacle = receptacle.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
        # target = target.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
        batch = kwargs["batch"]
        target = batch["target_query"]
        receptacle = batch["image"]

        input_shape = receptacle.shape
        x, x_im_feats = self.clip(receptacle)

        target_embedding, _ = self.target_encoder(
            target, apply_resize_tfms=False
        )

        x = self.conv1(x)
        x = self.up1(x, x_im_feats[-2])

        x = self.up2(x, x_im_feats[-3])

        x = self.target_fuser(x, target_embedding, x2_proj=self.lin_proj)
        x = self.up3(x, x_im_feats[-4])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = F.interpolate(
            x, size=(input_shape[-2], input_shape[-1]), mode="bilinear"
        )
        return x


@registry.register_affordance_model(name="clip_unet")
class CLIPUNet(CLIPUNetImgQuery):
    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__(
            input_shape,
            target_input_shape,
            output_dim,
            upsample_factor,
            bilinear,
            batchnorm,
        )

    def init_target_encoder(self):
        self.target_encoder = None
        self.target_encoder_out_dim = 1024

    def init_discriminator(self):
        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 13 * 18, 1024),
            nn.ReLU(),
        )
        self.discriminator_fc = nn.Linear(1024, 1)
        self.discriminator_out_dim = 1024

    def init_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.clip_out_dim, 1024, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

        self.lang_fuser1 = FusionMult(input_dim=self.clip_out_dim // 2)
        self.lang_fuser2 = FusionMult(input_dim=self.clip_out_dim // 4)
        self.lang_fuser3 = FusionMult(input_dim=self.clip_out_dim // 8)

        self.lang_proj1 = nn.Linear(self.target_encoder_out_dim, 1024)
        self.lang_proj2 = nn.Linear(self.target_encoder_out_dim, 512)
        self.lang_proj3 = nn.Linear(self.target_encoder_out_dim, 256)

        self.up1 = Up(
            self.clip_out_dim, 1024 // self.upsample_factor, self.bilinear
        )
        self.up2 = Up(1024, 512 // self.upsample_factor, self.bilinear)
        self.up3 = Up(512, 256 // self.upsample_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(
                128,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                64,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                64,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                32,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                32,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                16,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward_encoder(self, image, query):
        x, x_im_feats = self.clip(image)

        x = self.conv1(x)

        x = self.lang_fuser1(x, query, x2_proj=self.lang_proj1)
        x = self.up1(x, x_im_feats[-2])

        x = self.lang_fuser2(x, query, x2_proj=self.lang_proj2)
        x = self.up2(x, x_im_feats[-3])

        x = self.lang_fuser3(x, query, x2_proj=self.lang_proj3)
        x = self.up3(x, x_im_feats[-4])
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        disc_out = None
        output = {}
        batch = kwargs["batch"]
        discriminator_only = kwargs.get("discriminator_only", False)
        target = batch["target_query"]
        receptacle = batch["image"]

        input_shape = receptacle.shape

        x = self.forward_encoder(receptacle, target)

        if self.add_discriminator:
            disc = self.discriminator(x)
            disc_out = self.discriminator_fc(disc)
            output["disc_out"] = disc_out

        if not discriminator_only:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.conv2(x)
            x = F.interpolate(
                x, size=(input_shape[-2], input_shape[-1]), mode="bilinear"
            )
            output["affordance"] = x
        return output
