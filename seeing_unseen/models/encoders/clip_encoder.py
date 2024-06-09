from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from torch import nn as nn
from torchvision import transforms as T


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple,
        backbone_type="prepool",
        clip_model="RN50",
    ):
        super().__init__()

        self.backbone_type = backbone_type
        model, preprocess = clip.load(clip_model)

        resize_transforms = []

        # expected input: H x W x C (np.uint8 in [0-255])
        if input_shape[0] != 224 or input_shape[1] != 224:
            print("Using CLIP preprocess for resizing+cropping to 224x224")
            resize_transforms = [
                # resize and center crop to 224
                preprocess.transforms[0],
                preprocess.transforms[1],
            ]

        self.resize_transforms = T.Compose(resize_transforms)
        preprocess_transforms = [
            # already tensor, but want float
            T.ConvertImageDtype(torch.float32),
            # normalize with CLIP mean, std
            preprocess.transforms[4],
        ]
        self.preprocess = T.Compose(preprocess_transforms)
        # expected output: H x W x C (np.float32)

        self.backbone = model.visual

        if "none" in backbone_type:
            self.backbone.attnpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            self.output_shape = (2048, 1)
        elif self.clip_prepool:
            self.output_shape = (2048, 7, 7)

            # Overwrite forward method to return both attnpool and avgpool
            # concatenated together (attnpool + avgpool).
            bound_method = forward_prepool.__get__(
                self.backbone, self.backbone.__class__
            )
            setattr(self.backbone, "forward", bound_method)

        for param in self.backbone.parameters():
            param.requires_grad = False
        # for module in self.backbone.modules():
        #     if "BatchNorm" in type(module).__name__:
        #         module.momentum = 0.0
        self.backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    def forward(self, batch: torch.Tensor, apply_resize_tfms: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if apply_resize_tfms:
            batch = torch.stack(
                [self.resize_transforms(img) for img in batch]
            )
        # print("Batch shape: {}".format(batch.shape))
        # batch = torch.stack(
        #     [self.preprocess(img) for img in batch]
        # )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
        batch = self.preprocess(batch)

        if not self.clip_prepool:
            batch = self.backbone(batch)
            batch = batch.to(torch.float32)
            return batch, []

        batch, batch_im_feats = self.backbone(batch)
        batch = batch.to(torch.float32)
        # print("Batch out shape: {} - {}".format(batch.shape, batch_im_feats[0].shape))
        return batch, batch_im_feats

    @property
    def clip_prepool(self):
        return "prepool" in self.backbone_type



def forward_prepool(self, x):
    """
    Adapted from https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138
    Expects a batch of images where the batch number is even. The whole batch
    is passed through all layers except the last layer; the first half of the
    batch will be passed through avgpool and the second half will be passed
    through attnpool. The outputs of both pools are concatenated returned.
    """

    im_feats = []
    def stem(x):
        for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
            x = relu(bn(conv(x)))
            im_feats.append(x)
        x = self.avgpool(x)
        im_feats.append(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)

    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        x = layer(x)
        im_feats.append(x)
    return x, im_feats


if __name__ == "__main__":
    clip_model = ResNetCLIPEncoder((224, 224, 3))
    clip_model = clip_model.cuda()

    batch = torch.rand((2, 224, 224, 3)).cuda()
    o_batch, o_batch_im_feats = clip_model(batch)
    print("Output: {} - {}".format(o_batch.shape, o_batch_im_feats[0].shape))

