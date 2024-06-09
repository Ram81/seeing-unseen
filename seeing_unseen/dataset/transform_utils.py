from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import ttach as tta
from ttach.base import Compose, ImageOnlyTransform, Merger
from ttach.transforms import HorizontalFlip, Resize

from seeing_unseen.core.base import BaseTransform
from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry


class GaussianBlur(ImageOnlyTransform):
    """Adds GaussianBlur to image"""

    identity_param = False

    def __init__(self, kernel_size: List[int]):
        from torchvision.transforms import v2 as v2_transforms

        self.transform = v2_transforms.Compose(
            [
                v2_transforms.GaussianBlur(kernel_size=kernel_size),
                v2_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),
            ]
        )
        super().__init__("kernel_size", kernel_size)

    def apply_aug_image(self, image, value=0, **kwargs):
        return self.transform(image)


class GaussNoise(ImageOnlyTransform):
    """Adds GaussianNoise to image"""

    identity_param = False

    def __init__(self, sigma: List[float]):
        from torchvision.transforms import v2 as v2_transforms

        from seeing_unseen.dataset.transforms import GaussianNoise

        self.transform = v2_transforms.Compose(
            [
                GaussianNoise(sigma=sigma),
                v2_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),
            ]
        )
        super().__init__("sigma", sigma)

    def apply_aug_image(self, image, value=0, **kwargs):
        return self.transform(image)


class ColorJitter(ImageOnlyTransform):
    """Adds GaussianNoise to image"""

    identity_param = False

    def __init__(
        self, brightness: float, contrast: float, saturation: float, hue: float
    ):
        from torchvision.transforms import v2 as v2_transforms

        self.transform = v2_transforms.Compose(
            [
                v2_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),
                v2_transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
            ]
        )
        super().__init__("brightness", [brightness, contrast, saturation, hue])

    def apply_aug_image(self, image, value=0, **kwargs):
        return self.transform(image)


class HorizontalFlipCustom(HorizontalFlip):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def apply_aug_image(self, image, apply=False, **kwargs):
        image_aug = super().apply_aug_image(image, apply=True) / 255.0
        return image_aug

    def augment_image(self, image):
        return self.apply_aug_image(image)


@registry.register_transforms(name="segmentation_tta")
class AffordanceTTA(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transforms = [
            HorizontalFlipCustom(),
            GaussianBlur(kernel_size=(7, 11)),
            GaussNoise(sigma=(7.0, 13.0)),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        ]
        # if kwargs["random_resize_crop_prob"] > 0:
        #     resolution = kwargs["resized_resolution"]
        #     original_size = kwargs["original_size"]
        #     self.transforms = [
        #         Resize(sizes=resolution, original_size=original_size[1:])
        #     ] + self.transforms
        #     logger.info(
        #         "Adding Resize resoluton: {} - {}".format(
        #             resolution, original_size
        #         )
        #     )
        logger.info("TTA transforms: {}".format(self.transforms))
        self.aug = GaussianBlur(kernel_size=(7, 11))

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        return self.aug.apply_aug_image(x), masks


@registry.register_transforms(name="resize_only_tta")
class ResizeOnlyTTA(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        resolution = kwargs["resized_resolution"]
        original_size = kwargs["original_size"]

        logger.info(
            "Resize resoluton: {} - {}".format(resolution, original_size)
        )

        self.transforms = [
            Resize(
                sizes=resolution,
                original_size=original_size[1:],
                interpolation="bilinear",
            )
        ]
        self.aug = GaussianBlur(kernel_size=(7, 11))

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        return self.aug.apply_aug_image(x), masks


class TTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> Union[str, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            size = 480
            if isinstance(transformer, Resize):
                size = transformer.params[1]

            augmented_image = transformer.apply_aug_image(
                batch["image"].clone(), size=size
            )

            new_batch = {
                "image": augmented_image,
                "target_query": batch["target_query"],
            }
            augmented_output = self.model(batch=new_batch, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.apply_deaug_mask(
                augmented_output, apply=True, size=size
            )
            merger.append(deaugmented_output)

        result = merger.result
        # print("Merger result: {} - {}".format(result.min(), result.max()))
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
