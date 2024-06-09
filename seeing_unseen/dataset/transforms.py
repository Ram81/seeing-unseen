from typing import Any, Dict, List, Optional, Sequence

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from seeing_unseen.core.base import BaseTransform
from seeing_unseen.core.registry import registry

try:
    from torchvision import datapoints
    from torchvision.transforms.v2 import Transform

    class GaussianNoise(Transform):
        """[BETA] Adds GaussianNoise image with randomly chosen Gaussian blur.

        .. v2betastatus:: GausssianBlur transform

        If the input is a Tensor, it is expected
        to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            kernel_size (int or sequence): Size of the Gaussian kernel.
            sigma (float or tuple of float (min, max)): Standard deviation to be used for
                creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
                of float (min, max), sigma is chosen uniformly at random to lie in the
                given range.
        """

        _v1_transform_cls = None

        def __init__(self, sigma: Sequence[float] = (0.1, 2.0)) -> None:
            super().__init__()
            self.sigma = sigma

        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            target_shape = [
                inp.shape
                for inp in flat_inputs
                if isinstance(inp, torch.Tensor) and len(inp.shape) == 4
            ][0]
            noise = []
            for i in range(target_shape[0]):
                sigma = (
                    torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
                )
                gaussian_noise = torch.normal(
                    mean=0, std=sigma, size=target_shape[1:]
                )
                noise.append(gaussian_noise)
            return dict(gaussian_noise=torch.stack(noise))

        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            if isinstance(inpt, datapoints.Mask) or len(inpt.shape) < 4:
                return inpt
            return inpt.float() + params["gaussian_noise"].to(inpt.device)

    class ClipValues(Transform):
        _v1_transform_cls = None

        def __init__(self) -> None:
            super().__init__()

        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            return dict()

        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            if isinstance(inpt, datapoints.Mask) or len(inpt.shape) < 4:
                return inpt
            return torch.clip(inpt.float(), min=0, max=255.0)

except:
    pass


@registry.register_transforms(name="segmentation_base")
class SegmentationBaseTransform(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from torchvision.transforms import v2 as v2_transforms

        resolution = [int(i) for i in kwargs["resized_resolution"]]
        self.transforms = v2_transforms.Compose(
            [
                v2_transforms.RandomHorizontalFlip(p=0.5),
                v2_transforms.RandomApply(
                    [v2_transforms.Resize(size=resolution, interpolation=3)],
                    p=kwargs["resize_prob"] > 0,
                ),
                v2_transforms.RandomApply(
                    [
                        v2_transforms.RandomResizedCrop(
                            size=512, scale=(0.2, 1.0), interpolation=3
                        )
                    ],
                    p=kwargs["random_resize_crop_prob"],
                ),
                v2_transforms.RandomApply(
                    [
                        v2_transforms.RandomAffine(
                            degrees=30,
                            translate=(0.0625, 0.0625),
                            scale=(0.9, 1.1),
                        )
                    ],
                    p=0.5,
                ),
                v2_transforms.RandomApply(
                    [v2_transforms.GaussianBlur(kernel_size=(7, 11))], p=0.6
                ),
                v2_transforms.RandomApply(
                    [GaussianNoise(sigma=(7.0, 13.0))], p=0.6
                ),
                ClipValues(),
                v2_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),
                v2_transforms.RandomApply(
                    [
                        v2_transforms.ColorJitter(
                            brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.3,
                        )
                    ]
                ),
            ]
        )

    def __call__(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> Any:
        if masks is None:
            x_aug = self.transforms(x)
            return x_aug

        x_aug, mask_augs = self.transforms(x, datapoints.Mask(masks))
        return x_aug, mask_augs.data


@registry.register_transforms(name="albumentations_base")
class AlbumentationsTransform(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, rotate_limit=30),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.5,
                            alpha=10,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(
                            p=0.5, distort_limit=0.15, shift_limit=0.07
                        ),
                    ],
                    p=0.5,
                ),
                A.GaussianBlur(p=0.4, blur_limit=(1, 5)),
                A.GaussNoise(p=0.7, var_limit=(10, 50)),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5
                ),
                ToTensorV2(),
            ]
        )

    def apply(self, x: np.ndarray):
        return self.transforms(image=x)["image"] / 255

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        if masks is None:
            x_aug = self.apply(x)
            return x_aug

        augmented = self.transforms(image=x, mask=masks)
        return augmented["image"] / 255, augmented["mask"]


@registry.register_transforms(name="albumentations_hard")
class AlbumentationsTransformHard(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, rotate_limit=30),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            p=0.7,
                            alpha=10,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                        ),
                        A.GridDistortion(p=0.7),
                        A.OpticalDistortion(
                            p=0.7, distort_limit=0.15, shift_limit=0.07
                        ),
                    ],
                    p=0.7,
                ),
                A.GaussianBlur(p=0.7, blur_limit=(7, 11)),
                A.GaussNoise(p=0.7, var_limit=(100, 300)),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5
                ),
                ToTensorV2(),
            ]
        )

    def apply(self, x: np.ndarray):
        return self.transforms(image=x)["image"] / 255

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        if masks is None:
            x_aug = self.apply(x)
            return x_aug

        augmented = self.transforms(image=x, mask=masks)
        return augmented["image"] / 255, augmented["mask"]


@registry.register_transforms(name="none")
class NoTransform(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()


@registry.register_transforms(name="eval_ablation")
class EvalTransform(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transforms = A.Compose(
            [
                ToTensorV2(),
            ]
        )

    def apply(self, x: np.ndarray):
        return self.transforms(image=x)["image"] / 255

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        if masks is None:
            augmented = self.transforms(image=x)
            return augmented["image"] / 255

        augmented = self.transforms(image=x, mask=masks)
        return augmented["image"] / 255, augmented["mask"]


@registry.register_transforms(name="mae")
class MAETransform(BaseTransform):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from torchvision.transforms import v2 as v2_transforms

        self.transforms = v2_transforms.Compose(
            [
                v2_transforms.RandomHorizontalFlip(p=0.5),
                # v2_transforms.RandomApply([v2_transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation=3)], p=1),
                v2_transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
                ),
                v2_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def apply(self, x: np.ndarray):
        return self.transforms(x)

    def __call__(
        self, x: np.ndarray, masks: Optional[np.ndarray] = None
    ) -> Any:
        from torchvision import datapoints

        if masks is None:
            return self.transforms(x)

        image, mask = self.transforms(x, datapoints.Mask(masks))
        return image, mask.data
