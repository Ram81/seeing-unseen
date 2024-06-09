import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from skimage import measure

from seeing_unseen.utils import depth_utils as du

SMOOTH = 1e-6


def iou(preds: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    intersection = (
        (preds * labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (
        torch.where((preds + labels) > 0, 1, 0).float().sum((1, 2))
    )  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0

    iou_lambda = (
        lambda x: torch.clamp(20 * (iou - x), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds
    ious = {
        "iou@0.5": iou_lambda(0.5).mean(),
        "iou@0.75": iou_lambda(0.75).mean(),
    }
    return ious


class SemanticPlaceMetrics:
    def __init__(
        self,
        img_size: List[int],
        hfov: int,
        return_imgs: bool = False,
        min_area: float = 0.005,
        iot_threshold: float = 0.5,
    ) -> None:
        self.img_size = img_size
        self.hfov = hfov
        self.camera_matrix = du.get_camera_matrix(
            img_size[0], img_size[1], hfov
        )
        self.eps = 1e-6
        self.num_batches = 0
        self.min_area = min_area
        self.iot_threshold = iot_threshold

        self.return_imgs = return_imgs

    def get_surface_grounded_affordance(
        self, observations: Dict, affordance_key: str = "affordance"
    ):
        surface_normal = du.depth_to_surface_normals_np(
            observations["depth"], camera_matrix=self.camera_matrix
        )
        upward_facing_surface_mask = du.upward_facing_surface_mask(
            surface_normal
        )

        # Get the affordance grounded to the surface
        affordance = (observations[affordance_key] > 0).astype(np.uint8)
        if len(affordance.shape) == 2:
            upward_facing_surface_mask = np.squeeze(
                upward_facing_surface_mask, axis=-1
            )

        grounded_affordance = upward_facing_surface_mask * affordance
        return grounded_affordance

    def affordance_on_receptacle_surface_accuracy(self, observations: Dict):
        """
        Compute the accuracy of the predicted affordance grounded to the receptacle surface.
        """
        # Get the surface grounded affordance
        grounded_affordance = self.get_surface_grounded_affordance(
            observations,
            affordance_key="affordance",
        )

        # Get the mask of the receptacle
        grounded_receptacle_mask = self.get_surface_grounded_affordance(
            observations,
            affordance_key="receptacle_mask",
        )

        if np.sum(observations["mask"]) == 0:
            return (
                int(np.sum(observations["affordance"]) == 0),
                grounded_affordance,
                grounded_receptacle_mask,
            )

        # Compute the accuracy
        accuracy = np.sum(grounded_affordance * grounded_receptacle_mask) / (
            np.sum(grounded_affordance) + self.eps
        )
        return accuracy, grounded_affordance, grounded_receptacle_mask

    def affordance_on_receptacle_accuracy(self, observations: Dict):
        """
        Compute the accuracy of the predicted affordance grounded to the surface.
        """
        if np.sum(observations["mask"]) == 0:
            return int(np.sum(observations["affordance"]) == 0)

        # Compute the accuracy
        accuracy = np.sum(
            observations["affordance"] * (observations["receptacle_mask"] > 0)
        ) / (np.sum(observations["affordance"]) + self.eps)
        return accuracy

    def affordance_on_surface_accuracy(self, observations: Dict):
        """
        Compute the accuracy of the predicted affordance grounded to the surface.
        """
        # Get the surface grounded affordance
        grounded_affordance = self.get_surface_grounded_affordance(
            observations,
            affordance_key="affordance",
        )

        if np.sum(observations["mask"]) == 0:
            return int(np.sum(observations["affordance"]) == 0)

        # Compute the accuracy
        accuracy = np.sum(grounded_affordance) / (
            np.sum(observations["affordance"]) + self.eps
        )
        return accuracy

    def cluster_binary_map(self, binary_map):
        if not isinstance(binary_map, np.ndarray):
            binary_map = binary_map.cpu().numpy()
        labels, num_labels = measure.label(
            binary_map, connectivity=2, return_num=True
        )  # Perform connected component analysis
        return labels.astype(np.int32), num_labels

    def semantic_precision_recall(
        self,
        affordance: np.ndarray,
        target: np.ndarray,
        mode: str = "receptacle",
    ):
        """
        Compute the precision of the predicted affordance on receptacles.
        """
        affordance_components, num_affordance_labels = self.cluster_binary_map(
            affordance
        )

        num_target_labels = len(np.unique(target)) - 1

        metrics = {
            "precision": 0,
            "recall": 0,
            "mean_iot": 0,
        }

        if num_target_labels == 0:
            metrics["recall"] = 1
            metrics["mean_iot"] = (num_affordance_labels == 0) * 1
            metrics["precision"] = (num_affordance_labels == 0) * 1
            return metrics

        if (num_affordance_labels - 1) == 0:
            metrics["precision"] = num_target_labels == 0
            metrics["mean_iot"] = (num_target_labels == 0) * 1
            metrics["recall"] = (num_target_labels == 0) * 1
            return metrics

        # Compute the accuracy
        tp = 0
        fp = 0
        targets_covered = []
        fail_iot = []
        mean_iot = 0
        count = 0
        for label in range(1, num_affordance_labels + 1):
            target_mask = target * (affordance_components == label)

            if (
                np.sum(target_mask > 0) / np.prod(target_mask.shape)
                < self.min_area
            ):
                continue

            r_covered = np.unique(target_mask).tolist()
            idx_s = 0
            if r_covered[0] == 0:
                idx_s = 1
            for receptacle_id in r_covered[idx_s:]:
                iot = np.sum((target_mask == receptacle_id) > 0) / np.sum(
                    affordance_components == label
                )
                mean_iot += iot

                if iot > self.iot_threshold:
                    targets_covered.append(receptacle_id)
                else:
                    fp += 1
                    fail_iot.append(iot)
            count += 1

        tp = np.unique(targets_covered).shape[0]
        fn = num_target_labels - np.unique(targets_covered).shape[0]
        if count > 0:
            mean_iot = mean_iot / count

        if mode == "target" and (tp / (tp + fn + self.eps)) > 0:
            print(
                np.unique(target),
                len(np.unique(affordance_components)),
                num_target_labels,
                np.unique(target),
                tp,
                fn,
                fp,
                np.unique(targets_covered),
                fail_iot,
                tp / (tp + fn + self.eps),
            )

        if tp > (tp + fn + self.eps):
            print(
                "invalid recall values: {} - {} - {} -{}".format(
                    tp, fn, self.eps, mode
                )
            )

        assert tp <= (tp + fn), "Recall denominator is wrong"

        metrics["precision"] = len(set(targets_covered)) / (
            len(set(targets_covered)) + fp + self.eps
        )
        metrics["recall"] = tp / (tp + fn + self.eps)
        metrics["mean_iot"] = mean_iot
        return metrics

    def semantic_classification_metrics(self, observations: Dict):
        affordance_receptacle_metrics = self.semantic_precision_recall(
            observations["affordance"], observations["receptacle_mask"]
        )

        grounded_affordance = self.get_surface_grounded_affordance(
            observations,
            affordance_key="affordance",
        )

        # Get the mask of the receptacle
        grounded_receptacle_mask = self.get_surface_grounded_affordance(
            observations,
            affordance_key="receptacle_mask",
        )
        affordance_receptacle_surface_metrics = self.semantic_precision_recall(
            grounded_affordance,
            grounded_receptacle_mask * observations["receptacle_mask"],
            mode="receptacle_surface",
        )

        affordance_target_metrics = self.semantic_precision_recall(
            observations["affordance"], observations["mask"], mode="target"
        )

        metrics = {}
        prefix = ["receptacle", "receptacle_surface", "target"]
        data = [
            affordance_receptacle_metrics,
            affordance_receptacle_surface_metrics,
            affordance_target_metrics,
        ]

        for i in range(len(prefix)):
            for k, v in data[i].items():
                metrics[f"{prefix[i]}_{k}"] = v

        return metrics

    def get_metrics(self, batch: List[Dict], mode: str = "val") -> Dict:
        """
        Compute the accuracy of the predicted affordance grounded to the surface.
        """
        metrics = defaultdict(list)
        per_sample_metrics = defaultdict(list)
        aff_gs = []
        rec_gs = []

        out_path = "data/visualizations/semantic_metrics_debug/"
        for _, observations in enumerate(batch):
            metrics["affordance_on_surface_accuracy"].append(
                self.affordance_on_surface_accuracy(observations)
            )
            metrics["affordance_on_receptacle_accuracy"].append(
                self.affordance_on_receptacle_accuracy(observations)
            )
            (
                accuracy,
                aff_g,
                rec_g,
            ) = self.affordance_on_receptacle_surface_accuracy(observations)
            metrics["affordance_on_receptacle_surface_accuracy"].append(
                accuracy
            )
            if mode == "val":
                semantic_cls_metrics = self.semantic_classification_metrics(
                    observations
                )
                for k, v in semantic_cls_metrics.items():
                    metrics[k].append(v)

            aff_gs.append(aff_g)
            rec_gs.append(rec_g)

        self.num_batches += 1

        for k, v in metrics.items():
            per_sample_metrics[k] = v
            metrics[k] = torch.tensor(np.sum(v) / len(v))

        print(
            "REcalls: {}".format(
                {k: v for k, v in metrics.items() if "recall" in k}
            )
        )

        if self.return_imgs:
            return metrics, aff_gs, rec_gs
        return metrics, per_sample_metrics
