import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.dataset.dataset import collate_fn
from seeing_unseen.dataset.transform_utils import TTAWrapper
from seeing_unseen.trainer.metrics import SemanticPlaceMetrics, iou
from seeing_unseen.trainer.trainer import SemanticPlacementTrainer
from seeing_unseen.utils.ddp_utils import rank0_only
from seeing_unseen.utils.utils import write_json


@registry.register_trainer(name="semantic_placement_evaluator")
class SemanticPlacementEvaluator(SemanticPlacementTrainer):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_dir: str,
        checkpoint_dir: str,
        log_dir: str,
    ) -> None:
        super().__init__(cfg, dataset_dir, checkpoint_dir, log_dir)

    def init_dataset(self) -> None:
        # Create datasets for training & validation, download if necessary
        dataset_cls = registry.get_dataset(self.cfg.dataset.name)
        eval_dataset_dir = self.cfg.dataset.root_dir

        transform_args = self.cfg.dataset.transform_args
        original_size = self.input_shape
        if (
            transform_args["random_resize_crop_prob"] > 0
            or transform_args["resize_prob"] > 0
        ):
            self.input_shape = [
                int(i) for i in transform_args["resized_resolution"]
            ]

        logger.info(
            "Transform args: {} - {} - {}".format(
                transform_args,
                self.input_shape,
                [type(i) for i in self.input_shape],
            )
        )

        self.val_transforms = registry.get_transforms(
            self.cfg.dataset.val_transforms
        )(**transform_args, **{"original_size": self.input_shape})

        if self.cfg.training.eval_with_tta:
            logger.info("Using TTA for evaluation....")
            self.tta_wrapper = TTAWrapper(
                self.model,
                self.val_transforms.transforms,
                output_mask_key="affordance",
            )

        self.eval_datasets = {}
        for split in self.cfg.training.eval_splits:
            dataset = dataset_cls(
                split=split,
                trfms="none",
                root_dir=eval_dataset_dir,
            )
            self.eval_datasets[split] = dataset

        logger.info(
            "Val transfoms: {} -- {}".format(
                self.cfg.training.eval_splits, self.cfg.dataset.val_transforms
            )
        )

        self.train_sampler = None
        self.eval_sampler = {}

        if self._is_distributed:
            for split in self.cfg.training.eval_splits:
                self.eval_sampler[split] = DistributedSampler(
                    self.eval_datasets[split], shuffle=False
                )

        # Initialize data loaders
        self.eval_loader = {}
        for split in self.cfg.training.eval_splits:
            self.eval_loader[split] = DataLoader(
                self.eval_datasets[split],
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                sampler=self.eval_sampler.get(split),
                num_workers=self.cfg.training.dataset.num_workers,
                shuffle=False,
                pin_memory=True,
            )

        # Report split sizes
        logger.info(
            "Validation set has {} instances".format(
                [
                    len(self.eval_datasets[split])
                    for split in self.cfg.training.eval_splits
                ]
            )
        )

        # Intialize metrics wrapper
        self.semantic_metrics = SemanticPlaceMetrics(self.input_shape, 90)

    def init_model(self) -> None:
        model_cls = registry.get_affordance_model(self.cfg.model.name)
        self.model = model_cls(
            input_shape=self.input_shape,
            target_input_shape=self.target_input_shape,
        ).to(self.device)
        self.pretrained_state = defaultdict(int)

        if self.cfg.training.pretrained:
            path = self.cfg.training.pretrained_checkpoint
            logger.info(
                "Initializing using pretrained weights from {}".format(path)
            )
            self.pretrained_state = self.load_state(path, ckpt_only=True)

        if self._is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False,
            )

    def load_state(self, path, ckpt_only: bool = False):
        if not os.path.exists(path):
            logger.info("No checkpoint found at {}".format(path))
            return {"epoch": 0}

        state_dict = torch.load(path, map_location="cpu")

        if len(state_dict.keys()) == 0:
            return {"epoch": 0}

        # To handle case when model is saved before commit 862850571316b82613dad67525f8c1bf643b4f10
        ckpt_dict = (
            state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict
        )
        if not self._is_distributed:
            missing_keys = self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
            )
        else:
            missing_keys = self.model.load_state_dict(ckpt_dict)
        logger.info("Missing keys: {}".format(missing_keys))
        return {
            "epoch": (
                state_dict["epoch"]
                if "epoch" in state_dict and not ckpt_only
                else 0
            ),
        }

    def evaluate(
        self, epoch, val_loader, val_split
    ) -> Tuple[float, Dict[str, float]]:
        loss_fn = registry.get_loss_fn(self.cfg.training.loss.name)(
            self.cfg.training.loss[self.cfg.training.loss.name]
        )
        total_vloss = 0.0

        avg_metrics = defaultdict(float)
        num_batches = len(val_loader)

        input_imgs = []
        gt_masks = []
        pred_logits = []
        target_query = []
        receptacle_masks = []
        sampels_per_category = defaultdict(int)
        print("Metrics Shape: {}".format(self.input_shape))

        avg_eval_time = 0

        # Disable gradient computation and reduce memory consumption.
        eval_metrics = []
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(val_loader), disable=not rank0_only()
            ):
                for key, val in batch.items():
                    if type(val) == list:
                        continue
                    batch[key] = batch[key].float().to(self.device)

                start_time = time.time()
                if self.cfg.training.eval_with_tta:
                    voutputs = self.tta_wrapper(batch)["affordance"].squeeze(1)
                    batch["image"] /= 255.0
                else:
                    batch = self.apply_transforms(batch, split="val")

                    voutputs = self.model(batch=batch)["affordance"].squeeze(1)
                start_time = time.time() - start_time
                avg_eval_time += start_time
                vloss = loss_fn(voutputs, batch["mask"])

                img_batch_npy = batch["image"].cpu().numpy()
                mask_batch_npy = batch["mask"].cpu().numpy()
                receptacle_mask_batch_npy = (
                    batch["receptacle_mask"].cpu().numpy()
                )
                outputs_batch_npy = voutputs.cpu().numpy()
                if "text" in self.cfg.dataset.name:
                    categories = batch["target_category"]
                    for idx, category in enumerate(categories):
                        sampels_per_category[category] += 1
                        if sampels_per_category[category] <= 100:
                            target_query.append(category)
                            input_imgs.append(img_batch_npy[idx])
                            gt_masks.append(mask_batch_npy[idx])
                            pred_logits.append(outputs_batch_npy[idx])
                            receptacle_masks.append(
                                receptacle_mask_batch_npy[idx]
                            )
                else:
                    target_query.extend(
                        [img for img in batch["target_query"].cpu().numpy()]
                    )
                    input_imgs.extend([img for img in img_batch_npy])
                    gt_masks.extend([mask for mask in mask_batch_npy])
                    pred_logits.extend([mask for mask in outputs_batch_npy])
                    receptacle_masks.extend(
                        [mask for mask in receptacle_mask_batch_npy]
                    )

                metrics, per_sample_metrics = self.metrics(voutputs, batch)
                if os.path.isdir(self.cfg.visualization.metrics_dir):
                    for idx, category in enumerate(batch["target_category"]):
                        new_record = {
                            "category": category,
                            **{
                                k: v[idx] for k, v in per_sample_metrics.items()
                            },
                        }
                        if idx == 0:
                            print("sample metric", new_record)
                        eval_metrics.append(new_record)

                if self._is_distributed:
                    vloss = (
                        self._all_reduce(vloss)
                        / torch.distributed.get_world_size()
                    )

                    metrics_order = sorted(metrics.keys())
                    stats = torch.stack([metrics[k] for k in metrics_order])
                    stats = self._all_reduce(stats)

                    for k, v in zip(metrics_order, stats):
                        metrics[k] = v / torch.distributed.get_world_size()

                for k, v in metrics.items():
                    avg_metrics[k] += v.cpu().item() / num_batches

                total_vloss += vloss

            if os.path.isdir(self.cfg.visualization.metrics_dir):
                write_json(
                    eval_metrics,
                    os.path.join(
                        self.cfg.visualization.metrics_dir,
                        "eval_metrics_{}.json".format(epoch),
                    ),
                )

        if epoch % self.cfg.visualization.interval == 0 and rank0_only():
            self.visualize(
                input_imgs,
                receptacle_masks,
                pred_logits,
                target_query,
                epoch,
                val_split,
            )

        print("Avg eval time: {}".format(avg_eval_time / num_batches))

        avg_vloss = total_vloss / num_batches
        return avg_vloss, avg_metrics
