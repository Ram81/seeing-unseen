import glob
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from seeing_unseen.core.base import BaseTrainer
from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.dataset.dataset import collate_fn
from seeing_unseen.dataset.transform_utils import TTAWrapper
from seeing_unseen.trainer.metrics import SemanticPlaceMetrics, iou
from seeing_unseen.utils.ddp_utils import rank0_only
from seeing_unseen.utils.utils import save_image, write_json
from seeing_unseen.utils.viz_utils import (
    overlay_heatmap_with_annotations,
    overlay_mask_with_gaussian_blur,
)


@registry.register_trainer(name="semantic_placement_trainer")
class SemanticPlacementTrainer(BaseTrainer):
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
        eval_dataset_dir = self.cfg.dataset.val_dir

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

        self.train_transforms = registry.get_transforms(
            self.cfg.dataset.train_transforms
        )(**transform_args, **{"original_size": original_size})
        self.val_transforms = registry.get_transforms(
            self.cfg.dataset.val_transforms
        )(**transform_args, **{"original_size": original_size})

        if self.cfg.training.eval_with_tta:
            logger.info("Using TTA for evaluation....")
            self.tta_wrapper = TTAWrapper(
                self.model,
                self.val_transforms.transforms,
                output_mask_key="affordance",
            )

        self.train_dataset = dataset_cls(
            split="train",
            trfms="none",
            root_dir=self.dataset_dir,
            load_original_image=self.cfg.training.dataset.load_original_img,
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
            "Train transfoms: {} -- {}".format(
                self.train_dataset.transforms, self.cfg.dataset.train_transforms
            )
        )
        logger.info(
            "Val transfoms: {} -- {}".format(
                self.cfg.training.eval_splits, self.cfg.dataset.val_transforms
            )
        )

        self.train_sampler = None
        self.eval_sampler = {}

        if self._is_distributed:
            self.train_sampler = DistributedSampler(self.train_dataset)

            for split in self.cfg.training.eval_splits:
                self.eval_sampler[split] = DistributedSampler(
                    self.eval_datasets[split], shuffle=False
                )

        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            sampler=self.train_sampler,
            num_workers=self.cfg.training.dataset.num_workers,
            drop_last=True,
            pin_memory=True,
        )

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
            "Training set has {} instances".format(len(self.train_dataset))
        )
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

        logger.info("Is model init distrib: {}".format(self._is_distributed))
        if self._is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False,
            )

        if self.cfg.training.pretrained:
            path = self.cfg.training.pretrained_checkpoint
            logger.info(
                "Initializing using pretrained weights from {}".format(path)
            )
            self.pretrained_state = self.load_state(path, ckpt_only=True)

        if self.cfg.training.optimizer == "Adam":
            trainable_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.training.lr,
            )
            logger.info(
                "Total trainable parameters: {}/{}".format(
                    sum([p.numel() for p in trainable_params]),
                    sum([p.numel() for p in self.model.parameters()]),
                )
            )
            logger.info("Initializing using Adam optimizer")
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                momentum=0.9,
                weight_decay=0.0005,
            )
            logger.info("Initializing using SGD optimizer")
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.cfg.training.lr_scheduler.step_decay,
            gamma=self.cfg.training.lr_scheduler.gamma,
        )

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def save_state(self, epoch):
        state_dict = {
            "epoch": epoch,
            "ckpt_dict": self.model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        model_path = os.path.join(
            self.checkpoint_dir, "ckpt_{}.pth".format(epoch)
        )
        torch.save(state_dict, model_path)

    def load_state(self, path, ckpt_only: bool = False):
        state_dict = torch.load(path, map_location="cpu")

        if "optim_dict" in state_dict and not ckpt_only:
            self.optimizer.load_state_dict(state_dict["optim_dict"])

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

    def observations_batch_from_batch(
        self, batch: Dict[str, torch.Tensor], preds: torch.Tensor
    ) -> List[Dict]:
        observations_batch = []
        for idx in range(batch["image"].shape[0]):
            obs = {
                "image": batch["image"][idx].cpu().numpy(),
                "depth": batch["depth"][idx].cpu().numpy(),
                "mask": batch["mask"][idx].cpu().numpy(),
                "affordance": preds[idx].cpu().numpy(),
                "receptacle_mask": batch["receptacle_mask"][idx].cpu().numpy(),
                "target_category": batch["target_category"][idx],
            }
            observations_batch.append(obs)
        return observations_batch

    def metrics(
        self,
        preds: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        mode: str = "val",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        metrics = defaultdict(torch.Tensor)
        dims = list(range(1, len(preds.shape)))

        labels = batch["mask"]

        # Apply sigmoid to predictions
        preds_t = (self.activation(preds) > 0.5).float()

        tp = torch.sum(preds_t * labels, dim=dims)  # TP
        fp = torch.sum(preds_t * (1 - labels), dim=dims)  # FP
        fn = torch.sum((1 - preds_t) * labels, dim=dims)  # FN
        tn = torch.sum((1 - preds_t) * (1 - labels), dim=dims)  # TN

        # is_mask_non_zero = torch.max(preds_t.view(preds_t.size(0)), dim=1)

        metrics["pixel_accuracy"] = (
            (tp + tn) / (tp + tn + fp + fn + self.eps)
        ).mean()
        metrics["dice"] = (2 * tp / (2 * tp + fp + fn + self.eps)).mean()
        metrics["precision"] = (tp / (tp + fp + self.eps)).mean()
        metrics["recall"] = (tp / (tp + fn + self.eps)).mean()
        metrics["specificity"] = (tn / (tn + fp + self.eps)).mean()
        metrics["mask_area"] = (
            preds_t.sum(dim=dims) / np.prod(preds.shape[1:])
        ).mean()

        metrics.update(**iou(preds_t, labels))
        metrics.update(**self.model_metrics())

        # Create observation batch and compute semantic metrics
        per_sample_metrics = {}
        if "depth" in batch:
            obs_batch = self.observations_batch_from_batch(batch, preds_t)
            (
                semantic_metrics,
                per_sample_metrics,
            ) = self.semantic_metrics.get_metrics(obs_batch, mode)
            for k, v in semantic_metrics.items():
                metrics[k] = v.to(metrics["dice"].device)

        return metrics, per_sample_metrics

    def model_metrics(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.model.parameters()
            if param.grad is not None
        ]
        if len(grads) == 0:
            return {}
        norm = torch.cat(grads).norm()
        return {"grad_norm": norm}

    def apply_transforms(
        self, batch: Dict[str, torch.Tensor], split: str = "train"
    ) -> Any:
        transforms = (
            self.train_transforms if split == "train" else self.val_transforms
        )
        batch["image"], batch["mask"] = transforms(
            batch["image"], batch["mask"]
        )
        return batch

    def train_one_epoch(
        self, epoch_index: int
    ) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        running_loss = 0.0
        last_loss = 0.0

        running_metrics = defaultdict(float)
        avg_metrics = defaultdict(float)

        num_batches = len(self.train_loader)

        if self._is_distributed:
            self.train_loader.sampler.set_epoch(epoch_index)

        loss_fn = registry.get_loss_fn(self.cfg.training.loss.name)(
            self.cfg.training.loss[self.cfg.training.loss.name]
        )
        logger.info(
            "[Worker {}] Loss fn: {} - {}, Rank 0 - {}".format(
                self.local_rank,
                self.cfg.training.loss.name,
                loss_fn,
                rank0_only(),
            )
        )
        self.scheduler.step()

        batch_load_time = 0
        batch_aug_time = 0
        batch_proc_time = 0
        batch_update_time = 0
        start_time = time.time()
        for i, batch in enumerate(self.train_loader):
            # logger.info(
            #     "[Process: {}] Step: {} done".format(self.local_rank, i)
            # )
            for key, val in batch.items():
                if type(val) == list:
                    continue
                batch[key] = val.float().to(self.device)
            batch_load_time += time.time() - start_time

            start_time = time.time()
            batch = self.apply_transforms(batch, split="train")
            batch_aug_time += time.time() - start_time

            update_start_time = time.time()

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(batch=batch)["affordance"].squeeze(1)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, batch["mask"])
            loss.backward()

            metrics = self.metrics(outputs, batch, mode="train")

            if torch.isnan(loss):
                o_inputs = torch.sigmoid(outputs)
                logger.info(
                    "[Process: {}] Step: {}\t Loss: {}\t Metrics: {}\t Loss pre: {}\t P Mask: {} inp: {} - {}".format(
                        self.local_rank,
                        i,
                        loss.item(),
                        metrics,
                        loss.item(),
                        o_inputs.sum((1, 2)),
                        batch["image"].min(),
                        batch["image"].max(),
                    )
                )
                torch.save(batch, "batch_{}.pth".format(self.local_rank))
                logger.info(
                    "[Process: {}] Step: {}\t inp: {}, {}".format(
                        self.local_rank, i, batch["image"], batch["original"]
                    )
                )
                sys.exit(1)

            # Adjust learning weights
            self.optimizer.step()

            batch_update_time += time.time() - update_start_time
            aggregate_start_time = time.time()

            if self._is_distributed:
                loss = (
                    self._all_reduce(loss) / torch.distributed.get_world_size()
                )

                metrics_order = sorted(metrics.keys())
                stats = torch.stack([metrics[k] for k in metrics_order])
                stats = self._all_reduce(stats)

                for k, v in zip(metrics_order, stats):
                    metrics[k] = v / torch.distributed.get_world_size()

            for k, v in metrics.items():
                running_metrics[k] += v.cpu().item()
                avg_metrics[k] += v.cpu().item() / num_batches

            # Gather data and report
            total_loss += loss.item()
            running_loss += loss.item()

            batch_proc_time += time.time() - aggregate_start_time

            if rank0_only() and i % self.log_interval == 0 and i > 0:
                last_loss = running_loss / (self.log_interval + 1)
                tb_x = epoch_index * len(self.train_loader) + i + 1

                self.log(
                    tb_x,
                    {
                        "train_per_batch/loss": last_loss,
                        "train_per_batch/learning_rate": self.scheduler.get_last_lr()[
                            0
                        ],
                    },
                )
                self.log(
                    tb_x,
                    {
                        f"train_per_batch/{k}": v / (self.log_interval + 1)
                        for k, v in running_metrics.items()
                    },
                )

                logger.info(
                    "[Process: {}] Step: {}\t Update time: {}\t Metrics time: {}".format(
                        self.local_rank,
                        i,
                        batch_update_time / (self.log_interval + 1),
                        batch_proc_time / (self.log_interval + 1),
                    )
                )
                logger.info(
                    "[Process: {}] Step: {}\t Load time: {}\t Augment time: {}".format(
                        self.local_rank,
                        i,
                        batch_load_time / (self.log_interval + 1),
                        batch_aug_time / (self.log_interval + 1),
                    )
                )
                logger.info(
                    "[Process: {}] Step: {}\t Loss:: {}\t Metrics: {}".format(
                        self.local_rank,
                        i,
                        running_loss / (self.log_interval + 1),
                        {
                            k: v / (self.log_interval + 1)
                            for k, v in running_metrics.items()
                        },
                    )
                )

                batch_load_time = 0
                batch_proc_time = 0
                batch_update_time = 0
                batch_aug_time = 0

                running_loss = 0.0
                running_metrics = defaultdict(float)

            start_time = time.time()

        avg_loss = total_loss / num_batches
        return avg_loss, avg_metrics

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
        sampels_per_category = defaultdict(int)

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

                if self.cfg.training.eval_with_tta:
                    voutputs = self.tta_wrapper(batch)["affordance"].squeeze(1)
                    batch["image"] /= 255.0
                else:
                    batch = self.apply_transforms(batch, split="val")

                    voutputs = self.model(batch=batch)["affordance"].squeeze(1)
                vloss = loss_fn(voutputs, batch["mask"])

                img_batch_npy = batch["image"].cpu().numpy()
                mask_batch_npy = batch["mask"].cpu().numpy()
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
                else:
                    target_query.extend(
                        [img for img in batch["target_query"].cpu().numpy()]
                    )
                    input_imgs.extend([img for img in img_batch_npy])
                    gt_masks.extend([mask for mask in mask_batch_npy])
                    pred_logits.extend([mask for mask in outputs_batch_npy])

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
                            print(new_record)
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

                if os.path.isdir(self.cfg.visualization.metrics_dir):
                    write_json(
                        eval_metrics,
                        os.path.join(
                            self.cfg.visualization.metrics_dir,
                            "eval_metrics_{}.json".format(epoch),
                        ),
                    )

                total_vloss += vloss

        if epoch % self.cfg.visualization.interval == 0 and rank0_only():
            self.visualize(
                input_imgs,
                gt_masks,
                pred_logits,
                target_query,
                epoch,
                val_split,
            )

        avg_vloss = total_vloss / num_batches
        return avg_vloss, avg_metrics

    def train(self):
        EPOCHS = self.cfg.training.epochs

        logger.info(
            "[Worker {}] Is distributed: {}".format(
                self.local_rank, self._is_distributed
            )
        )
        logger.info(
            "[Worker {}] Train size: {}, Val size: {}".format(
                self.local_rank,
                len(self.train_loader),
                [
                    len(self.eval_loader[split])
                    for split in self.cfg.training.eval_splits
                ],
            )
        )
        # Synchronize all processes
        if self._is_distributed:
            torch.distributed.barrier()

        logger.info("[Process: {}] Starting training".format(self.local_rank))
        for epoch in range(self.pretrained_state["epoch"], EPOCHS):
            logger.info(
                "[Process: {}] EPOCH {}:".format(self.local_rank, epoch + 1)
            )

            # Train model
            self.model.train()
            avg_loss, avg_metrics = self.train_one_epoch(epoch)

            logger.info(
                "[Process: {}] Synchronize training processes".format(
                    self.local_rank
                )
            )
            # Synchronize all processes
            if self._is_distributed:
                torch.distributed.barrier()

            logger.info("[Process: {}] Evaluating...".format(self.local_rank))
            # Evaluate model
            self.model.eval()

            eval_metrics = {}

            for split in self.cfg.training.eval_splits:
                avg_eval_loss, avg_eval_metrics = self.evaluate(
                    epoch, self.eval_loader[split], val_split=split
                )
                eval_metrics[split] = {
                    "loss": avg_eval_loss,
                    "metrics": avg_eval_metrics,
                }

            val_losses = [
                eval_metrics[split]["loss"]
                for split in self.cfg.training.eval_splits
            ]
            val_metrics = [
                eval_metrics[split]["metrics"]
                for split in self.cfg.training.eval_splits
            ]

            if rank0_only():
                logger.info(
                    "[Epoch {}] Train loss: {}, metrics: {}".format(
                        epoch, avg_loss, avg_metrics
                    )
                )
                logger.info(
                    "[Epoch {}] Val loss: {}, metrics: {}".format(
                        epoch, val_losses, val_metrics
                    )
                )

                self.log(
                    epoch + 1,
                    {
                        "train/loss_per_epoch": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                    },
                )
                self.log(
                    epoch + 1,
                    {f"train/{k}_per_epoch": v for k, v in avg_metrics.items()},
                )
                for split in self.cfg.training.eval_splits:
                    self.log(
                        epoch + 1,
                        {
                            f"{split}/loss_per_epoch": eval_metrics[split][
                                "loss"
                            ]
                        },
                    )
                    self.log(
                        epoch + 1,
                        {
                            f"{split}/{k}_per_epoch": v
                            for k, v in eval_metrics[split]["metrics"].items()
                        },
                    )

                self.save_state(epoch + 1)

            # Synchronize all processes
            if self._is_distributed:
                torch.distributed.barrier()

    def eval(self, checkpoint_dir):
        checkpoints = [checkpoint_dir]
        if os.path.isdir(checkpoint_dir):
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
            checkpoint_mp = {}
            for ckpt in checkpoints:
                epoch = int(ckpt.split("/")[-1].split(".")[0].split("_")[-1])
                checkpoint_mp[epoch] = ckpt
            checkpoints = [
                checkpoint_mp[k] for k in sorted(checkpoint_mp.keys())
            ]

        logger.info(
            "[Eval Worker {}] Is distributed: {}, Checkpoints: {}".format(
                self.local_rank, self._is_distributed, len(checkpoints)
            )
        )
        logger.info(
            "[Eval Worker {}] Val {} dataset size: {}".format(
                self.local_rank,
                self.cfg.training.eval_splits,
                [
                    len(self.eval_loader[split])
                    for split in self.cfg.training.eval_splits
                ],
            )
        )
        for ckpt_idx, ckpt_path in tqdm(enumerate(checkpoints)):
            logger.info("Checkpoint {}: {}".format(ckpt_idx + 1, ckpt_path))
            epoch = int(ckpt_path.split("/")[-1].split(".")[0].split("_")[-1])

            # Load checkpoint
            self.load_state(ckpt_path, ckpt_only=True)

            # Evaluate model
            self.model.eval()

            eval_metrics = {}
            for split in self.cfg.training.eval_splits:
                avg_eval_loss, avg_eval_metrics = self.evaluate(
                    epoch, self.eval_loader[split], val_split=split
                )
                eval_metrics[split] = {
                    "loss": avg_eval_loss,
                    "metrics": avg_eval_metrics,
                }

            val_losses = [
                eval_metrics[split]["loss"]
                for split in self.cfg.training.eval_splits
            ]
            val_metrics = [
                eval_metrics[split]["metrics"]
                for split in self.cfg.training.eval_splits
            ]

            if rank0_only():
                logger.info(
                    "[Epoch {}] Val loss: {}, metrics: {}".format(
                        epoch, val_losses, val_metrics
                    )
                )
                for split in self.cfg.training.eval_splits:
                    self.log(
                        epoch + 1,
                        {
                            f"{split}/loss_per_epoch": eval_metrics[split][
                                "loss"
                            ]
                        },
                    )
                    self.log(
                        epoch + 1,
                        {
                            f"{split}/{k}_per_epoch": v
                            for k, v in eval_metrics[split]["metrics"].items()
                        },
                    )

    def visualize(
        self,
        input_img: List[np.ndarray],
        targets: List[np.ndarray],
        preds: List[np.ndarray],
        target_query: List[np.ndarray],
        epoch: int,
        split: str,
    ):
        if not self.cfg.visualization.visualize:
            return

        output_dir = os.path.join(
            self.cfg.visualization.output_dir,
            "{}/epoch_{}".format(split, epoch),
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving visualizations to {}".format(output_dir))

        sample_size = self.cfg.visualization.sample_size
        if sample_size == -1:
            sample_size = len(input_img)
        logger.info(
            "Total samples: {} - {} - {}".format(
                len(input_img), len(target_query), sample_size
            )
        )

        op_masks = os.path.join(output_dir, "heatmap")
        os.makedirs(op_masks, exist_ok=True)

        id_label_masks = os.path.join(output_dir, "individual_masks")
        os.makedirs(id_label_masks, exist_ok=True)

        activation = nn.Sigmoid()
        for sample_idx, (img, target, logits, query) in enumerate(
            zip(input_img, targets, preds, target_query)
        ):
            img = np.transpose(img * 255, (1, 2, 0)).astype(np.uint8)
            target = (target * 255).astype(np.uint8)
            pred_mask = activation(torch.from_numpy(logits)).numpy()
            pred_heatmap = ((1 - pred_mask) * 255).astype(np.uint8)

            if self.cfg.visualization.save_observations:
                superimposed_affordance = overlay_mask_with_gaussian_blur(
                    pred_mask > 0.5, img
                )

                # overlayed_preds = overlay_heatmap(img, pred_heatmap)
                save_image(
                    superimposed_affordance,
                    os.path.join(
                        op_masks,
                        "pred_{}_{}_mask_.png".format(query, sample_idx),
                    ),
                )

            if self.cfg.visualization.save_separate_masks:
                superimposed_masks = overlay_heatmap_with_annotations(
                    img, pred_mask > 0.5
                )

                for idd in range(len(superimposed_masks[:5])):
                    save_image(
                        superimposed_masks[idd],
                        os.path.join(
                            id_label_masks,
                            "pred_{}_{}_mask_{}.png".format(
                                query, sample_idx, idd
                            ),
                        ),
                    )

            # if "text" in self.cfg.dataset.name:
            #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
            #     ax1.imshow(img)
            #     ax1.set_title("Input image \n Query: {}".format(query))
            #     ax2.imshow(overlay_semantic_mask(img, target))
            #     ax2.set_title("Ground truth mask")
            #     ax3.imshow(overlay_heatmap(img, pred_heatmap))
            #     ax3.set_title("Predicted mask")
            #     sns.heatmap(logits, ax=ax4, xticklabels=100, yticklabels=100)
            #     ax4.set_aspect(
            #         logits.shape[1] / logits.shape[0]
            #     )  # here 0.5 Y/X ratio
            #     ax4.set_title("Predicted logits")

            #     fig.savefig(
            #         os.path.join(
            #             output_dir, "pred_{}_{}.png".format(query, sample_idx)
            #         )
            #     )
            #     plt.close(fig)
            # else:
            #     query = np.transpose(query * 255, (1, 2, 0)).astype(np.uint8)

            #     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            #         1, 5, figsize=(15, 5)
            #     )
            #     ax1.imshow(img)
            #     ax1.set_title("Input image")
            #     ax2.imshow(query)
            #     ax2.set_title("Target patch")
            #     ax3.imshow(overlay_semantic_mask(img, target))
            #     ax3.set_title("Ground truth mask")
            #     ax4.imshow(overlay_heatmap(img, pred_heatmap))
            #     ax4.set_title("Predicted mask")
            #     sns.heatmap(logits, ax=ax5, xticklabels=100, yticklabels=100)
            #     ax5.set_aspect(
            #         logits.shape[1] / logits.shape[0]
            #     )  # here 0.5 Y/X ratio
            #     ax5.set_title("Predicted logits")

            #     fig.savefig(
            #         os.path.join(output_dir, "pred_{}.png".format(sample_idx))
            #     )
            #     plt.close(fig)

            if sample_idx > sample_size:
                break
