import argparse
import glob
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from seeing_unseen.models.encoders.detic_perception import DeticPerception
from seeing_unseen.third_party.sdedit import denoise_base, load_denoise_base
from seeing_unseen.third_party.sdedit import load_up as load_denoise_up
from seeing_unseen.third_party.sdedit import upsample as denoised_up
from seeing_unseen.utils.bbox_utils import BBoxUtils
from seeing_unseen.utils.utils import (
    binary_mask_to_rle,
    decode_rle_mask,
    dilate_mask,
    load_image,
    load_json,
    random_id,
    save_array_to_img,
    save_image,
    show_mask,
    show_points,
    write_json,
)
from seeing_unseen.utils.viz_utils import (
    overlay_semantic_mask,
    visualize_bounding_boxes,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class SemanticPlacementGenerator:
    def __init__(
        self,
        split: str = "train",
        point_labels: int = 1,
        dilate_kernel_size: int = 15,
        sam_model_type: str = "vit_h",
        sam_ckpt: str = "data/pretrained_models/sam_vit_h_4b8939.pth",
        lama_config: str = "config/lama/default.yaml",
        lama_ckpt: str = "data/pretrained_models/big-lama",
        output_dir: str = "data/impaint_anything/results/",
        use_blip2: bool = False,
        visualize: bool = False,
        uuid: str = "split_1",
        inpainting_model: str = "lama",
        augmentation_model: str = "both",
        segmentation_model: str = "detic",
        validation_method: str = "detector",
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dilate_kernel_size = dilate_kernel_size
        self.point_labels = point_labels
        self.output_dir = output_dir
        self.visualize = visualize
        self.use_blip2 = use_blip2
        self.split = split
        self.crop_shape = (128, 128)
        self.uuid = uuid
        self.inpainting_model = inpainting_model
        self.augmentation_model = augmentation_model
        self.segmentation_model = segmentation_model
        self.validation_method = validation_method

        print("Metadata:")
        print("Inpainting model: {}".format(self.inpainting_model))
        print("Validation method: {}".format(self.validation_method))
        print("Augmentation method: {}".format(self.augmentation_model))
        print("Segmentation method: {}".format(self.segmentation_model))

        self.record_id = 0
        self.records = []

        self.init_segmentation()
        self.init_lama(lama_config, lama_ckpt)
        self.init_sam(sam_model_type, sam_ckpt)
        self.init_glide()
        self.init_stable_diffusion()
        self.init_dirs()

    def init_segmentation(self):
        self.vocabulary = ".,plant,potted_plant,pillow,toaster,laptop,lamp,table_lamp,alarm_clock,vase,trash_can,garbage_can,coffee_table,chair,stool,cup,mug,plate,bowl"
        if self.segmentation_model == "detic":
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=self.vocabulary,
            )
        else:
            raise NotImplementedError
        self.vocabulary = self.vocabulary.replace("_", " ").split(",")

    @torch.no_grad()
    def init_lama(self, config_p: str, ckpt_p: str):
        self.lama_config = OmegaConf.load(config_p)
        self.lama_config.model.path = ckpt_p

        train_config_path = os.path.join(
            self.lama_config.model.path, "config.yaml"
        )

        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(
            self.lama_config.model.path,
            "models",
            self.lama_config.model.checkpoint,
        )
        self.lama = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location="cpu"
        )
        self.lama.freeze()
        if not self.lama_config.get("refine", False):
            self.lama.to(self.device)

    def init_sam(self, model_type: str, ckpt_p: str):
        sam = sam_model_registry[model_type](checkpoint=ckpt_p)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def init_stable_diffusion(self):
        self.sd = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        self.sd = self.sd.to("cuda")

        self.sd_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.sd_img2img = self.sd_img2img.to("cuda")

    def init_glide(self):
        glide = {}
        glide["denoise"] = load_denoise_base()
        glide["denoise_up"] = load_denoise_up()
        self.glide = glide

    def init_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "detic"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "original"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    def predict_masks_with_sam(
        self,
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
    ):
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        self.sam.set_image(img)
        masks, scores, logits = self.sam.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return masks, scores, logits

    def sd_augment(self, image):
        prompt = "high resolution, 4k"
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        image_aug = self.sd_img2img(
            prompt=prompt, image=image, strength=0.05, guidance_scale=7.5
        ).images[0]

        if image.size != image_aug.size:
            image_aug = image_aug.resize((image.size[0], image.size[1]))
        return np.array(image_aug).astype(np.uint8)

    @torch.no_grad()
    def glide_augment(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        inp_image = img.resize((256, 256), resample=Image.BICUBIC)
        inp_image = (
            torch.from_numpy(np.array(inp_image).astype(np.uint8)) / 127.5 - 1
        )

        inp_image = F.avg_pool2d(inp_image.permute(2, 0, 1).unsqueeze(0), 4)
        out_image64 = denoise_base(
            0.05, *self.glide["denoise"], inp_image.cuda(), "high resolution"
        )
        cur_res = denoised_up(
            0, *self.glide["denoise_up"], out_image64, "high resolution"
        )

        out_image = (
            ((cur_res + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        )
        out_image = out_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_image = Image.fromarray(out_image)
        out_image = out_image.resize(img.size)
        return np.array(out_image).astype(np.uint8)

    def augment_inpainted_img(self, img):
        if self.augmentation_model == "sdedit":
            return [(self.glide_augment(img), "glide")]
        elif self.augmentation_model == "sd":
            return [(self.sd_augment(img), "sd")]
        elif self.augmentation_model == "both":
            aug_imgs = []
            aug_imgs.append((self.sd_augment(img), "sd"))
            aug_imgs.append((self.glide_augment(img), "glide"))
            return aug_imgs
        # If no valid model specified use randomly
        if random.random() > 0.4:
            return [(self.sd_augment(img), "sd")]
        return [(self.glide_augment(img), "glide")]

    def inpaint_with_sd(self, img, mask):
        prompt = "background"
        num_inference_steps = 50
        negative_prompt = None
        guidance_scale = 7.5

        assert len(mask.shape) == 2
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        inpainted_img = self.sd(
            prompt=prompt,
            image=img,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        ).images[0]

        inpaint_aug = self.augment_inpainted_img(inpainted_img)
        return np.array(inpainted_img), inpaint_aug

    @torch.no_grad()
    def inpaint_img_with_lama(self, inp_img, mask, mod: int = 8):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(inp_img).float().div(255.0)
        mask = torch.from_numpy(mask).float()

        batch = {}
        batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
        batch["mask"] = mask[None, None]
        unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
        batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
        batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
        batch = move_to_device(batch, self.device)
        batch["mask"] = (batch["mask"] > 0) * 1

        if not self.lama_config.refine:
            batch = self.lama(batch)
            cur_res = batch[self.lama_config.out_key][0].permute(1, 2, 0)
            cur_res = cur_res.detach().cpu().numpy()

            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]
        else:
            batch["unpad_to_size"] = [
                torch.tensor([batch["image"].shape[2]]),
                torch.tensor([batch["image"].shape[3]]),
            ]
            batch = refine_predict(batch, self.lama, **self.lama_config.refiner)
            cur_res = batch[0].permute(1, 2, 0).detach().cpu().numpy()

        orig_inpaint_image = np.clip(cur_res * 255, 0, 255).astype("uint8")

        # Resize and augment using GLIDE/Stable diffusion
        inpaint_aug = self.augment_inpainted_img(orig_inpaint_image)
        return orig_inpaint_image, inpaint_aug

    def load_image(self, path):
        img = load_image(path)
        img = img.resize((512, 512), resample=Image.BICUBIC)
        return np.array(img).astype(np.uint8)

    def merge_bounding_boxes(self, object_instance_to_bbox: Dict) -> Dict:
        boxes = [bb[1] for bb in list(object_instance_to_bbox.values())]
        instance_ids = list(object_instance_to_bbox.keys())

        boxes = sorted(
            zip(boxes, instance_ids), key=lambda pair: (pair[0][1], pair[0][0])
        )

        num_boxes = len(boxes)
        offset = [2.0, 2.0, 2.0, 2.0]

        nuked_box = [0 for i in range(num_boxes)]
        for i in range(num_boxes - 1):
            box1, instance1_id = boxes[i]
            category1 = instance1_id.split("_")[0]
            if category1 not in ["book", "cup", "mug", "plate", "bowl"]:
                continue
            if nuked_box[i]:
                continue
            for j in range(i + 1, num_boxes):
                box2, instance2_id = boxes[j]
                category2 = instance2_id.split("_")[0]
                if category2 not in ["book", "cup", "mug", "plate", "bowl"]:
                    continue
                if not category1 == category2:
                    continue
                if BBoxUtils.bbox_overlap_2d(box1, box2, offset):
                    boxes[i] = (
                        BBoxUtils.bbox_merge_2d(box1, box2),
                        instance1_id,
                    )
                    nuked_box[j] = 1

        filtered_object_instance_to_bbox = {}
        for i in range(num_boxes):
            if not nuked_box[i]:
                box, instance_id = boxes[i]
                bbox_center = (box[:2] + box[2:]) / 2
                filtered_object_instance_to_bbox[instance_id] = (
                    bbox_center.tolist(),
                    box.tolist(),
                )

        return filtered_object_instance_to_bbox

    def get_bbox_or_mask_center(self, bbox, mask):
        # return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        return (bbox[:2] + bbox[2:]) / 2

    def detect_objects(self, observation, input_img_path):
        """
        Args:
            observation (np.ndarray): The input image.
        Returns:
            results (dict): The detection results.
            results.preds (Instances): The detected instances using detic. Each instance
            has the following fields:
                - pred_boxes (Boxes): The bounding box of this instance.
                - scores (Tensor): The classification scores of this instance.
                - pred_classes (Tensor): The predicted label of this instance.
                - pred_masks (Tensor): The segmentation mask of this instance.
        """
        results = self.segmentation.predict(
            observation, draw_instance_predictions=True
        )
        if self.visualize:
            img_stem = Path(input_img_path).stem
            out_dir = Path(self.output_dir) / "detic"
            save_array_to_img(
                results["semantic_frame"],
                Path(out_dir) / f"{img_stem}_detic.png",
            )

        obj_instances = results["preds"]["instances"]
        object_category_to_instance_id = defaultdict(list)
        object_instance_to_bbox = defaultdict(Tuple)
        object_instance_to_mask = defaultdict(np.ndarray)
        boxes = []
        categories = []
        for category_id, bbox, score in zip(
            obj_instances.pred_classes,
            obj_instances.pred_boxes,
            obj_instances.scores,
        ):
            if category_id is None or score < 0.5:
                continue

            if isinstance(category_id, torch.Tensor):
                category_id = category_id.detach().cpu().item()
            category_name = self.vocabulary[category_id]
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.detach().cpu().numpy()
            # mask = mask.detach().cpu().numpy()

            instance_id = "{}_{}".format(
                category_name,
                len(object_category_to_instance_id[category_name]),
            )

            # Store only 4 instances per category
            if len(object_category_to_instance_id[category_name]) >= 4:
                continue

            bbox_center = self.get_bbox_or_mask_center(bbox, None)

            object_category_to_instance_id[category_name].append(instance_id)
            object_instance_to_bbox[instance_id] = (bbox_center, bbox)

            # object_instance_to_mask[instance_id] = mask * 255
            boxes.append((bbox_center, bbox.tolist()))
            categories.append(category_name)

        object_instance_to_bbox = self.merge_bounding_boxes(
            object_instance_to_bbox
        )

        if self.visualize:
            annotated_frame = visualize_bounding_boxes(
                observation, boxes, categories
            )
            save_array_to_img(
                annotated_frame,
                Path(out_dir) / f"{img_stem}_detic_filtered.png",
            )

        return object_instance_to_bbox, object_instance_to_mask

    def sam_mask(self, img, latest_coords, instance_ids, input_img_path):
        point_labels = [self.point_labels for _ in range(len(latest_coords))]
        masks, scores, _ = self.predict_masks_with_sam(
            img,
            latest_coords,
            point_labels,
        )
        masks = masks.astype(np.uint8) * 255

        # dilate mask to avoid unmasked edge effect
        if self.dilate_kernel_size is not None:
            masks = [
                dilate_mask(mask, self.dilate_kernel_size) for mask in masks
            ]

        # visualize the segmentation results
        if self.visualize:
            img_stem = Path(input_img_path).stem
            out_dir = Path(self.output_dir) / img_stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, mask in enumerate(masks):
                # path to the results
                mask_p = out_dir / "record_{}_mask_{}_{}.png".format(
                    self.record_id, "_".join(instance_ids), idx
                )
                img_points_p = out_dir / f"with_points.png"
                img_mask_p = out_dir / f"with_{Path(mask_p).name}"

                # save the mask
                # save_array_to_img(mask, mask_p)

                # save the pointed and masked image
                dpi = plt.rcParams["figure.dpi"]
                height, width = img.shape[:2]
                plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
                plt.imshow(img)
                plt.axis("off")
                show_points(
                    plt.gca(),
                    latest_coords,
                    point_labels,
                    size=(width * 0.04) ** 2,
                )
                plt.savefig(img_points_p, bbox_inches="tight", pad_inches=0)
                show_mask(plt.gca(), mask, random_color=False)
                plt.savefig(img_mask_p, bbox_inches="tight", pad_inches=0)
                plt.close()

        return masks, scores

    def get_sam_masks(
        self, instance_ids, object_instance_to_bbox, img, input_img
    ):
        instance_id_to_mask = {}
        for i in range(len(instance_ids)):
            instance_id = instance_ids[i]
            img_cpy = img.copy()

            bbox_center, bbox = object_instance_to_bbox[instance_id]
            masks, scores = self.sam_mask(
                img_cpy, [bbox_center], [instance_id], input_img
            )

            # inpaint the masked image
            mask_idx = 0
            whitelisted_objs = [
                "book",
                "laptop",
                "cup",
                "mug",
                "plate",
                "bowl",
                "coffee",
                "chair",
                "stool",
            ]
            for obj_whitelisted in whitelisted_objs:
                if obj_whitelisted in instance_id:
                    mask_idx = np.argmax(scores)
                    break

            mask = masks[mask_idx]
            binary_mask = (mask == 255).astype(np.uint8)
            # print("Instance: {}, Mask sum: {}".format(instance, binary_mask.sum() / np.prod(binary_mask.shape)))
            if binary_mask.sum() / np.prod(binary_mask.shape) > 0.1:
                continue
            instance_id_to_mask[instance_id] = mask
        return instance_id_to_mask

    def inpaint(
        self,
        input_img: str,
        records: List[Dict[str, Any]] = [],
        process_id: int = 0,
        file_id: int = 0,
    ):
        img = self.load_image(input_img)

        # Detec objects using Detic
        object_instance_to_bbox, _ = self.detect_objects(img, input_img)

        if len(object_instance_to_bbox.keys()) < 1:
            return (
                object_instance_to_bbox,
                0,
                {
                    "reason_discarded": "not enough objects",
                    "count": 1,
                    "object_instance_to_bbox": object_instance_to_bbox,
                },
            )

        instance_ids = list(object_instance_to_bbox.keys())
        instance_id_to_masks = self.get_sam_masks(
            instance_ids, object_instance_to_bbox, img, input_img
        )
        instance_ids = list(instance_id_to_masks.keys())

        print("IIDs: {}".format(instance_ids))
        print("IIDs mask: {}".format(instance_id_to_masks.keys()))
        num_inpainted = 0
        total_samples = 0
        for i in range(len(instance_ids)):
            instance1_id = instance_ids[i]
            instance1_bbox_center, instance1_bbox = object_instance_to_bbox[
                instance1_id
            ]

            for j in range(i, len(instance_ids)):
                if j - i > 1:
                    break

                total_samples += 1
                instances = [instance1_id]
                bbox_centers = [instance1_bbox_center]
                bboxes = [instance1_bbox]

                instance2_id = instance_ids[j]
                instance2_bbox_center, instance2_bbox = object_instance_to_bbox[
                    instance2_id
                ]

                # Add distractors after generating inpainted image for single object
                if i != j:
                    instances.append(instance2_id)
                    bbox_centers.append(instance2_bbox_center)
                    bboxes.append(instance2_bbox)

                    # Randomly add distractor instances
                    leftover_instances = (
                        instance_ids[:i]
                        + instance_ids[i + 1 : j - 1]
                        + instance_ids[j + 1 :]
                    )
                    if len(leftover_instances) > 0:
                        samples = random.choice(
                            list(range(1, min(2, len(leftover_instances)) + 1))
                        )
                        distractor_instances = random.sample(
                            leftover_instances, samples
                        )
                        for d_instance in distractor_instances:
                            (
                                d_instance_bbox_center,
                                d_instance_bbox,
                            ) = object_instance_to_bbox[d_instance]
                            instances.append(d_instance)
                            bbox_centers.append(d_instance_bbox_center)
                            bboxes.append(d_instance_bbox)

                img_stem = Path(input_img).stem
                out_dir = Path(self.output_dir) / img_stem

                img_inpainted = img.copy()
                img_sd_augmented = []
                object_masks = []
                obj_mask = np.zeros(img.shape[:2])
                for bbox_center, instance in zip(bbox_centers, instances):
                    mask = instance_id_to_masks[instance]
                    obj_mask += mask

                    mask_p = out_dir / "record_{}_mask_{}_0.png".format(
                        self.record_id, "_".join(instances)
                    )
                    img_inpainted_p = (
                        out_dir / f"inpainted_with_{Path(mask_p).name}"
                    )

                    if self.inpainting_model == "sd" or random.random() < 0.4:
                        img_inpainted, img_sd_augmented = self.inpaint_with_sd(
                            img_inpainted, mask
                        )
                    else:
                        (
                            img_inpainted,
                            img_sd_augmented,
                        ) = self.inpaint_img_with_lama(img_inpainted, mask)
                    # img_inpainted = self.inpaint_with_sd(img_inpainted, mask)
                    object_masks.append((mask > 0).astype(np.uint8))

                    if self.visualize:
                        save_array_to_img(img_inpainted, img_inpainted_p)

                # Validate impainted image with Detic instance matching
                is_inpainted = self.validate_with_detector(
                    img_inpainted, bboxes, instances, out_dir
                )

                if self.visualize and not all(is_inpainted):
                    save_array_to_img(
                        img_inpainted,
                        out_dir
                        / "record_{}_failed_inpainted_with_{}.png".format(
                            self.record_id, "_".join(instances)
                        ),
                    )
                elif all(is_inpainted):
                    # out_success_dir = Path(self.output_dir) / img_stem / "success"
                    # os.makedirs(out_success_dir, exist_ok=True)
                    # save_array_to_img(img_inpainted, out_success_dir / "success_inpainted_with_{}.png".format("_".join(instances)))
                    record = self.sample_to_record(
                        img,
                        img_inpainted,
                        object_masks,
                        bboxes,
                        input_img,
                        instances,
                        process_id,
                        file_id=file_id,
                    )
                    if record is not None:
                        records.append(record)

                    if (
                        img_sd_augmented is not None
                        and len(img_sd_augmented) > 0
                    ):
                        for img_sd, aug_model in img_sd_augmented:
                            record = self.sample_to_record(
                                img,
                                img_sd,
                                object_masks,
                                bboxes,
                                input_img,
                                instances,
                                process_id,
                                sd_augmented=aug_model,
                                file_id=file_id,
                            )
                            if record is not None:
                                records.append(record)
                    num_inpainted += 1

                    self.save_records_to_disk(records, process_id)
        metadata = {
            "reason_discarded": "detector",
            "count": total_samples,
            "inpainted": num_inpainted,
            "object_instance_to_bbox": object_instance_to_bbox,
        }
        return object_instance_to_bbox, num_inpainted, metadata

    def validate_with_detector(self, img, bboxes, instance_ids, out_dir):
        is_inpainted = []
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        object_instance_to_bbox, _ = self.detect_objects(img, "")

        if len(object_instance_to_bbox.keys()) == 0:
            is_inpainted = [True for i in range(len(instance_ids))]
            return is_inpainted

        for bbox_coords, instance_id in zip(bboxes, instance_ids):
            category_id = instance_id.split("_")[0]
            instance_inpainted = True
            for instance_id, (
                bbox_center,
                bbox,
            ) in object_instance_to_bbox.items():
                if category_id not in instance_id:
                    continue
                if BBoxUtils.iou_2d(bbox_coords, bbox) > 0.9:
                    instance_inpainted = False
                    break
            is_inpainted.append(instance_inpainted)
        return is_inpainted

    def sample_to_record(
        self,
        original_img,
        img_impainted,
        masks,
        bboxes,
        input_img_path,
        instance_ids,
        process_id,
        sd_augmented: str = "",
        file_id: int = 0,
    ) -> Optional[Dict[str, Any]]:
        for annotation_id, mask in enumerate(masks):
            if np.sum(mask) / np.prod(mask.shape) > 0.08:
                return None

        record_id = random_id()
        random_int = "{0:06d}".format(random.randint(0, 1e5))
        r_uid = "{}{}{}".format(self.uuid, process_id, self.record_id)

        # Save input image
        image_path = os.path.join(
            self.output_dir,
            "images/record_{}_{}_{}.png".format(record_id, random_int, r_uid),
        )
        save_array_to_img(img_impainted, image_path)

        orig_img = os.path.join(
            self.output_dir,
            "original/record_{}_{}_{}.png".format(record_id, random_int, r_uid),
        )
        save_array_to_img(original_img, orig_img)

        record = {
            "id": record_id,
            "original_img": input_img_path,
            "img_path": image_path,
            "width": img_impainted.shape[1],
            "height": img_impainted.shape[0],
            "annotations": [],
            "sd_augmented": sd_augmented,
            "file_id": file_id,
        }

        annotations = []
        for annotation_id, (mask, instance_id) in enumerate(
            zip(masks, instance_ids)
        ):
            object_category = instance_id.split("_")[0]
            bbox = bboxes[annotation_id]
            rle_mask = binary_mask_to_rle(mask)

            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "segmentation": rle_mask,
                    "annotation_path": "",
                    "object_category": object_category,
                    "bbox": [int(i) for i in bbox],
                }
            )

        record["annotations"] = annotations
        self.record_id += 1
        return record

    def save_records_to_disk(self, records, process_id: int = 0):
        write_json(
            records,
            os.path.join(
                self.output_dir,
                "{}_{}_{}_records.json".format(
                    self.split, self.uuid, process_id
                ),
            ),
        )

    def batch_inpaint(self, args):
        total_time = 0
        records = []

        files, process_id = args

        output_path = os.path.join(
            self.output_dir,
            "{}_{}_{}_records.json".format(self.split, self.uuid, process_id),
        )
        if os.path.exists(output_path):
            records.extend(load_json(output_path))
            self.record_id = len(records) + 1

            max_file_idx = max([int(r["file_id"]) for r in records])
            count_files = len(files)
            files = files[max_file_idx + 1 :]
            print("Resuming job from file id: {}".format(max_file_idx))
            print(
                "Total files processed: {}/{}, Pending: {}".format(
                    max_file_idx, count_files, len(files)
                )
            )
            print("Total existing records: {}".format(len(records)))

        total = len(files)
        print("Files: {}".format(total))
        files = sorted(files)

        failure_meta_output_path = os.path.join(
            self.output_dir,
            "{}_{}_{}_failure_meta.json".format(
                self.split, self.uuid, process_id
            ),
        )
        failure_metadata = []
        for idx, file in tqdm(enumerate(files)):
            start_time = time.time()
            print("[Progress: {}/{}] File: {}".format(idx, total, file))
            instances, num_inpainted, metadata = self.inpaint(
                file, records, process_id, file_id=idx
            )
            end_time = time.time()
            failure_metadata.append(metadata)
            if len(instances.keys()) > 0 and num_inpainted > 0:
                print(
                    "Avg. inpainting time: {}, Total time: {}, Instances: {}".format(
                        (end_time - start_time) / len(instances),
                        end_time - start_time,
                        len(instances),
                    )
                )
            else:
                print("Skipping object inpainting.")
            total_time += end_time - start_time
            write_json(failure_metadata, failure_meta_output_path)
        print(
            "Average inpainting time: {}, Total time: {}".format(
                total_time / len(files), total_time
            )
        )


def setup_args(parser):
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to a directory of imgs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        dest="visualize",
        help="Boolean flag to visualize results.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = glob.glob(os.path.join(args.input_path, "*open*"))
    print("Files: {}".format(len(files)))

    inpaint_anything = SemanticPlacementGenerator(
        visualize=args.visualize, output_dir=args.output_dir
    )
    inpaint_anything.batch_inpaint((files, 0))
