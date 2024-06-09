from typing import List

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from seeing_unseen.core.registry import registry
from seeing_unseen.models.base import SPModel
from seeing_unseen.models.encoders.detic_perception import DeticPerception
from seeing_unseen.models.encoders.owl_vit import OwlViT
from seeing_unseen.utils.utils import load_json


@registry.register_affordance_model(name="llm_detect_owl_vit")
class LLMDetectOwlViT(SPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_receptacle_map = load_json(
            "data/metadata/object_receptacle_mapping.json"
        )
        self.device = torch.device("cuda:0")

        self.init_segmentation()
        self.init_sam()

    def init_segmentation(self):
        self.segmentation = OwlViT(device=self.device)

    def init_sam(
        self,
        sam_model_type: str = "vit_h",
        sam_ckpt: str = "data/pretrained_models/sam_vit_h_4b8939.pth",
    ):
        sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

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

    def forward(self, **kwargs):
        observations = kwargs["batch"]

        receptacles = [
            self.obj_receptacle_map[target.replace(" ", "_")]
            for target in observations["target_category"]
        ]

        new_batch = {
            "image": [img for img in observations["image"]],
            "target_category": receptacles,
        }
        results = self.segmentation.predict(new_batch)
        predictions = []
        # Get segmentation mask
        for idx in range(observations["image"].shape[0]):
            category = new_batch["target_category"][idx]
            boxes, scores, labels = (
                results[idx]["boxes"],
                results[idx]["scores"],
                results[idx]["labels"],
            )
            target_mask = torch.zeros(observations["image"][idx].shape[1:])
            if len(boxes) > 0:
                max_confidence_score_idx = np.argsort(scores.cpu().numpy())[-1]
                box = [int(b) for b in boxes[max_confidence_score_idx].tolist()]
                bbox_center = (
                    boxes[max_confidence_score_idx][:2]
                    + boxes[max_confidence_score_idx][2:]
                ) / 2

                obs_image = observations["image"][idx]

                masks, scores, _ = self.predict_masks_with_sam(
                    (obs_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8
                    ),
                    [bbox_center.cpu().numpy()],
                    [1],
                )
                mask_idx = np.argmax(scores)

                if mask_idx is None:
                    continue
                mask = masks[mask_idx]

                label = category[labels[max_confidence_score_idx]]
                # target_mask[box[0] : box[2], box[1] : box[3]] = 1
                target_mask = (torch.from_numpy(mask) > 0).float()
            predictions.append(target_mask.to(observations["image"].device))

        predictions = torch.stack(predictions)

        output = {"affordance": predictions}
        return output


@registry.register_affordance_model(name="llm_detect_grounded_sam")
class LLMDetectGroundedSAM(SPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_receptacle_map = load_json(
            "data/metadata/object_receptacle_mapping.json"
        )
        self.device = torch.device("cuda:0")

        self.init_segmentation()
        self.init_sam()

    def init_sam(
        self,
        sam_model_type: str = "vit_h",
        sam_ckpt: str = "data/pretrained_models/sam_vit_h_4b8939.pth",
    ):
        sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def init_segmentation(self):
        self.vocabulary = ".,couch,bed,chair,shelves,chest_of_drawers,table,kitchen_counter,floor"
        self.vocabulary = self.vocabulary.replace("_", " ").split(",")
        self.segmentation = GroundedSAMPerception(
            custom_vocabulary=self.vocabulary,
            device=self.device,
        )

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

    def forward(self, **kwargs):
        observations = kwargs["batch"]

        predictions = []
        # Get segmentation mask
        for idx in range(observations["image"].shape[0]):
            obs_image = observations["image"][idx]
            results = self.segmentation.predict(
                (obs_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                ),
                draw_instance_predictions=True,
            )

            obj_instances = results["preds"]["instances"]
            target_mask = torch.zeros(obs_image.shape[1:])
            receptacles = self.obj_receptacle_map[
                observations["target_category"][idx].replace(" ", "_")
            ]

            max_mask_area = 0
            for category_id, bbox, score in zip(
                obj_instances.pred_classes,
                obj_instances.pred_boxes,
                obj_instances.scores,
            ):
                if category_id is None:
                    continue

                category_name = self.vocabulary[category_id].replace(" ", "_")
                if score < 0.5 or category_name not in receptacles:
                    continue

                bbox_center = (bbox[:2] + bbox[2:]) / 2
                masks, scores, _ = self.predict_masks_with_sam(
                    (obs_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8
                    ),
                    [bbox_center],
                    [1],
                )
                mask_idx = np.argmax(scores)

                if mask_idx is None:
                    continue
                mask = masks[mask_idx]
                mask_area = np.sum(mask) / np.prod(mask.shape)

                if mask_area > max_mask_area:
                    max_mask_area = mask_area
                    # target_mask = torch.logical_or(target_mask, mask > 0)
                    target_mask = (torch.from_numpy(mask) > 0).float()

            predictions.append(target_mask.to(obs_image.device))
        predictions = torch.stack(predictions).to(observations["image"].device)

        output = {"affordance": predictions}
        return output


@registry.register_affordance_model(name="llm_detect_detic")
class LLMDetect(SPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obj_receptacle_map = load_json(
            "data/metadata/object_receptacle_mapping.json"
        )
        self.init_segmentation()

    def init_segmentation(self):
        self.vocabulary = ".,couch,bed,chair,shelves,chest_of_drawers,table,kitchen_counter,floor"
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=self.vocabulary,
        )
        self.vocabulary = self.vocabulary.replace("_", " ").split(",")

    def forward(self, **kwargs):
        observations = kwargs["batch"]

        predictions = []
        # Get segmentation mask
        for idx in range(observations["image"].shape[0]):
            obs_image = observations["image"][idx]
            results = self.segmentation.predict(
                (obs_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                ),
                draw_instance_predictions=True,
            )

            obj_instances = results["preds"]["instances"]
            target_mask = torch.zeros(
                obs_image.shape[1:], device=obs_image.device
            )
            receptacles = self.obj_receptacle_map[
                observations["target_category"][idx].replace(" ", "_")
            ]

            cats = []
            max_mask_area = 0
            for category_id, bbox, mask in zip(
                obj_instances.pred_classes,
                obj_instances.pred_boxes,
                obj_instances.pred_masks,
            ):
                category_id = category_id.detach().cpu().item()
                category_name = self.vocabulary[category_id].replace(" ", "_")
                cats.append(category_name)
                mask_area = torch.sum(mask) / np.prod(mask.shape)

                if category_name in receptacles and mask_area > max_mask_area:
                    max_mask_area = mask_area
                    target_mask = (mask > 0).float()

            predictions.append(target_mask)
        predictions = torch.stack(predictions)

        output = {"affordance": predictions}
        return output
