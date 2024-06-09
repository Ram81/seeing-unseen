import requests
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


class OwlViT:
    def __init__(self, device) -> None:
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).to(self.device)

    def predict(self, batch):
        inputs = self.processor(
            images=batch["image"],
            text=batch["target_category"],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor(
            [batch["image"][i].shape[1:3] for i in range(len(batch["image"]))]
        )
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes.to(self.device),
            threshold=0.1,
        )
        return results
