import math

import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from seeing_unseen.core.registry import registry
from seeing_unseen.models.base import SPModel


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


@registry.register_affordance_model(name="llava")
class LLaVaVLM(SPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prompt = """Localize the area in image as a bounding box in normalized coordinates to place the {}."""

        self.conv_mode = "llava_v1"
        self.temperature = 0.2
        self.top_p = None
        self.num_beams = 1
        self.model_path = "data/llava_weights/llava-v1.5-7b"
        self.model_base = None
        self.num_chunks = 1
        self.chunk_idx = 1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._init_llava()

    def _init_llava(self):
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len

    def caption_to_mask(self, caption: str, image: torch.Tensor):
        _, width, height = image.shape
        target_mask = torch.zeros(image.shape[1:])

        bbox_str = caption[caption.find("[") + 1 : caption.find("]")]
        bbox = [float(b) for b in bbox_str.split(",")]
        no_mask = True

        if len(bbox) != 4:
            return target_mask, no_mask

        bbox = [
            int(bbox[0] * width),
            int(bbox[1] * height),
            int(bbox[2] * width),
            int(bbox[3] * height),
        ]

        target_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        no_mask = False
        # if np.sum(target_mask) / np.prod(target_mask.shape) > 0.40:
        #     target_mask *= 0
        #     no_mask = True
        return target_mask, no_mask

    def forward(self, **kwargs):
        observations = kwargs["batch"]
        questions = [
            self.prompt.format(category)
            for category in observations["target_category"]
        ]

        predictions = []
        no_masks = 0
        for image, question in zip(observations["image"], questions):
            target_mask = torch.zeros(image.shape[1:])

            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + question
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )

            image_tensor = self.image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]

            stop_str = (
                conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            )
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            target_mask, no_mask = self.caption_to_mask(outputs, image)
            no_masks += no_mask
            predictions.append(target_mask.to(image.device))

        predictions = torch.stack(predictions)
        outputs = {"affordance": predictions}
        return outputs
