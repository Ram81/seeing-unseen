import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import imageio
import numpy as np
import tqdm
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import SummaryWriter
from allenact.utils.viz_utils import AbstractViz, AgentViewViz
from PIL import Image


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 15,
    quality: Optional[float] = 10,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def concatenate_frames(images: List[np.ndarray]) -> np.ndarray:
    imgs = []
    for img in images:
        img = Image.fromarray(img).convert("RGB")
        img = img.resize((300, 300), Image.ANTIALIAS)
        imgs.append(img)

    widths, heights = zip(*(i.size for i in imgs))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return np.array(new_im)


def visualize_bounding_boxes(image, bounding_boxes, categories):
    """
    Visualizes bounding boxes on an image with different colors for different categories.

    Args:
        image_path (str): The path to the image file.
        bounding_boxes (list): List of bounding boxes in the format (x_min, y_min, x_max, y_max).
        categories (list): List of category labels corresponding to each bounding box.
    """
    # Convert the image to RGB
    image_rgb = np.array(image)

    # Draw bounding boxes on the image
    for (center, bbox), category in zip(bounding_boxes, categories):
        bbox = [int(x) for x in bbox]
        x_min, y_min, x_max, y_max = bbox
        color = np.random.randint(
            0, 256, 3
        ).tolist()  # Random color for each category
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            image_rgb,
            category,
            (x_min, y_min - 10 - random.choice(list(range(10)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        cv2.circle(image_rgb, (int(center[0]), int(center[1])), 3, color, -1)

    return image_rgb


def overlay_semantic_mask(image, mask, alpha=0.5):
    """
    Overlay a semantic mask on an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The semantic mask.
        alpha (float): The alpha value for blending the mask (0 to 1).

    Returns:
        numpy.ndarray: The image with the overlayed mask.
    """
    # Convert the mask to RGB
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Apply the mask on the image
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

    return overlay


def overlay_heatmap(image, mask, alpha=0.5):
    """
    Overlay a heatmap on an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The semantic mask.
        alpha (float): The alpha value for blending the mask (0 to 1).

    Returns:
        numpy.ndarray: The image with the overlayed mask.
    """
    # Convert the mask to RGB
    mask_heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Apply the mask on the image
    overlay = cv2.addWeighted(image, 1 - alpha, mask_heatmap, alpha, 0)

    return overlay


def overlay_heatmap_with_annotations(image, mask, alpha=0.5, font_size=0.25):
    """
    Overlay a heatmap on an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The semantic mask.
        alpha (float): The alpha value for blending the mask (0 to 1).

    Returns:
        numpy.ndarray: The image with the overlayed mask.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    print(np.sum(mask == 0), np.sum(mask))

    colors = [
        (100, 196, 254),
        (254, 184, 100),
        (100, 254, 133),
        (196, 100, 254),
        (254, 100, 149),
        (254, 105, 100),
    ]

    labels_sorted = [
        ((labels == i).sum() / np.prod(labels.shape), i)
        for i in range(1, num_labels)
    ]
    labels_sorted = sorted(labels_sorted, key=lambda x: x[0], reverse=True)

    image_with_masks = []
    # for i in range(1, num_labels):
    for area, label in labels_sorted:
        # Convert the mask to RGB
        mask_heatmap = cv2.applyColorMap(
            (mask * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        color = colors[0]  # [(i - 1) % len(colors)]
        current_mask = (labels == label).astype(np.int32)

        area = current_mask.sum() / np.prod(current_mask.shape)

        print(area, current_mask.shape)

        if area <= 0.005:
            continue

        mask_heatmap[:, :, 0] = ((current_mask > 0) * color[0]).astype(
            np.uint8
        )  # 100
        mask_heatmap[:, :, 1] = ((current_mask > 0) * color[1]).astype(np.uint8)
        mask_heatmap[:, :, 2] = ((current_mask > 0) * color[2]).astype(np.uint8)

        # Apply the mask on the image
        # overlay = cv2.addWeighted(image, 1 - alpha, mask_heatmap, alpha, 0)
        affordance_weight_mask = (current_mask > 0) * 0.7
        image_weight_mask = (current_mask > 0) * 0.3 + (current_mask == 0) * 1
        overlay = (
            np.expand_dims(affordance_weight_mask, -1) * mask_heatmap
            + np.expand_dims(image_weight_mask, -1) * image
        ).astype(np.uint8)

        center_x, center_y = int(centroids[label][0]), int(centroids[label][1])
        w, h = 10, 10

        cv2.rectangle(
            overlay,
            (center_x - w // 2, center_y - h // 2),
            (center_x + w // 2, center_y + h // 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            overlay,
            str(1),
            (center_x - 2, center_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        image_with_masks.append(overlay)

    # for i in range(1, num_labels):
    #     center_x, center_y = int(centroids[i][0]), int(centroids[i][1])

    #     # Get the label index
    #     label_index = i

    #     # Draw the label index at the center of the component
    #     w, h = 10, 10
    #     cv2.rectangle(
    #         overlay,
    #         (center_x - w // 2, center_y - h // 2),
    #         (center_x + w // 2, center_y + h // 2),
    #         (0, 0, 0),
    #         -1,
    #     )
    #     cv2.putText(
    #         overlay,
    #         str(label_index),
    #         (center_x - 2, center_y + 2),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         font_size,
    #         (255, 255, 255),
    #         1,
    #         cv2.LINE_AA,
    #     )

    return image_with_masks


def smooth_mask(mask, kernel=None, num_iterations=3):
    """Dilate and then erode.

    Arguments:
        mask: the mask to clean up

    Returns:
        mask: the dilated mask
        mask2: dilated, then eroded mask
    """
    if kernel is None:
        kernel = np.ones((5, 5))
    mask = mask.astype(np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=num_iterations)
    # second step
    mask2 = mask
    mask2 = cv2.erode(mask2, kernel, iterations=num_iterations)
    mask2 = np.bitwise_and(mask, mask2)
    return mask1, mask2


class DiskAgentViewViz(AgentViewViz):
    def __init__(
        self,
        label: str = "agent_view",
        output_path: str = "",
        max_episodes: int = 500,
        **kwargs,
    ):
        super().__init__(label, **kwargs)
        self.output_path = output_path
        self.max_episodes = max_episodes

    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        if render is None:
            return
        keys = list(render.keys())
        print("Render : {}".format(keys))

        datum_id = self._source_to_str(
            self.vector_task_sources[0], is_vector_task=True
        )

        print("task output: {}".format(task_outputs[0]))
        sampled_episodes = task_outputs
        for i, episode in enumerate(sampled_episodes):
            episode_id = episode["task_info"]["id"]
            print(episode.keys())
            print(episode["task_info"].keys())
            uuid = "episodeId={}-success={}".format(
                episode_id,
                int(episode["success"]),
            )
            overlay_label = uuid.replace("-", "\n")
            # assert episode_id in render
            if episode_id not in render:
                get_logger().warning(
                    "skipping viz for missing episode {}".format(episode_id)
                )
                continue
            images = [
                self._overlay_label(step[datum_id], overlay_label)
                for step in render[episode_id]
            ]
            images_to_video(images, self.output_path, uuid)


class DiskTopDownAgentViewViz(AbstractViz):
    def __init__(
        self,
        label: str = "agent_top_down_view",
        output_path: str = "",
        max_clip_length: int = 100,  # control memory used when converting groups of images into clips
        max_video_length: int = -1,  # no limit, if > 0, limit the maximum video length (discard last frames)
        max_episodes: int = 500,
        vector_task_source: List[Tuple[str, Dict[str, Any]]] = [
            (
                "render_custom",
                {"mode": "rgb_right"},
            ),
            (
                "render",
                {"mode": "raw_rgb_list"},
            ),
        ],
        episode_ids: Optional[Sequence[Union[Sequence[str], str]]] = None,
        fps: int = 4,
        max_render_size: int = 500,
        **other_base_kwargs,
    ):
        self.output_path = output_path
        self.max_episodes = max_episodes

        super().__init__(
            label,
            vector_task_sources=vector_task_source,
            **other_base_kwargs,
        )
        self.max_clip_length = max_clip_length
        self.max_video_length = max_video_length
        self.fps = fps
        self.max_render_size = max_render_size

        self.episode_ids = (
            (
                list(episode_ids)
                if not isinstance(episode_ids[0], str)
                else [list(cast(List[str], episode_ids))]
            )
            if episode_ids is not None
            else None
        )

    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        if render is None:
            return
        keys = list(render.keys())

        datum_id = self._source_to_str(
            self.vector_task_sources[0], is_vector_task=True
        )
        rgb_datum_id = self._source_to_str(
            self.vector_task_sources[1], is_vector_task=True
        )

        sampled_episodes = random.sample(
            task_outputs, min(self.max_episodes, len(task_outputs))
        )  # self.max_episodes)
        for i, episode in enumerate(sampled_episodes):
            episode_id = episode["task_info"]["id"]
            uuid = "episodeId={}-success={}".format(
                episode_id,
                int(episode["success"]),
            )
            overlay_label = uuid.split("-")
            # assert episode_id in render
            if episode_id not in render:
                get_logger().warning(
                    "skipping viz for missing episode {}".format(episode_id)
                )
                continue
            all_images = []

            for step in render[episode_id]:
                rgb_img = self._overlay_label(step[rgb_datum_id], overlay_label)
                rgb_img = Image.fromarray(rgb_img)
                rgb_img = rgb_img.resize((500, 500), Image.ANTIALIAS)

                img = Image.fromarray(step[datum_id])
                img = img.resize((500, 500), Image.ANTIALIAS)

                images = [rgb_img, img]
                widths, heights = zip(*(i.size for i in images))

                total_width = sum(widths)
                max_height = max(heights)

                new_im = Image.new("RGB", (total_width, max_height))

                x_offset = 0
                for im in images:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                all_images.append(np.array(new_im))

            images_to_video(all_images, self.output_path, uuid)

    @staticmethod
    def _overlay_label(
        img,
        text,
        pos=(0, 0),
        bg_color=(255, 255, 255),
        fg_color=(255, 255, 255),
        scale=0.4,
        thickness=1,
        margin=5,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        txt_size = cv2.getTextSize(text[0], font_face, scale, thickness)

        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1]

        pos = (pos[0], pos[1] + txt_size[0][1] + margin)

        for txt in text:
            cv2.putText(
                img=img,
                text=txt,
                org=pos,
                fontFace=font_face,
                fontScale=scale,
                color=fg_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            pos = (pos[0], pos[1] + txt_size[0][1] + margin)
        return img


def overlay_mask_with_gaussian_blur(affordance, img):
    if affordance.dtype != np.uint8:
        affordance = affordance.astype(np.uint8)

    affordance = ((1 - affordance) * 255).astype(np.uint8)

    semantic_obs_blurred = cv2.GaussianBlur(affordance, (57, 57), 11)
    semantic_obs_colored = cv2.applyColorMap(
        semantic_obs_blurred, cv2.COLORMAP_JET
    )

    affordance_weight_mask = (affordance == 0) * 0.7
    image_weight_mask = (affordance == 0) * 0.3 + (affordance > 0).astype(
        np.uint8
    )
    superimposed_affordance = (
        np.expand_dims(affordance_weight_mask, -1) * semantic_obs_colored
        + np.expand_dims(image_weight_mask, -1) * img
    ).astype(np.uint8)
    return superimposed_affordance
