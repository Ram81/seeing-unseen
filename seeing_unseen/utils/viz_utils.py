import os
import random
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import tqdm
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
