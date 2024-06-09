import glob
import gzip
import json
import os
import os.path as osp
import pickle
import random
import string
import threading
from collections import defaultdict
from itertools import groupby
from typing import List, Optional, Union

import cv2
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

lock = threading.Lock()


def load_json(path):
    file = open(path, "r")
    return json.loads(file.read())


def load_gzip(path):
    file = gzip.open(path, "r")
    return json.loads(file.read())


def load_parquet(path):
    table = pq.ParquetFile(path)
    return table


def write_gzip(data, output_path):
    file = gzip.open(output_path, "w")
    for i, l in enumerate(data):
        file.write((json.dumps(l) + "\n").encode("utf-8"))  # type: ignore


def write_json(data, output_path):
    file = open(output_path, "w")
    file.write(json.dumps(data))


def load_gz_dataset(path, dataset_size):
    file = gzip.open(path, "r")
    print(type(file))
    for l in file:
        print(type(l), "first element")
        break
    data = [line for line in tqdm(file, total=dataset_size)]
    return data


def load_gzip(path):
    with gzip.open(path, "r") as file:
        data = json.loads(file.read().decode("utf-8"))
    return data


def write_gzip(data, path):
    if not isinstance(data, str):
        data = json.dumps(data)
    with gzip.open(path, "w") as f:
        f.write((data + "\n").encode("utf-8"))


def save_image(img, file_name):
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    im = Image.fromarray(img)
    im.save(file_name)


def load_image(file_name):
    return Image.open(file_name).convert("RGB")


def load_image_fast(file_name):
    return iio.imread(file_name)


def save_pickle(data, path):
    file = open(path, "wb")
    data = pickle.dump(data, file)


def load_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    return data


def binary_mask_to_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask)
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(
        groupby(binary_mask.ravel(order="F"))
    ):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    encoded_rle = encode_rle(rle)
    encoded_rle["counts"] = encoded_rle["counts"].decode("utf-8")
    return encoded_rle


def encode_rle(binary_mask):
    rle_mask = mask_utils.frPyObjects(
        binary_mask, binary_mask.get("size")[0], binary_mask.get("size")[1]
    )
    return rle_mask


def random_id(N=8):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )


def decode_rle_mask(rle_mask):
    # Decode the RLE mask
    mask_array = mask_utils.decode(rle_mask)
    # Convert the mask array to an image
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]
    image = Image.fromarray(mask_array.astype(np.uint8) * 255)
    return image


def filter_mask_from_bbox(bbox, height, width, offset=[5.0, 5.0, 5.0, 5.0]):
    mask = np.zeros((height, width))
    bbox_int = [int(x) for x in bbox]
    bbox_offset = (
        max(bbox_int[0] - 5, 0),
        max(bbox_int[1] - 5, 0),
        min(bbox_int[2] + 5, width),
        min(bbox_int[3] + 5, height),
    )
    mask[bbox_offset[0] : bbox_offset[2], bbox_offset[1] : bbox_offset[3]] = 1
    return mask


def count_samples(input_path):
    files = glob.glob(osp.join(input_path, "*json"))
    count = 0
    for file in files:
        samples = load_json(file)
        count += len(samples)
    print("Total samples: ", count)
    return count


def convert_xyz_to_torch_tensor(list_of_xyz):
    if type(list_of_xyz) == dict:
        return torch.Tensor(
            [list_of_xyz["x"], list_of_xyz["y"], list_of_xyz["z"]]
        )
    return torch.Tensor([[x["x"], x["y"], x["z"]] for x in list_of_xyz])


def convert_torch_tensor_to_xyz(tensor_xyz):
    tensor_xyz = tensor_xyz.numpy()
    return [dict(x=x[0], y=x[1], z=x[2]) for x in tensor_xyz]


def get_possible_spawn_locations_close_to_agent(controller, receptacle_id):
    locations = convert_xyz_to_torch_tensor(
        controller.get_locations_on_receptacle(receptacle_id)
    )
    current_agent_position = convert_xyz_to_torch_tensor(
        controller.get_current_agent_position()
    )
    distances = (locations - current_agent_position).norm(dim=-1)
    closest_indices = torch.topk(
        distances, k=min(20, len(distances)), largest=False
    ).indices
    tensor_xyz = locations[closest_indices]
    return convert_torch_tensor_to_xyz(tensor_xyz)


def get_config(
    config_path: str,
    overrides: Optional[list] = None,
) -> DictConfig:
    abs_path = osp.abspath(config_path)
    with lock, initialize_config_dir(
        version_base=None,
        config_dir=osp.dirname(abs_path),
    ):
        cfg = compose(
            config_name=osp.basename(abs_path),
            overrides=overrides if overrides is not None else [],
        )
    return cfg


def count_episodes(path):
    files = glob.glob(os.path.join(path, "*.json.gz"))
    count = 0
    categories = defaultdict(int)
    for f in tqdm(files):
        dataset = load_gzip(f)
        for episode in dataset["episodes"]:
            categories[episode["object_category"]] += 1
        count += len(dataset["episodes"])
    print("Total episodes: {}".format(count))
    print("Categories: {}".format(categories))
    print("Total categories: {}".format(len(categories)))
    return count, categories


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1
    )
    return mask


def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask, np.ones((dilate_factor, dilate_factor), np.uint8), iterations=1
    )
    return mask


def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: "red", 1: "green"}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker="*",
            s=size,
            edgecolor="white",
            linewidth=1.25,
        )


def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            keep_looping = False

    cv2.setMouseCallback("image", mouse_callback)

    while keep_looping:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return last_point
