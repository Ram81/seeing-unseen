from typing import Dict, List, Tuple

import numpy as np
from shapely import affinity, geometry


class BBoxUtils:
    def __init__(self):
        pass

    @staticmethod
    def compute_bounding_box_properties(corners: List[Dict]):
        # Compute width, height, and length
        max_x = max([c[0] for c in corners])
        min_x = min([c[0] for c in corners])

        max_y = max([c[1] for c in corners])
        min_y = min([c[1] for c in corners])

        max_z = max([c[2] for c in corners])
        min_z = min([c[2] for c in corners])

        width = abs(max_x - min_x)
        height = abs(max_y - min_y)
        length = abs(max_z - min_z)

        # Compute volume
        volume = width * height * length

        return volume, width, height, length

    @staticmethod
    def box_properties(corners: List[Dict]):
        # Compute width, height, and length
        max_x = max([c[0] for c in corners])
        min_x = min([c[0] for c in corners])

        max_y = max([c[1] for c in corners])
        min_y = min([c[1] for c in corners])

        width = max_x - min_x
        height = max_y - min_y
        return width, height

    @staticmethod
    def is_point_inside_bounding_box(
        point: Dict, corners: List[List], check_3d: bool = False
    ):
        # Compute width, height, and length
        max_x = max([c[0] for c in corners])
        min_x = min([c[0] for c in corners])

        max_y = max([c[1] for c in corners])
        min_y = min([c[1] for c in corners])

        max_z = max([c[2] for c in corners])
        min_z = min([c[2] for c in corners])

        if check_3d:
            if (
                min_x <= point["x"]
                and point["x"] <= max_x
                and min_y <= point["y"]
                and point["y"] <= max_y
                and min_z <= point["z"]
                and point["z"] <= max_z
            ):
                return True
        else:
            if (
                min_x <= point["x"]
                and point["x"] <= max_x
                and min_z <= point["z"]
                and point["z"] <= max_z
            ):
                return True
        return False

    @staticmethod
    def bbox_from_3d(corners: List[List]):
        max_x = max([c[0] for c in corners])
        min_x = min([c[0] for c in corners])

        max_z = max([c[2] for c in corners])
        min_z = min([c[2] for c in corners])

        return min_x, max_x, min_z, max_z

    @staticmethod
    def project_3d_bbox_to_2d(corners: List[List], offset: List = [0, 0]):
        max_x = max([c[0] for c in corners])
        min_x = min([c[0] for c in corners])

        max_z = max([c[2] for c in corners])
        min_z = min([c[2] for c in corners])

        delta_x = abs(max_x - min_x) * offset[0]
        delta_z = abs(max_z - min_z) * offset[0]
        max_x += delta_x
        min_x -= delta_x

        min_z -= delta_z
        max_z += delta_z

        bbox_corners = [
            [min_x, min_z],
            [min_x, max_z],
            [max_x, max_z],
            [max_x, min_z],
        ]
        return bbox_corners

    @staticmethod
    def calculate_overlap(
        box1_corners: List, target_box_corners: List
    ) -> float:
        """
        Calculates the area of overlap between two 3D bounding boxes.

        Arguments:
        box1_corners, box2_corners -- Lists or arrays containing the coordinates of the eight corner points of the two 3D bounding boxes.
                                    Each corner point is represented as [x, y, z].

        Returns:
        overlap_area -- The area of overlap between the two bounding boxes. Returns 0 if there is no overlap.
        """
        # Create polygons from the corner points of the bounding boxes
        polygon1 = geometry.Polygon(box1_corners)
        polygon2 = geometry.Polygon(target_box_corners)

        # Calculate the intersection of the polygons
        intersection = polygon1.intersection(polygon2)

        # If the intersection is empty or not a polygon, there is no overlap
        if intersection.is_empty:
            return 0

        # Calculate the overlap area
        overlap_area = intersection.area

        return overlap_area / polygon2.area

    @staticmethod
    def exterior_coords_from_box(bbox) -> np.ndarray:
        xx, yy = bbox.exterior.coords.xy
        bbox = [[x, y] for x, y in zip(xx.tolist(), yy.tolist())]
        return np.array(bbox)

    @staticmethod
    def offset_bbox(corners, offset):
        new_corners = []
        for point in corners:
            for idx in range(len(offset)):
                point[idx] += offset[idx]
            new_corners.append(point)
        return new_corners

    @staticmethod
    def bbox_overlap_2d(
        box1: List[float], box2: List[float], offset: List[float] = [0, 0, 0, 0]
    ) -> bool:
        if (box1[2] + offset[2]) < box2[0] or (box1[0] - offset[0]) > box2[2]:
            return False

        if (box1[3] + offset[3]) < box2[1] or (box1[1] - offset[1]) > box2[3]:
            return False
        return True

    @staticmethod
    def iou_2d(
        box1: List[float], box2: List[float]
    ) -> float:
        x_left = max(box1[0], box1[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    @staticmethod
    def bbox_merge_2d(box1: List[float], box2: List[float]) -> np.ndarray:
        bbox_merged = [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]
        return np.array(bbox_merged)

    @staticmethod
    def bbox_difference_from_bounds(
        box1: List[float], upper_bounds: List[float]
    ) -> List[float]:
        return [i - j for i, j in zip(box1, upper_bounds)]
