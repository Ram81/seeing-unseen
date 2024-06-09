import itertools
from argparse import Namespace
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve

from seeing_unseen.utils import rotation as ru

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """Return a mask of all valid depth pixels."""
    return np.bitwise_and(
        depth != MIN_DEPTH_REPLACEMENT_VALUE,
        depth != MAX_DEPTH_REPLACEMENT_VALUE,
    )


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix = {"xc": xc, "zc": zc, "f": f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(
        np.arange(Y.shape[-1]), np.arange(Y.shape[-2] - 1, -1, -1)
    )
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (
        (x[::scale, ::scale] - camera_matrix.xc)
        * Y[::scale, ::scale]
        / camera_matrix.f
    )
    Z = (
        (z[::scale, ::scale] - camera_matrix.zc)
        * Y[::scale, ::scale]
        / camera_matrix.f
    )
    XYZ = np.concatenate(
        (
            X[..., np.newaxis],
            Y[::scale, ::scale][..., np.newaxis],
            Z[..., np.newaxis],
        ),
        axis=X.ndim,
    )
    return XYZ


def depth_to_surface_normals(depth, surfnorm_scalar=256):
    SURFNORM_KERNEL = torch.from_numpy(
        np.array(
            [
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )
    )[:, np.newaxis, ...].to(dtype=torch.float32, device=depth.device)
    with torch.no_grad():
        surface_normals = F.conv2d(depth, SURFNORM_KERNEL, padding=1)
        surface_normals[:, 2, ...] = 1
        surface_normals = surface_normals / surface_normals.norm(
            dim=1, keepdim=True
        )
    return surface_normals


def depth_to_surface_normals_np(depth, pixel_size=0.1, camera_matrix=None):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = convolve(depth, sobel_x)
    gradient_y = convolve(depth, sobel_y)

    # gradient_y, gradient_x = np.gradient(depth)

    du_dx = camera_matrix.f / depth  # x is xyz of camera coordinate
    dv_dy = camera_matrix.f / depth

    dz_dx = gradient_x * du_dx
    dz_dy = gradient_y * dv_dy

    # Compute surface normals
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(depth)), axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def upward_facing_surface_mask(surface_normal, threshold=0.9):
    upward_facing_points = (
        F.cosine_similarity(
            torch.tensor(surface_normal),
            torch.tensor([0, 1, 0]).view(1, 1, 3),
            dim=-1,
        )
        > threshold
    )
    return np.expand_dims(upward_facing_points.numpy(), axis=-1)


def numpy_to_pcd(
    xyz: np.ndarray, rgb: np.ndarray = None
) -> o3d.geometry.PointCloud:
    """Create an open3d pointcloud from a single xyz/rgb pair"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> (np.ndarray, np.ndarray):
    """Convert an open3d point cloud into xyz, rgb numpy arrays and return them."""
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def show_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray = None,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
):
    """Shows the point-cloud described by np.ndarrays xyz & rgb.
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    size controls scale of coordinate frame's size
    """
    pcd = numpy_to_pcd(xyz, rgb)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.io.write_point_cloud(save, pcd)
    # show_pcd(pcd, orig=orig, R=R, save=save, grasps=grasps, size=size)


def show_pcd(
    pcd: o3d.geometry.PointCloud,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
):
    """Shows the point-cloud described by open3d.geometry.PointCloud pcd
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    """
    geoms = create_visualization_geometries(
        pcd=pcd, orig=orig, R=R, grasps=grasps, size=size
    )
    o3d.visualization.draw_geometries(geoms)

    if save is not None:
        save_geometries_as_image(geoms, output_path=save)


def save_geometries_as_image(
    geoms: list,
    camera_extrinsic: Optional[np.ndarray] = None,
    look_at_point: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    zoom: Optional[float] = None,
    point_size: Optional[float] = None,
    near_clipping: Optional[float] = None,
    far_clipping: Optional[float] = None,
    live_visualization: bool = False,
):
    """
    Helper function to allow manipulation of the camera to get a better image of the point cloud.
    The live_visualization flag can help debug issues, by also spawning an interactable window.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for geom in geoms:
        vis.add_geometry(geom)
        vis.update_geometry(geom)

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    if camera_extrinsic is not None:
        # The extrinsic seems to have a different convention - switch from our camera to open3d's version
        camera_extrinsic_o3d = camera_extrinsic.copy()
        camera_extrinsic_o3d[:3, :3] = np.matmul(
            camera_extrinsic_o3d[:3, :3],
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        )
        camera_extrinsic_o3d[:, 3] = np.matmul(
            camera_extrinsic_o3d[:, 3],
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        )

        camera_params.extrinsic = camera_extrinsic_o3d
        view_control.convert_from_pinhole_camera_parameters(camera_params)

    if look_at_point is not None:
        view_control.set_lookat(look_at_point)

    if zoom is not None:
        view_control.set_zoom(zoom)

    if near_clipping is not None:
        view_control.set_constant_z_near(near_clipping)

    if far_clipping is not None:
        view_control.set_constant_z_far(far_clipping)

    render_options = vis.get_render_option()

    if point_size is not None:
        render_options.point_size = point_size

    if live_visualization:
        vis.run()
    print("save almost")

    vis.poll_events()
    vis.update_renderer()
    print("save start")
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()
    print("save done")


def create_visualization_geometries(
    pcd: Optional[o3d.geometry.PointCloud] = None,
    xyz: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    orig: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    size: Optional[float] = 1.0,
    arrow_pos: Optional[np.ndarray] = None,
    arrow_size: Optional[float] = 1.0,
    arrow_R: Optional[np.ndarray] = None,
    arrow_color: Optional[np.ndarray] = None,
    sphere_pos: Optional[np.ndarray] = None,
    sphere_size: Optional[float] = 1.0,
    sphere_color: Optional[np.ndarray] = None,
    grasps: list = None,
):
    """
    Creates the open3d geometries for a point cloud (one of xyz or pcd must be specified), as well as, optionally, some
    helpful indicators for points of interest -- an origin (orig), an arrow (including direction), a sphere, and grasp
    indicators.
    """
    assert (pcd is not None) != (
        xyz is not None
    ), "One of pcd or xyz must be specified"

    if xyz is not None:
        xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        if np.any(rgb > 1):
            print("WARNING: rgb values too high! Normalizing...")
            rgb = rgb / np.max(rgb)

    if pcd is None:
        pcd = numpy_to_pcd(xyz, rgb)

    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            origin=orig, size=size
        )
        if R is not None:
            coords = coords.rotate(R, orig)
        geoms.append(coords)

    if arrow_pos is not None:
        arrow = o3d.geometry.TriangleMesh.create_arrow()
        arrow = arrow.scale(
            arrow_size,
            center=np.zeros(
                3,
            ),
        )

        if arrow_color is not None:
            arrow = arrow.paint_uniform_color(arrow_color)

        if arrow_R is not None:
            arrow = arrow.rotate(arrow_R, center=(0, 0, 0))

        arrow = arrow.translate(arrow_pos)
        geoms.append(arrow)

    if sphere_pos is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)

        if sphere_color is not None:
            sphere = sphere.paint_uniform_color(sphere_color)

        sphere = sphere.translate(sphere_pos)
        geoms.append(sphere)

    if grasps is not None:
        for grasp in grasps:
            coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=grasp[:3, 3]
            )
            coords = coords.rotate(grasp[:3, :3])
            geoms.append(coords)

    return geoms


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(
        torch.arange(Y_t.shape[-1], device=device),
        torch.arange(Y_t.shape[-2] - 1, -1, -1, device=device),
    )
    grid_x = grid_x.transpose(1, 0)
    grid_z = grid_z.transpose(1, 0)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (
        (grid_x[:, ::scale, ::scale] - camera_matrix.xc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )
    Z_t = (
        (grid_z[:, ::scale, ::scale] - camera_matrix.zc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )

    XYZ = torch.stack((X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ


def transform_camera_view_t(
    XYZ, sensor_height, camera_elevation_degree, device
):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree)
    )
    XYZ = torch.matmul(
        XYZ.reshape(-1, 3),
        torch.from_numpy(R).float().transpose(1, 0).to(device),
    ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ
