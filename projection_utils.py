import torch
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from ood.detectron2_files.detectron2.utils.visualizer import Visualizer2


# George's code refined
def get3Dpoints(points2D, depth, intr, scale, Rw2c, Tw2c):
    """

    Parameters
    ----------
    points2D: ndarray
        Points in Image frame (x goes left to right, y goes top to bottom)
    depth
    intr
    scale
    Rw2c
    Tw2c

    Returns
    -------

    """
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    depth = depth / float(scale)
    Rc2w = np.linalg.inv(Rw2c)
    Tc2w = np.dot(-Rc2w, Tw2c)
    z = depth[points2D[:, 1], points2D[:, 0]]  # z is in numpy coordinate so reversed
    local3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)  # N, 3
    local3D[:, 0] = (points2D[:, 0] - cx) * z / fx
    local3D[:, 1] = (points2D[:, 1] - cy) * z / fy
    local3D[:, 2] = z
    points3D = np.dot(Rc2w, local3D.T) + Tc2w  # 3x3 . 3xN + 3x1
    return points3D.T  # N, 3


# George's code refined
def world2img(points3D, intr, Rw2c, Tw2c, im_dim):
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    local3D = np.dot(Rw2c, points3D.T) + Tw2c.reshape(3, 1)

    a = local3D[0, :] * fx / local3D[2, :] + cx
    b = local3D[1, :] * fy / local3D[2, :] + cy
    # q1 = a[:,np.newaxis] / local3D[2,:][:,np.newaxis] + cx
    # q2 = b[:,np.newaxis] / local3D[2,:][:,np.newaxis] + cy
    # q1 = a / local3D[2,:] + cx
    # q2 = b / local3D[2,:] + cy

    # x_proj = q1.reshape(q1.shape[0])
    # y_proj = q2.reshape(q2.shape[0])
    # x_proj = np.round(x_proj)
    # y_proj = np.round(y_proj)
    x_proj = np.round(a)
    y_proj = np.round(b)
    # keep only coordinates in the image frame
    inds_1 = np.where(x_proj >= 1)
    inds_2 = np.where(y_proj >= 1)
    inds_3 = np.where(x_proj < im_dim[1] - 1)
    inds_4 = np.where(y_proj < im_dim[0] - 1)
    idx = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
    x_proj = x_proj[idx]
    y_proj = y_proj[idx]
    x_proj = x_proj.astype(np.int)
    y_proj = y_proj.astype(np.int)
    points2D = np.zeros((x_proj.shape[0], 2), dtype=int)
    points2D[:, 0] = x_proj
    points2D[:, 1] = y_proj
    return points2D, idx


def c2w_from_w2c(Tw2c, Rw2c):
    """ Get camera to world coordinate transformations using
    world to camera transformations.

    Parameters
    ----------
    Tw2c: np.array (3,1)
        camera to world translation
    Rw2c:  np.array (3,3)
        camera to world rotation

    Returns
    -------
    Tc2w: np.array (3,1)
        world to camera translation
    Rc2w: np.array (3,3)
        world to camera translation rotation

    Camera to World Transform
    --------------------------
    Xw = Rw2c Xc + Tw2c
    --> (Rw2c)^(-1) Xw = Xc + (Rw2c)^(-1) Tw2c
    --> Xc = Rw2c^(-1) Xw  + [- Rw2c^(-1) Tw2c]

    Rc2w = Rw2c^(-1)
        = np.linalg.inv(Rw2c)
    tc2w = -Rw2c^(-1) . Tw2c = - Rc2w . Tw2c
        = -np.matmul(Rc2w, tw2c)
    """
    Rc2w = np.linalg.inv(Rw2c)
    Tc2w = np.dot(-Rc2w, Tw2c)

    return Tc2w, Rc2w


def world_coord_from_image(x, y, z, Rw2c, Tw2c, cx, cy, fx, fy):
    """

    Parameters
    ----------
    x: np.array (N,)
        x coord in Image frame (x goes left to right)
    y: np.array (N,)
        y coord in Image frame (y goes top to bottom)
    z: np.array (N,)
        Depth image already scaled by the scaling factor of scene.
        Reshaped to match x and y.
    Rw2c
    Tw2c
    cx
    cy
    fx
    fy

    Returns
    -------

    """
    x_flat_sub_cx = x - cx
    y_flat_sub_cy = y - cy

    Rc2w = np.linalg.inv(Rw2c)
    Tc2w = np.dot(-Rc2w, Tw2c)

    # camera 3d
    camera3d = np.array((x_flat_sub_cx * z / fx, y_flat_sub_cy * z / fy, z))  # 3, N
    # world 3d
    print(Rc2w.shape)
    print(Tc2w.shape)
    world3d = np.dot(Rc2w, camera3d) + Tc2w  # 3x3 . 3xN + 3x1

    return world3d  # 3 x N


def image_coord_from_world(world3d, Rw2c, Tw2c, cx, cy, fx, fy):
    camera3d = np.dot(Rw2c, world3d) + Tw2c

    x = camera3d[0, :] * fx / world3d[2, :] + cx
    y = camera3d[1, :] * fy / world3d[2, :] + cy

    x = np.round(x)
    y = np.round(y)

    points2d = np.array((x, y), dtype=int)  # 2, N
    return points2d


def projected_box_from_original(org_points, proj_points, org_boxes, org_img, proj_img,
                                visualize_each=False, parallel=False):
    """

    Parameters
    ----------
    org_points: np.array (N,2)
        Points in original image in Image coordinate frame
    proj_points: np.array (N,2)
        Projected points in projected image's coordinate frame
    org_boxes: np.array (M, 4)
        Boxes in original frame for which projected points are used.
    org_img: np.array (H, W)
        Original image from which points are projected.
        Used only for visualization.
    proj_img: np.array (H, W)
        Image where points form original are projected.
        Used only for visualization.
    visualize_each: bool
        Flag to visualize each pair of boxes in the intermediate step.
    parallel: bool
        Flag to calculate indices of all boxes in single pass using big array.

    Returns
    -------
    valid_projected_boxes: list
        List of boxes that enclose projected points of boxes from original.
    valid_projected_box_indices: list
        Ordered (w.r.t valid_projected_boxes) list of indices that
        correspond to the boxes from original images which survived the
        projection. The boxes in valid_projected_boxes come from boxes in
        original image at the index in this list.
    """
    if isinstance(org_boxes, torch.Tensor):
        org_boxes = org_boxes.cpu().numpy()

    valid_projected_boxes = []
    valid_projected_box_indices = []

    iterate_over = org_boxes

    if parallel:
        num_org_boxes = len(org_boxes)
        # Index with shape (num_org_boxes, 1)
        org_box_x1 = org_boxes[:, 0, np.newaxis]
        org_box_y1 = org_boxes[:, 1, np.newaxis]
        org_box_x2 = org_boxes[:, 2, np.newaxis]
        org_box_y2 = org_boxes[:, 3, np.newaxis]

        org_points_x = np.tile(org_points[:, 0], (num_org_boxes, 1))
        org_points_y = np.tile(org_points[:, 1], (num_org_boxes, 1))

        point_mask1 = org_points_x > org_box_x1
        point_mask2 = org_points_x < org_box_x2
        point_mask3 = org_points_y > org_box_y1
        point_mask4 = org_points_y < org_box_y2
        point_mask = reduce(np.logical_and, (point_mask1, point_mask2, point_mask3, point_mask4))
        iterate_over = point_mask

    for box_idx, box_var in enumerate(iterate_over):
        if not parallel:
            # Find indices of points inside the box from original image
            org_box = box_var
            inds1 = np.where(org_points[:, 0] > org_box[0])
            inds2 = np.where(org_points[:, 1] > org_box[1])
            inds3 = np.where(org_points[:, 0] < org_box[2])
            inds4 = np.where(org_points[:, 1] < org_box[3])
            inds = reduce(np.intersect1d, (inds1, inds2, inds3, inds4))

            # Find the projected points corresponding to box from original image
            box_proj_points = proj_points[inds, :]
            box_org_points = org_points[inds, :]
        else:
            box_mask = box_var
            org_box = org_boxes[box_idx]
            box_proj_points = proj_points[box_mask, :]  # box_var is box_mask
            box_org_points = org_points[box_mask, :]

        # Skip if no points inside the box (possible case when no z)
        if box_proj_points.shape[0] <= 0:
            continue

        # Find the max and min of projected points to get projected box
        x1 = box_proj_points[:, 0].min()
        y1 = box_proj_points[:, 1].min()
        x2 = box_proj_points[:, 0].max()
        y2 = box_proj_points[:, 1].max()
        box_proj = (x1, y1, x2, y2)

        valid_projected_boxes.append(box_proj)
        valid_projected_box_indices.append(box_idx)

        if visualize_each:
            v = Visualizer2(org_img[:, :, ::-1], scale=1.0)
            out = v.draw_box(box_coord=org_box)

            v2 = Visualizer2(proj_img[:, :, ::-1], scale=1.0)
            out2 = v2.draw_box(box_coord=box_proj)

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            ax[0].imshow(out.get_image())
            ax[0].scatter(box_org_points[:, 0], box_org_points[:, 1], s=1, edgecolors='r')
            ax[0].set_title(f"Original Image Boxes: \n{org_box}")

            ax[1].imshow(out2.get_image())
            ax[1].scatter(box_proj_points[:, 0], box_proj_points[:, 1], s=1, edgecolors='r')
            ax[1].set_title(f"Projected Image Boxes:")

            fig.suptitle(f'Proposal in Neighbor img {box_idx}:')
            plt.show()
            plt.close()

    return valid_projected_boxes, valid_projected_box_indices
