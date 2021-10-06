import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
from projection_utils import get3Dpoints, world2img


def project_img(src, src_depth, intrinsics, src_extrinsic, dst_extrinsic, dest=None, scale=1,
    viz=False):
    Tw2s = src_extrinsic['t']
    Rw2s = src_extrinsic['R']

    Tw2d = dst_extrinsic['t']
    Rw2d = dst_extrinsic['R']

    cx, cy, fx, fy = intrinsics['cx'], intrinsics['cy'], intrinsics['fx'], intrinsics['fy']
    intrinsics = (fx, fy, cx, cy)

    print(src_depth.shape)
    imh, imw = src_depth.shape
    valid_src_inds = np.where(src_depth > 0)
    src_img_coords = np.zeros((len(valid_src_inds[0]), 2), dtype=np.int)  # N, 2

    # Image coordinate frame xy is reverse of that in numpy array
    # i.e. x goes from left to right and y top to bottom
    src_img_coords[:, 0] = valid_src_inds[1]  # inds[1] is x in Image coordinate frame
    src_img_coords[:, 1] = valid_src_inds[0]  # inds[0] is y in Image coordinate frame

    src_world_3d = get3Dpoints(src_img_coords, depth=src_depth, intr=intrinsics, Rw2c=Rw2s, Tw2c=Tw2s, scale=scale)
    # Points in Src coordinate frame which were obtained by transforming Dest image pixels
    src_transformed_points, valid_inds = world2img(src_world_3d, intrinsics, Rw2c=Rw2d, Tw2c=Tw2d, im_dim=(imh, imw))
    # Points in the original Dest image frame which were used to get the transformed image points above
    src_survived_points = src_img_coords[valid_inds, :]
    # Note: src_transformed_points and src_survived_points have 1-1 relation i.e. former is the transformed from latter

    src_survived = np.zeros_like(src)    
    src_survived[src_survived_points[:, 1], src_survived_points[:, 0]] = src[src_survived_points[:, 1], src_survived_points[:, 0]]

    src_to_dest_img = np.zeros_like(src)
    src_to_dest_img[src_transformed_points[:, 1], src_transformed_points[:, 0]] = src[src_survived_points[:, 1], src_survived_points[:, 0]]

    # Check points in the Src image corresponding to the transformed points
    if dest is not None:
        dest_org_pixels = np.zeros_like(dest)
        dest_org_pixels[src_transformed_points[:, 1], src_transformed_points[:, 0]] = dest[src_transformed_points[:, 1], src_transformed_points[:, 0]]


    if viz:
        fig = plt.figure(1)
        ax1 = fig.add_subplot(231)
        ax1.imshow(src)
        ax1.set_title('Src')

        ax5 = fig.add_subplot(235)
        ax5.imshow(src_to_dest_img)
        ax5.set_title('Src to Dest Transformed')

        ax2 = fig.add_subplot(232)
        ax2.imshow(src_depth)
        ax2.set_title('Src Depth')

        ax3 = fig.add_subplot(233)
        ax3.imshow(src_survived)
        ax3.set_title('Src survived points')

        if dest is not None:
            ax4 = fig.add_subplot(234)
            ax4.imshow(dest)
            ax4.set_title('Dest')
            
            ax6 = fig.add_subplot(236)
            ax6.imshow(dest_org_pixels)
            ax6.set_title('Dest pixels in the valid coordinates')
        
        plt.show()

    return src_transformed_points, src_survived_points


if __name__ == '__main__':
    # src_name = '000110000010101.jpg'
    # src_name = '000110000020101.jpg'
    # src_name = '000110000590101.jpg'
    src_name = '000110001320101.jpg'
    dest_name = '000110000250101.jpg'

    rgb_folder = '/mnt/sda2/workspace/DATASETS/ActiveVision/Home_001_1/jpg_rgb'
    depth_folder = '/mnt/sda2/workspace/DATASETS/ActiveVision/Home_001_1/high_res_depth'

    extract_depth = lambda x: x.split('.')[0][:-1] + '3.png'

    rgb_src = np.array(Image.open(os.path.join(rgb_folder, src_name)))
    rgb_dest = np.array(Image.open(os.path.join(rgb_folder, dest_name)))

    # depth_dest = Image.open(os.path.join(depth_folder, extract_depth(dest_name)))
    # depth_dest = np.array(depth_dest, dtype=np.int)

    depth_src = Image.open(os.path.join(depth_folder, extract_depth(src_name)))
    depth_src = np.array(depth_src, dtype=np.int)

    camera_path = '/mnt/sda2/workspace/DATASETS/ActiveVision/camera_params/Home_001_1/cameras.txt'
    if os.path.isfile(camera_path):
        with open(camera_path, 'r') as f:
            file_contents = f.read()
        params_line = file_contents.split('\n')[3]
        params_split = params_line.split(' ')

        camera_params = {
            'fx': float(params_split[4]),
            'fy': float(params_split[5]),
            'cx': float(params_split[6]),
            'cy': float(params_split[7])
        }
    else:
        raise FileNotFoundError

    mat_file = loadmat('/mnt/sda2/workspace/DATASETS/ActiveVision/Home_001_1/image_structs.mat')
    image_structs = mat_file['image_structs'][0]
    scale = mat_file['scale'][0][0]

    img_name_idx = image_structs.dtype.names.index('image_name')
    t_col_idx = image_structs.dtype.names.index('t')
    R_col_idx = image_structs.dtype.names.index('R')
    all_t_list = [i[t_col_idx] for i in image_structs]
    all_R_list = [i[R_col_idx] for i in image_structs]
    all_images = [i[img_name_idx][0] for i in image_structs]

    dest_idx = all_images.index(dest_name)
    dest_extrinsic = {
        't': all_t_list[dest_idx],
        'R': all_R_list[dest_idx]
    }

    src_idx = all_images.index(src_name)
    src_extrinsic = {
        't': all_t_list[src_idx],
        'R': all_R_list[src_idx]
    }

    project_img(rgb_src, depth_src, camera_params, src_extrinsic=src_extrinsic, dst_extrinsic=dest_extrinsic,
                dest=rgb_dest, scale=scale)
