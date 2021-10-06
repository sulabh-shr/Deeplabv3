import os
import colorsys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
from projection import project_img

set1 = ['000110001850101.jpg',
        '000110001000101.jpg',
        '000110002330101.jpg',
        '000110007940101.jpg',
        '000110011150101.jpg',
        '000110013840101.jpg']

set2 = ['000110002210101.jpg',
        '000110000170101.jpg',
        '000110002690101.jpg',
        '000110006860101.jpg',
        '000110011870101.jpg',
        '000110013240101.jpg']

set3 = ['000110002570101.jpg',
        '000110002330101.jpg',
        '000110003050101.jpg',
        '000110000260101.jpg',
        '000110012110101.jpg',
        '000110012640101.jpg']

viz = True


def class_img_to_color(class_image, colors):
    """

    Args:
        class_image:
        colors:
        num_class:

    Notes:
        Assumption is that the class_image and the colors list will contain
        background class as well.
    Returns:

    """
    color_img = np.ones((class_image.shape[0], class_image.shape[1], 3), dtype=np.float32)
    classes_available = []
    unique_labels = np.unique(class_image)

    for c in unique_labels:
        class_mask = class_image == c
        if np.any(class_mask):
            color_img[class_mask, :] = colors[c]
            classes_available.append(c)
    print(classes_available)
    return color_img, classes_available


if __name__ == '__main__':
    np.random.seed(1)
    # src_name = '000110000010101.jpg'
    # src_name = '000110000020101.jpg'
    # src_name = '000110000590101.jpg'
    rgb_folder = '/mnt/sda2/workspace/DATASETS/ActiveVision/Home_001_1/jpg_rgb'
    depth_folder = '/mnt/sda2/workspace/DATASETS/ActiveVision/Home_001_1/high_res_depth'
    label_folder = '/mnt/sda2/workspace/OUTPUTS/SemSeg/ade20k/for_prop/numpy'

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

    num_classes = 65
    color_list = [(1.0 * (i / num_classes), 0.7, 0.5) for i in range(num_classes)]
    color_list.append((0, 0, 0))  # Background class
    color_list = [colorsys.hsv_to_rgb(*i) for i in color_list]

    for src1_name, dest_name, src2_name in zip(set1, set2, set3):
        data = {}
        dest_basename = dest_name.split('.')[0]
        dest_rgb = np.array(Image.open(os.path.join(rgb_folder, dest_name)))
        data['dest_rgb'] = dest_rgb
        src_names = [src1_name, src2_name]
        for src_idx, src_name in enumerate(src_names, start=1):
            src_basename = src_name.split('.')[0]
            src_depth_name = src_basename[:-1] + '3.png'
            src_rgb = np.array(Image.open(os.path.join(rgb_folder, src_name)))
            src_depth = Image.open(os.path.join(depth_folder, src_depth_name))
            src_depth = np.array(src_depth, dtype=np.int)
            data[f'src{src_idx}_rgb'] = src_rgb

            img_name_idx = image_structs.dtype.names.index('image_name')
            t_col_idx = image_structs.dtype.names.index('t')
            R_col_idx = image_structs.dtype.names.index('R')
            all_t_list = [i[t_col_idx] for i in image_structs]
            all_R_list = [i[R_col_idx] for i in image_structs]
            all_images = [i[img_name_idx][0] for i in image_structs]

            dest_extrinsic, src_extrinsic = dict(), dict()
            for d, n in zip((dest_extrinsic, src_extrinsic), (dest_name, src_name)):
                mat_idx = all_images.index(n)
                d['t'] = all_t_list[mat_idx]
                d['R'] = all_R_list[mat_idx]

            src_transformed_points, src_survived_points = project_img(src_rgb, src_depth, camera_params, src_extrinsic=src_extrinsic,
                                                                      dst_extrinsic=dest_extrinsic, dest=None, scale=scale)

            src_survived_rgb = np.zeros_like(src_rgb)
            src_survived_rgb[src_survived_points[:, 1], src_survived_points[:, 0]] = src_rgb[
                src_survived_points[:, 1], src_survived_points[:, 0]]

            dest_src_rgb = np.zeros_like(src_rgb)
            dest_src_rgb[src_transformed_points[:, 1], src_transformed_points[:, 0]] = src_rgb[
                src_survived_points[:, 1], src_survived_points[:, 0]]

            # with open(os.path.join(label_folder, f'{src_basename}_means.npy'), 'rb') as f:
            src_uncertainty = np.load(os.path.join(label_folder, f'{src_basename}_uncertainty.npy'))  # 640, 1137
            data[f'src{src_idx}_uncertainty'] = src_uncertainty
            src_means = np.load(os.path.join(label_folder, f'{src_basename}_means.npy'))  # 65, 640, 1137
            src_labels = np.argmax(src_means, axis=0).astype(np.int32)
            src_scores = np.max(src_means, axis=0).astype(np.float32)
            # Reshape to original rgb image size
            h, w, _ = src_rgb.shape
            src_labels = np.array(Image.fromarray(src_labels).resize((w, h), resample=Image.NEAREST))
            src_scores = np.array(Image.fromarray(src_scores).resize((w, h), resample=Image.NEAREST))
            data[f'src{src_idx}_labels'] = src_labels
            data[f'src{src_idx}_scores'] = src_scores

            dest_src_labels = np.ones_like(src_labels) * num_classes  # Init with all bg pixels
            dest_src_labels[src_transformed_points[:, 1], src_transformed_points[:, 0]] = \
                src_labels[src_survived_points[:, 1], src_survived_points[:, 0]]
            dest_src_color, dest_src_classes = class_img_to_color(dest_src_labels, colors=color_list)
            data[f'dest_src{src_idx}_labels'] = dest_src_labels
            data[f'dest_src{src_idx}_colors'] = dest_src_color

            dest_src_scores = np.zeros_like(src_scores)
            dest_src_scores[src_transformed_points[:, 1], src_transformed_points[:, 0]] = \
                src_scores[src_survived_points[:, 1], src_survived_points[:, 0]]
            data[f'dest_src{src_idx}_scores'] = dest_src_scores

            if viz:
                fig = plt.figure(src_idx)

                ax1 = fig.add_subplot(231)
                ax1.imshow(src_rgb)
                ax1.set_title(f'Src: {src_basename}')

                ax5 = fig.add_subplot(235)
                ax5.imshow(dest_src_rgb)
                ax5.set_title('Src to Dest Transformed')

                ax2 = fig.add_subplot(232)
                ax2.imshow(src_depth)
                ax2.set_title('Src Depth')

                ax3 = fig.add_subplot(233)
                ax3.imshow(src_survived_rgb)
                ax3.set_title('Src survived points')

                if dest_rgb is not None:
                    dest_org_pixels = np.zeros_like(dest_rgb)
                    dest_org_pixels[src_transformed_points[:, 1], src_transformed_points[:, 0]] = dest_rgb[
                        src_transformed_points[:, 1], src_transformed_points[:, 0]]

                    ax4 = fig.add_subplot(234)
                    ax4.imshow(dest_rgb)
                    ax4.set_title(f'Dest : {dest_basename}')

                    ax6 = fig.add_subplot(236)
                    ax6.imshow(dest_org_pixels)
                    ax6.set_title('Dest pixels in the valid coordinates')

        # Accumulate labels
        dest_srcs_labels = data['dest_src1_labels'].copy()
        current_score = data['dest_src1_scores']
        num_src = len(src_names)
        cols = num_src + 1
        for src_idx in range(1, num_src + 1):
            src_score = data[f'dest_src{src_idx}_scores']
            src_label = data[f'dest_src{src_idx}_labels']

            greater_score_mask = src_score > current_score
            dest_srcs_labels[greater_score_mask] = src_label[greater_score_mask]
            current_score[greater_score_mask] = src_score[greater_score_mask]
        dest_srcs_color, dest_srcs_classes = class_img_to_color(dest_srcs_labels, colors=color_list)
        data['dest_srcs_label'] = dest_srcs_labels
        data['dest_srcs_color'] = dest_srcs_color

        if viz:
            fig2 = plt.figure(num_src + 1)
            ax1 = fig2.add_subplot(2, cols, cols)
            ax1.imshow(data['dest_rgb'])
            ax1.set_title(f'Dest: {dest_name}')
            ax2 = fig2.add_subplot(2, cols, 2 * cols)
            ax2.imshow(data['dest_srcs_color'])

            for src_idx in range(1, num_src + 1):
                ax = fig2.add_subplot(2, cols, src_idx)
                ax.imshow(data[f'src{src_idx}_rgb'])
                ax.set_title(f'Src{src_idx}')
                ax2 = fig2.add_subplot(2, cols, src_idx + cols)
                ax2.imshow(data[f'dest_src{src_idx}_colors'])
            plt.show()
# src_to_dest_img = np.zeros_like(src)
# src_to_dest_img[src_transformed_points[:, 1], src_transformed_points[:, 0]] = src[src_survived_points[:, 1], src_survived_points[:, 0]]

# fig = plt.figure(1)

# fig = plt.figure(1)
# ax1 = fig.add_subplot(231)
# ax1.imshow(src_rgb)
# ax1.set_title(f'Src: {src_basename}')

#     src1_to_dest_means = np.zeros_like(coords[f'src1_means'])
#     src1_to_dest_means[transformed_coords[:, 1], transformed_coords[:, 0]] = coords['src1_means'][src_coords[:, 1], src_coords[:, 0]]

#     src1_to_dest_labels = np.argmax(src1_to_dest_means, axis=0)

# num_class = 65
# color_list = []
# for _ in range(num_class):
#     color_list.append(np.array([np.random.rand(),np.random.rand(),np.random.rand()]))

#     fig = plt.figure(1)
#     ax1 = fig.add_subplot(231)
#     ax1.imshow(src)
#     ax1.set_title('Src')

#     ax5 = fig.add_subplot(235)
#     ax5.imshow(src_to_dest_img)
#     ax5.set_title('Src to Dest Transformed')

#     ax2 = fig.add_subplot(232)
#     ax2.imshow(src_depth)
#     ax2.set_title('Src Depth')

#     ax3 = fig.add_subplot(233)
#     ax3.imshow(src_survived)
#     ax3.set_title('Src survived points')

#     if dest is not None:
#         ax4 = fig.add_subplot(234)
#         ax4.imshow(dest)
#         ax4.set_title('Dest')

#         ax6 = fig.add_subplot(236)
#         ax6.imshow(dest_org_pixels)
#         ax6.set_title('Dest pixels in the valid coordinates')

#     plt.show()
