import os
import math
import colorsys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from PIL import Image
from projection import project_img
from time import time

from dataloaders.datasets.ade20k import ADE20KDataset
from image_path_examples import paths

# set1 = ['000110001850101.jpg',
#         '000110001000101.jpg',
#         '000110002330101.jpg',
#         '000110007940101.jpg',
#         '000110011150101.jpg',
#         '000110013840101.jpg']
#
# set2 = ['000110002210101.jpg',
#         '000110000170101.jpg',
#         '000110002690101.jpg',
#         '000110006860101.jpg',
#         '000110011870101.jpg',
#         '000110013240101.jpg']
#
# set3 = ['000110002570101.jpg',
#         '000110002330101.jpg',
#         '000110003050101.jpg',
#         '000110000260101.jpg',
#         '000110012110101.jpg',
#         '000110012640101.jpg']

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


def color_range(num, s_range=((0.5, 1.0,), (0.5, 1.0), (1,)), l_range=(0.3, 0.5, 0.8),
                color_scale='hls', viz=True):
    """

    Args:
        num: number of colors
        s_range: tuple of int/floats or tuple of tuples
            s_range for e
        l_range: tuple

    Returns:
        list of colors in color_scale
    """
    assert len(s_range) == len(l_range), f'Need to provide s values for each l'

    if isinstance(s_range[0], (int, float)):
        s_range_per_l = {l_range[idx]: (s_range[idx],) for idx in range(len(l_range))}
    else:
        s_range_per_l = {l_range[idx]: s_range[idx] for idx in range(len(l_range))}
    variations = sum([len(v) for v in s_range_per_l.values()])

    hue_num = math.ceil(num / variations)
    hue_range = [(1.0 * (i / hue_num)) for i in range(hue_num)]
    print(hue_range)
    colors = []

    for h in hue_range:
        for li in l_range:
            for s in s_range_per_l[li]:

                color = (h, li, s)
                if color_scale == 'hls':
                    colors.append(color)
                elif color_scale == 'rgb':
                    colors.append(colorsys.hls_to_rgb(*color))
                else:
                    raise TypeError(f'Invalid color scale: {color_scale}')

                if len(colors) == num_classes:
                    if viz:
                        img = np.zeros((1080, 1080, 3), dtype=np.float32)
                        delta = math.floor(math.sqrt(1080 * 1080// len(colors)))
                        color_count = 0
                        for x in range(0, 1080, delta):
                            for y in range(0, 1080, delta):
                                if color_count == len(colors):
                                    break
                                img[x:x+delta, y:y+delta, :] = colorsys.hls_to_rgb(*colors[color_count])
                                color_count += 1
                        plt.figure()
                        plt.imshow(img)
                        plt.show()
                        plt.close()

                    return colors


if __name__ == '__main__':
    num_classes = count = 65
    color_list = color_range(num_classes)
    color_list.append((0, 1, 0))  # Background class
    color_list = [colorsys.hls_to_rgb(*i) for i in color_list]

    seed = int(time() % 1000)
    np.random.seed(seed)
    print(f'Using seed: {seed}')
    class_names = ADE20KDataset.class_names
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

    np.random.shuffle(paths)
    for path in paths:
        selected_sources = 2
        random_dest_idx = np.random.choice(range(2, len(path) - 2))
        # src_names = [path[random_dest_idx + i] for i in range(-selected_sources // 2, selected_sources // 2 + 1) if i != 0]
        src_names = [path[random_dest_idx + i] for i in (-2, 2)]
        dest_name = path[random_dest_idx]
        data = {}
        dest_basename = dest_name.split('.')[0]
        dest_rgb = np.array(Image.open(os.path.join(rgb_folder, dest_name)))
        data['dest_rgb'] = dest_rgb

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
                ax1.set_title(f'Src{src_idx}: {src_basename}')

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
                src_color, src_classes = class_img_to_color(data[f'src{src_idx}_labels'], colors=color_list)
                ax.imshow(src_color)
                ax.set_title(f'Src{src_idx}')
                ax2 = fig2.add_subplot(2, cols, src_idx + cols)
                ax2.imshow(data[f'dest_src{src_idx}_colors'])

            legend_elements = []
            for c in np.unique(dest_srcs_labels)[:-1]:  # all classes except bg
                legend_elements.append(Line2D([0], [0], color=color_list[c], label=class_names[c], linestyle='-', marker='P', linewidth=10))
            fig2.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=8)

            plt.show()
