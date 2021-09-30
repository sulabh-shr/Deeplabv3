import os
import numpy as np
from modeling.models import deeplabv3plus_resnet50
import torch
import torch.nn.functional as F
from PIL import Image
from parameters_ade20k import Parameters_ADE20K
from dataloaders.datasets import ade20k
from dataloaders import custom_transforms as tr
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time
import sys
import torch.nn as nn

mode = 'resNet' #'resNet', 'mobileNet'

save_folder = '/scratch/sshres2/workspace-scratch/outputs/semseg/ade20k/uncertainty3'
img_folder = os.path.join(save_folder, 'images')
plot_folder = os.path.join(save_folder, 'plots')
numpy_folder = os.path.join(save_folder, 'numpy')
demormalize = tr.DeNormalizeSingleImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
dataset = 'ade20k' 

if dataset == 'ade20k':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation'

#====================================================== change the parameters============================================================
par = Parameters_ADE20K()
par.test_batch_size = 1

#'''ResNet
# par.resume = 'run/ade20k/deeplab_resnet/experiment_6/checkpoint.pth.tar'
# par.resume = '/home/sshres2/workspace/my_git/Deeplabv3/run/ade20k/deeplab_resnet/experiment_2/checkpoint.pth.tar'
par.resume = '/home/sshres2/workspace/my_git/Deeplabv3/run/ade20k/deeplab_resnet/checkpoint.pth.tar'
#'''
#=========================================================== Define Dataloader ==================================================
if dataset == 'ade20k':
    dataset_val = ade20k.ADE20KDataset(par, dataset_dir=dataset_folder, split='val')
    dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=1)
    num_class = dataset_val.NUM_CLASSES

#================================================================================================================================
# Define network
model = deeplabv3plus_resnet50(num_classes=num_class, output_stride=par.out_stride,
    dropout=par.dropout).cuda()
model = nn.DataParallel(model)

#===================================================== Resuming checkpoint ====================================================
if par.resume is not None:
    if not os.path.isfile(par.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(par.resume))
    checkpoint = torch.load(par.resume)
    par.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(par.resume, checkpoint['epoch']))

#======================================================== evaluation stage =====================================================
    assert dataloader_val.batch_size == 1
    
    num_forwards = 10

    model.eval()

    count = 0
    color_list = []
    for _ in range(num_class):
        color_list.append(np.array([np.random.rand(),np.random.rand(),np.random.rand()]))

    # Make save folders
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(numpy_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)


    for iter_num, sample in enumerate(dataloader_val):
        #if dataset == 'cityscapes' and iter_num == 10:
        #    break
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets))
        images = images.cuda()
        denormalized_image = demormalize(sample['image'][0])
        #================================================ compute loss =============================================
        fig, ax = plt.subplots(2,2,figsize=(15,15))

        output_np_list = None           # F x C x H x W

        for f_idx in range(num_forwards):
            print(f'Inferencing on: {iter_num+1}/{len(dataloader_val)}| Pass #{f_idx}/{num_forwards}')
            with torch.no_grad():
                output = model(images)   # 1 x C x H x W
                output = F.softmax(output, dim=1)
                output = output.squeeze()  # C x H x W

                if output_np_list is None:
                    output_np_list = np.zeros((num_forwards, *output.shape))

                output_np_list[f_idx] = output.cpu().numpy()
        
        output_mean = np.mean(output_np_list, axis=0)       # Per class mean, C x H x W
        output_var = np.var(output_np_list, axis=0)         # Per class variance, C x H x W
        output_labels = np.argmax(output_mean, axis=0)      # H x W
        output_uncertainty = np.mean(output_var, axis=0)    # H x W

        legend_elements = []
        legend_classes = []
        pred_classes = np.unique(output_labels)
        output_img = np.zeros((output_labels.shape[0], output_labels.shape[1], 3))
        for c in pred_classes:
            output_img[output_labels==c,:] = color_list[c]
            legend_elements.append(Line2D([0], [0], color=color_list[c], label=dataset_val.class_names[c], linestyle='-', linewidth=3))
            legend_classes.append(c)
        
        gt_classes = np.unique(output_labels)
        gt_img = np.zeros((output_labels.shape[0], output_labels.shape[1], 3))
        for c in gt_classes:
            gt_img[targets[0]==c,:] = color_list[c]
            if c not in legend_classes:
                legend_elements.append(Line2D([0], [0], color=color_list[c], label=dataset_val.class_names[c], linestyle='-', linewidth=3))
                legend_classes.append(c)

        # ax[0][0].imshow(images[0].permute(1,2,0).cpu().numpy())
        denormalized_image = np.swapaxes(np.swapaxes(denormalized_image, 0, 1), 1, 2).astype(np.uint8)
        ax[0][0].imshow(denormalized_image)
        ax[0][0].set_title('Image')
        ax[0][1].imshow(gt_img)
        ax[0][1].set_title('GT Image')

        ax[1][0].imshow(output_img)
        ax[1][0].set_title('Pred')
        ax[1][1].imshow(output_uncertainty)
        ax[1][1].set_title('Uncertainty')
        
        ax[0][0].legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(os.path.join(plot_folder, f'{count}.png'))
        with open(os.path.join(numpy_folder, f'{count}_means.npy'), 'wb') as f:
            np.save(f, output_mean)

        with open(os.path.join(numpy_folder, f'{count}_uncertainty.npy'), 'wb') as f:
            np.save(f, output_uncertainty)

        Image.fromarray(denormalized_image).save(os.path.join(img_folder, f'{count}.png'))

        count += 1
