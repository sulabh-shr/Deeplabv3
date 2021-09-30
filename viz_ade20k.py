import os
import numpy as np
from modeling.models import deeplabv3plus_resnet50
from parameters import Parameters
import torch
import torch.nn.functional as F
from PIL import Image
from parameters_ade20k import Parameters_ADE20K
from dataloaders.datasets import ade20k
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time


mode = 'resNet' #'resNet', 'mobileNet'

save_folder = '/scratch/sshres2/workspace-scratch/outputs/semseg/ade20k/'
dataset = 'ade20k' 

if dataset == 'ade20k':
    dataset_folder = '/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation'

#====================================================== change the parameters============================================================
par = Parameters()
par.test_batch_size = 1

#'''ResNet
# par.resume = 'run/ade20k/deeplab_resnet/experiment_6/checkpoint.pth.tar'
par.resume = '/home/sshres2/workspace/my_git/Deeplabv3/run/ade20k/deeplab_resnet/experiment_2/checkpoint.pth.tar'
#'''

#=========================================================== Define Dataloader ==================================================
if dataset == 'ade20k':
    dataset_val = ade20k.ADE20KDataset(par, dataset_dir=dataset_folder, split='val')
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    num_class = dataset_val.NUM_CLASSES

#================================================================================================================================
# Define network
model = deeplabv3plus_resnet50(num_classes=num_class, output_stride=par.out_stride).cuda()

#===================================================== Resuming checkpoint ====================================================
if par.resume is not None:
    if not os.path.isfile(par.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(par.resume))
    checkpoint = torch.load(par.resume)
    par.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(par.resume, checkpoint['epoch']))

#======================================================== evaluation stage =====================================================
    model.eval()

    count = 0
    color_list = []
    for _ in range(num_class):
        color_list.append(np.array([np.random.rand(),np.random.rand(),np.random.rand()]))

    for iter_num, sample in enumerate(dataloader_val):
        #if dataset == 'cityscapes' and iter_num == 10:
        #    break
        print('iter_num = {}'.format(iter_num))
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets))
        images = images.cuda()

        #================================================ compute loss =============================================
        with torch.no_grad():
            output = model(images) #output.shape = batch_size x num_classes x H x W
            print(output.shape)
            for idx, output_img in enumerate(output):
                print(f'Inferencing on: {idx+1}/{len(output)}')
                fig, ax = plt.subplots(1,2,figsize=(30,15))
                legend_elements = []
                output_img = torch.argmax(output_img, dim=0).cpu().numpy()  # H x W
                present_classes = np.unique(output_img)

                output_label_img = np.zeros((output_img.shape[0], output_img.shape[1], 3))
                print(present_classes, flush=True)
                for c in present_classes:
                    output_label_img[output_img==c,:] = color_list[c]
                    legend_elements.append(Line2D([0], [0], color=color_list[c], label=dataset_val.class_names[c], linestyle='-', linewidth=3))
            
                ax[0].imshow(images[idx].permute(1,2,0).cpu().numpy())
                ax[1].imshow(output_label_img)
                ax[1].legend(handles=legend_elements, loc='upper right')
                
                plt.savefig(os.path.join(save_folder, f'{count}.png'))
                count += 1
            # print('z.shape = {}'.format(z.shape))
            #H, W = targets.shape[-2:]
            #input_shape = (int(H/2), int(W/2))
            #z_interpolated = F.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)
            #print('interpolated z.shape = {}'.format(z_interpolated.shape))
            
            #z_interpolated = z_interpolated.data.cpu().numpy()
            # z = z.data.cpu().numpy()
            #targets = targets.numpy().astype(np.uint8)
            # assert 1==2

        # N, _, _, _ = z.shape
        # for i in range(N):
            #result = {}
            #result['feature'] = z_interpolated[i]
            #target = cv2.resize(targets[i], input_shape[::-1], cv2.INTER_NEAREST)
            #print('target.shape = {}'.format(target.shape))
            #result['label'] = target
            # result = z[i]
            #assert 1==2

            
            # np.save('{}/{}_deeplab_ft.npy'.format(saved_folder, count), result)


            #assert 1==2
            # count += 1
        #assert 1==2