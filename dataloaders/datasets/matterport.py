import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
import random

class MatterportDataset(data.Dataset):
    
    def __init__(self, par, dataset_dir, split='train', scene_name = None):

        self.dataset_dir = dataset_dir
        self.split = split
        self.par = par

        if scene_name is None:
            assert(1==2, 'scene name is not provided.')
        else:
            self.scene_name = scene_name
            print('load scene: {}'.format(scene_name))

        img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(self.dataset_dir, self.scene_name), allow_pickle=True).item()
        self.img_list = list(img_act_dict.keys())
        self.ins2cat_dict = np.load('{}/{}/dict_ins2category.npy'.format(self.dataset_dir, self.scene_name), allow_pickle=True).item()

        self.void_classes = [0, -1] # matterport has category id -1. Not sure why.
        self.valid_classes = [x for x in range(1, 41)]
        self.class_names = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', \
                            'window', 'sofa','bed', 'curtain', 'chest_of_drawers','plant','sink','stairs', \
                            'ceiling','toilet','stool', 'towel', 'mirror','tv_monitor','shower','column', \
                            'bathtub', 'counter','fireplace','lighting','beam','railing','shelving','blinds', \
                            'gym_equipment','seating','board_panel','furniture','appliances','clothes','objects', \
                            'misc',]
        self.NUM_CLASSES = 40#len(self.valid_classes)

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        print("Found {} {} images".format(len(self.img_list), self.split))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        npy_file = np.load('{}/{}/others/{}.npy'.format(self.dataset_dir, self.scene_name, self.img_list[index]), allow_pickle=True).item()
        InsSeg_img = npy_file['sseg']
        sseg_img = self.convertInsSegToSSeg(InsSeg_img, self.ins2cat_dict)

        img_path = '{}/{}/images/{}.jpg'.format(self.dataset_dir, self.scene_name, self.img_list[index])

        _img = Image.open(img_path).convert('RGB')
        _tmp = self.encode_segmap(sseg_img)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to 255
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def convertInsSegToSSeg (self, InsSeg, ins2cat_dict):
        ins_id_list = list(ins2cat_dict.keys())
        SSeg = np.zeros(InsSeg.shape, dtype=np.int32)
        for ins_id in ins_id_list:
            SSeg = np.where(InsSeg==ins_id, ins2cat_dict[ins_id], SSeg)
        return SSeg

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomCrop(self.par.base_size, self.par.crop_size, fill=255),
            tr.RandomColorJitter(),
            tr.RandomHorizontalFlip(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            #tr.FixedResize(resize_ratio=self.par.resize_ratio),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            #tr.FixedResize(resize_ratio=self.par.resize_ratio),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

'''
if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
'''
