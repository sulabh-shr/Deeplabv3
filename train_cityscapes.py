import os
import numpy as np
from modeling.models import deeplabv3plus_resnet101, deeplabv3plus_resnet50, deeplabv3plus_mobilenet
from utils.loss import SegmentationLosses
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lr_scheduler import PolyLR
from modeling.utils import set_bn_momentum

from parameters import Parameters
from dataloaders.datasets import cityscapes
from torch.utils.data import DataLoader
import torch

par = Parameters()

#=========================================================== Define Saver =======================================================
saver = Saver(par)
# Define Tensorboard Summary
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

#=========================================================== Define Dataloader ==================================================
dataset_train = cityscapes.CityscapesDataset(par, dataset_dir='/projects/kosecka/yimeng/Datasets/Cityscapes', split='train')
num_class = dataset_train.NUM_CLASSES
dataloader_train = DataLoader(dataset_train, batch_size=par.batch_size, shuffle=True, num_workers=int(par.batch_size/2))

dataset_val = cityscapes.CityscapesDataset(par, dataset_dir='/projects/kosecka/yimeng/Datasets/Cityscapes', split='val')
dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))

#================================================================================================================================
# Define network
model = deeplabv3plus_resnet50(num_classes=num_class, output_stride=par.out_stride).cuda()

set_bn_momentum(model.backbone, momentum=0.01)

#=========================================================== Define Optimizer ================================================
import torch.optim as optim
train_params = [{'params': model.backbone.parameters(), 'lr': par.lr*0.1},
                {'params': model.classifier.parameters(), 'lr': par.lr}]
optimizer = optim.SGD(train_params, lr=par.lr, momentum=0.9, weight_decay=1e-4)
scheduler = PolyLR(optimizer, 10000, power=0.9)

# Define Criterion
# whether to use class balanced weights
weight = None
criterion = SegmentationLosses(weight=weight, cuda=par.cuda).build_loss(mode=par.loss_type)

# Define Evaluator
evaluator = Evaluator(num_class)

#===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if par.resume is not None:
    if not os.path.isfile(par.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(par.resume))
    checkpoint = torch.load(par.resume)
    par.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {})".format(par.resume, checkpoint['epoch']))

#=================================================================trainin
for epoch in range(par.epochs):
    train_loss = 0.0
    model.train()
    num_img_tr = len(dataloader_train)
    
    for iter_num, sample in enumerate(dataloader_train):
        print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
        images, targets = sample['image'], sample['label']
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets.shape))
        images, targets = images.cuda(), targets.cuda()
        
        #================================================ compute loss =============================================
        output = model(images)
        #print('output.shape = {}'.format(output.shape))

        loss = criterion(output, targets)

        #================================================= compute gradient =================================================
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('Train loss: %.3f' % (train_loss / (iter_num + 1)))
        writer.add_scalar('train/total_loss_iter', loss.item(), iter_num + num_img_tr * epoch)

        '''
        # Show 10 * 3 inference results each epoch
        if iter_num % (num_img_tr // 10) == 0:
            global_step = iter_num + num_img_tr * epoch
            summary.visualize_image(writer, par.dataset, images, targets, output, global_step)
        '''

    writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * par.batch_size + images.data.shape[0]))
    print('Loss: %.3f' % train_loss)

#======================================================== evaluation stage =====================================================

    if epoch % par.eval_interval == 0:
        model.eval()
        evaluator.reset()
        test_loss = 0.0
        for iter_num, sample in enumerate(dataloader_val):
            print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
            images, targets = sample['image'], sample['label']
            #print('images = {}'.format(images))
            #print('targets = {}'.format(targets))
            images, targets = images.cuda(), targets.cuda()

            #========================== compute loss =====================
            with torch.no_grad():
                output = model(images)
            loss = criterion(output, targets)

            test_loss += loss.item()
            print('Test loss: %.3f' % (test_loss / (iter_num + 1)))
            pred = output.data.cpu().numpy()
            targets = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(targets, pred)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        writer.add_scalar('val/mIoU', mIoU, epoch)
        writer.add_scalar('val/Acc', Acc, epoch)
        writer.add_scalar('val/Acc_class', Acc_class, epoch)
        writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * par.batch_size + images.data.shape[0]))
        print("Acc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best)

trainer.writer.close()







