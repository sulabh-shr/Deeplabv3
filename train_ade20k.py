import os
import numpy as np
from modeling.models import deeplabv3plus_resnet101, deeplabv3plus_resnet50, deeplabv3plus_mobilenet
from utils.loss import SegmentationLosses
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lr_scheduler import PolyLR
from modeling.utils import set_bn_momentum
import matplotlib.pyplot as plt
from parameters_ade20k import Parameters_ADE20K
from dataloaders.datasets import ade20k
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

#=========================================================== Parser =======================================================

parser = argparse.ArgumentParser()
parser.add_argument('--resume', help='Path to checkpoint for resuming training', type=str,
	default=None)
args = parser.parse_args()


par = Parameters_ADE20K()
if args.resume:
	par.resume = args.resume

#=========================================================== Define Saver =======================================================
saver = Saver(par)
# Define Tensorboard Summary
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

#=========================================================== Define Dataloader ==================================================

dataset_train = ade20k.ADE20KDataset(par, dataset_dir='/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation', split='train')
print(f'Total length of Train dataset = {len(dataset_train)}')
num_classes = dataset_train.NUM_CLASSES
dataloader_train = DataLoader(dataset_train, batch_size=par.batch_size, shuffle=True, num_workers=int(par.batch_size/2))
print(f'Total length of Train dataloader = {len(dataloader_train)}')

dataset_val = ade20k.ADE20KDataset(par, dataset_dir='/projects/kosecka/yimeng/Datasets/ADE20K/Semantic_Segmentation', split='val')
print(f'Total length of Val dataset = {len(dataset_val)}')
dataloader_val = DataLoader(dataset_val, batch_size=par.test_batch_size, shuffle=False, num_workers=int(par.test_batch_size/2))
print(f'Total length of Val dataloader = {len(dataloader_val)}')

#================================================================================================================================
# Define network
model = deeplabv3plus_resnet50(num_classes=num_classes, output_stride=par.out_stride, dropout=par.dropout).cuda()
set_bn_momentum(model.backbone, momentum=0.01)

#=========================================================== Define Optimizer ================================================
import torch.optim as optim
train_params = [{'params': model.backbone.parameters(), 'lr': par.lr*0.1},
				{'params': model.classifier.parameters(), 'lr': par.lr}]
model = nn.DataParallel(model)
print(model)
optimizer = optim.SGD(train_params, lr=par.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

# Define Criterion
# whether to use class balanced weights
weight = None
criterion = SegmentationLosses(weight=weight, cuda=par.cuda).build_loss(mode=par.loss_type)

# Define Evaluator
evaluator = Evaluator(num_classes)

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

#================================================================= training ==============
print_freq = 10
for epoch in tqdm(range(par.start_epoch, par.epochs)):
	print(f'==============| Epoch = {epoch+1} |==============')
	train_loss = 0.0
	model.train()
	num_img_tr = len(dataloader_train)
	
	for iter_num, sample in enumerate(dataloader_train):
		if iter_num % print_freq == 0:
			print(f'Epoch = {epoch+1}/{par.epochs} | Iter_num = {iter_num}/{num_img_tr}')
			print('Train loss: %.3f' % (train_loss / (iter_num + 1)))

		images, targets = sample['image'], sample['label']
		#print('images = {}'.format(images.shape))
		#print('targets = {}'.format(targets.shape))
		#assert 1==2
		images, targets = images.cuda(), targets.cuda()
		
		#================================================ compute loss =============================================
		output = model(images)
		loss = criterion(output, targets)

		#================================================= compute gradient =================================================
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		
		writer.add_scalar('train/total_loss_iter', loss.item(), iter_num + num_img_tr * epoch)

	writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
	print('[Epoch: %d, numImages: %5d]' % (epoch+1, iter_num * par.batch_size + images.data.shape[0]))
	print('Loss: %.3f' % train_loss)

#======================================================== evaluation stage =====================================================

	if epoch % par.eval_interval == 0:
		model.eval()
		evaluator.reset()
		test_loss = 0.0
		print('\nValidation:')
		for iter_num, sample in enumerate(dataloader_val):
			if iter_num%print_freq == 0:
				print('Val epoch = {}| Val iter_num = {}'.format(epoch, iter_num))
				print('Test loss: %.3f' % (test_loss / (iter_num + 1)))

			images, targets = sample['image'], sample['label']
			#print('images = {}'.format(images))
			#print('targets = {}'.format(targets))
			images, targets = images.cuda(), targets.cuda()

			#========================== compute loss =====================
			with torch.no_grad():
				output = model(images)
			loss = criterion(output, targets)

			test_loss += loss.item()
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
		print('\tEpoch: %d, numImages: %5d' % (epoch, iter_num * par.batch_size + images.data.shape[0]))
		print("\tAcc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))
		print('\tLoss: %.3f' % test_loss)

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
	scheduler.step(mIoU)

writer.close()







