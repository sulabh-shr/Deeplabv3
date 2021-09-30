import os

class Parameters_ADE20K(object):
	def __init__(self):
		self.backbone = 'resnet' #'resnet', 'xception', 'drn', 'mobilenet'
		self.out_stride = 16 #8
		self.dataset = 'ade20k' # 'pascal', 'coco', 'cityscapes'
		self.checkname = 'sseg'
		self.use_sbd = True
		self.workers = 4
		self.base_size = 640
		self.crop_size = 640
		self.resize_ratio = 1.0
		self.sync_bn = True
		self.freeze_bn = False
		self.loss_type = 'ce' # 'ce', 'focal'

		self.dropout = 0.2

		# training hyper params
		self.epochs = 70
		self.batch_size = 32
		self.test_batch_size = 32
		self.use_balanced_weights = False
		self.start_epoch = 0

		# optimizer params
		self.lr = 0.01
		self.lr_scheduler = 'poly' # 'poly', 'step', 'cos'

		# cuda, seed and logging
		self.cuda = True

		# checking point
		self.resume = None
		self.checkname = 'deeplab_{}'.format(self.backbone)

		# finetuning pre-trained models
		self.ft = False

		# evaluation option
		self.eval_interval = 1
		self.no_val = False
