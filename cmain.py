from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
# import cupy as cp
import os
import matplotlib
from tqdm import tqdm
import pandas as pd
from pprint import pprint
from utils.eval_tool import eval_detection_voc
from torch.utils import data as data_
from data.VOCdataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils import array_tool as at


class Config:
  # data
  voc_data_dir = './VOCdevkit/VOC2007/'
  min_size = 600  # image resize
  max_size = 1000  # image resize
  num_workers = 8
  test_num_workers = 8
  
  # sigma for l1_smooth_loss
  rpn_sigma = 3.
  roi_sigma = 1.
  
  # param for optimizer
  # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
  weight_decay = 0.0005
  lr_decay = 0.1  # 1e-3 -> 1e-4
  lr = 1e-4
  
  # visualization
  env = 'faster-rcnn'  # visdom env
  port = 8097
  plot_every = 40  # vis every N iter
  
  # preset
  data = 'voc'
  pretrained_model = 'vgg16'
  
  # training
  epoch = 14
  
  use_adam = False  # Use Adam optimizer
  use_chainer = False  # try match everything as chainer
  use_drop = False  # use dropout in RoIHead
  # debug
  debug_file = '/tmp/debugf'
  
  test_num = 10000
  # model
  load_path = None
  
  caffe_pretrain = True  # use caffe pretrained model instead of torchvision
  caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
  
  def _parse(self, kwargs):
    state_dict = self._state_dict()
    for k, v in kwargs.items():
      if k not in state_dict:
        raise ValueError('UnKnown Option: "--%s"' % k)
      setattr(self, k, v)
    
    print('======user config========')
    pprint(self._state_dict())
    print('==========end============')
  
  def _state_dict(self):
    return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
            if not k.startswith('_')}


opt = Config()


def eval(dataloader, faster_rcnn, test_num=10000):
  pred_bboxes, pred_labels, pred_scores = list(), list(), list()
  gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
  for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
    gt_bboxes += list(gt_bboxes_.numpy())
    gt_labels += list(gt_labels_.numpy())
    gt_difficults += list(gt_difficults_.numpy())
    pred_bboxes += pred_bboxes_
    pred_labels += pred_labels_
    pred_scores += pred_scores_
    if ii == test_num: break
  
  result = eval_detection_voc(
    pred_bboxes, pred_labels, pred_scores,
    gt_bboxes, gt_labels, gt_difficults,
    use_07_metric=True)
  return result


def train():
  
  dataset = Dataset(voc_data_dir=['/dataset/VOCdevkit/VOC2007', '/dataset/VOCdevkit/VOC2012'], size=(600, 1000))
  dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True)
  
  testset = TestDataset(voc_data_dir=['/dataset/VOCdevkit/VOC2007'])
  test_dataloader = data_.DataLoader(testset, batch_size=1, shuffle=False)
  
  faster_rcnn = FasterRCNNVGG16()  # anchor_scales=[8,16,32,64]
  trainer = FasterRCNNTrainer(faster_rcnn).cuda()
  print('model construct completed')
  
  best_path = os.listdir("./checkpoints")
  best_path.sort()
  opt.load_path =os.path.join("./checkpoints",best_path[-2])
  if opt.load_path:
    trainer.load(opt.load_path)
    print('load pretrained model from %s' % opt.load_path)
    
  best_map = 0
  lr_ = opt.lr
  record_pd = pd.read_csv("results.csv")
  for epoch in range(8,opt.epoch):
    trainer.reset_meters()
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
      scale = at.scalar(scale)
      img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
      trainer.train_step(img, bbox, label, scale)
    
    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                              str(eval_result['map']),
                                              str(trainer.get_meter_data()))
    
    dict2 = trainer.get_meter_data()
    new = dict({'lr': lr_, 'map': eval_result['map']}, **dict2)
    record_pd = record_pd.append(new, ignore_index=True)
    record_pd.to_csv("results.csv", index=0)
    print(log_info)
    
    if eval_result['map'] > best_map:
      best_map = eval_result['map']
      best_path = trainer.save(best_map=best_map)
    if epoch == 9:
      trainer.load(best_path)
      trainer.faster_rcnn.scale_lr(opt.lr_decay)
      lr_ = lr_ * opt.lr_decay


train()
