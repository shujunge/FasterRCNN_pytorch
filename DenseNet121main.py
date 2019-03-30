from __future__ import  absolute_import
import os
import pandas as pd
from tqdm import tqdm
from utils.eval_tool import eval_detection_voc
from torch.utils import data as data_
from data.VOCdataset import Dataset, TestDataset, inverse_normalize
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.config import opt
from model.faster_rcnn_densenet121 import FasterRCNNDensenet121

results_path="DenseNet121.csv"
weights_path="DenseNet121_weight/"





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
  
  faster_rcnn = FasterRCNNDensenet121()
  print('model construct completed')
  trainer = FasterRCNNTrainer(faster_rcnn).cuda()
  if opt.load_path:
    trainer.load(opt.load_path)
    print('load pretrained model from %s' % opt.load_path)
    
  print("trainer lr:",trainer.faster_rcnn.optimizer.param_groups[0]['lr'])
  best_map = 0
  lr_ = opt.lr
  record_pd = pd.DataFrame(
    columns=['lr', 'map', 'rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss'])
  for epoch in range(opt.epoch):
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
    record_pd.to_csv(results_path, index=0)
    print(log_info)
  
    if eval_result['map'] > best_map:
      best_map = eval_result['map']
      best_path = trainer.save(best_map=best_map,save_path=os.path.join(weights_path,"weights_%s.pth"%round(best_map,3)))
    if epoch == 9:
      trainer.load(best_path)
      trainer.faster_rcnn.scale_lr(opt.lr_decay)
      lr_ = lr_ * opt.lr_decay


train()