from __future__ import absolute_import
from pprint import pprint
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
from utils.config import opt

def inverse_normalize(img):
  if opt.caffe_pretrain:
    img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    return img[::-1, :, :]
  # approximate un-normalize for visualize
  return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
  """
  https://github.com/pytorch/vision/issues/223
  return appr -1~1 RGB
  """
  normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
  img = normalize(t.from_numpy(img))
  return img.numpy()


def caffe_normalize(img):
  """
  return appr -125-125 BGR
  """
  img = img[[2, 1, 0], :, :]  # RGB-BGR
  img = img * 255
  mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
  img = (img - mean).astype(np.float32, copy=True)
  return img


def preprocess(img, min_size=600, max_size=1000):
  """Preprocess an image for feature extraction.

  The length of the shorter edge is scaled to :obj:`self.min_size`.
  After the scaling, if the length of the longer edge is longer than
  :param min_size:
  :obj:`self.max_size`, the image is scaled to fit the longer edge
  to :obj:`self.max_size`.

  After resizing the image, the image is subtracted by a mean image value
  :obj:`self.mean`.

  Args:
      img (~numpy.ndarray): An image. This is in CHW and RGB format.
          The range of its value is :math:`[0, 255]`.

  Returns:
      ~numpy.ndarray: A preprocessed image.

  """
  C, H, W = img.shape
  scale1 = min_size / min(H, W)
  scale2 = max_size / max(H, W)
  scale = min(scale1, scale2)
  img = img / 255.
  img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
  # both the longer and shorter should be less than
  # max_size and min_size
  if opt.caffe_pretrain:
    normalize = caffe_normalize
  else:
    normalize = pytorch_normalze
  return normalize(img)


class Transform(object):
  def __init__(self, min_size=600, max_size=1000):
    self.min_size = min_size
    self.max_size = max_size
  
  def __call__(self, in_data):
    img, bbox, label = in_data
    _, H, W = img.shape
    img = preprocess(img, self.min_size, self.max_size)
    _, o_H, o_W = img.shape
    scale = o_H / H
    bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
    
    # horizontally flip
    img, params = util.random_flip(
      img, x_random=True, return_param=True)
    bbox = util.flip_bbox(
      bbox, (o_H, o_W), x_flip=params['x_flip'])
    
    return img, bbox, label, scale


class VOCBboxDataset:
  def __init__(self, data_dir, split='trainval',
               use_difficult=False, return_difficult=False,
               ):
    
    all_ann = []
    images = []
    for path in data_dir:
      id_list_file = os.path.join(
        path, 'ImageSets/Main/{0}.txt'.format(split))
      print(id_list_file)
      with open(id_list_file) as f:
        ids = f.read().splitlines()
      
      all_ann.extend([ET.parse(os.path.join(path, 'Annotations', id + '.xml')) for id in ids])
      images.extend([os.path.join(path, 'JPEGImages', id + '.jpg') for id in ids])
    
    self.all_ann = all_ann
    self.images = images
    self.data_dir = data_dir
    self.use_difficult = use_difficult
    self.return_difficult = return_difficult
    self.label_names = VOC_BBOX_LABEL_NAMES
    del images, all_ann
  
  def __len__(self):
    return len(self.images)
  
  def get_example(self, i):
    """Returns the i-th example.

    Returns a color image and bounding boxes. The image is in CHW format.
    The returned image is RGB.

    Args:
        i (int): The index of the example.

    Returns:
        tuple of an image and bounding boxes

    """
    anno = self.all_ann[i]
    bbox = list()
    label = list()
    difficult = list()
    for obj in anno.findall('object'):
      # when in not using difficult split, and the object is
      # difficult, skipt it.
      if not self.use_difficult and int(obj.find('difficult').text) == 1:
        continue
      
      difficult.append(int(obj.find('difficult').text))
      bndbox_anno = obj.find('bndbox')
      # subtract 1 to make pixel indexes 0-based
      bbox.append([
        int(bndbox_anno.find(tag).text) - 1
        for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
      name = obj.find('name').text.lower().strip()
      label.append(VOC_BBOX_LABEL_NAMES.index(name))
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    # When `use_difficult==False`, all elements in `difficult` are False.
    difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
    
    # Load a image
    img_file = self.images[i]
    img = util.read_image(img_file, color=True)
    
    # if self.return_difficult:
    #     return img, bbox, label, difficult
    return img, bbox, label, difficult
  
  __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
  'aeroplane',
  'bicycle',
  'bird',
  'boat',
  'bottle',
  'bus',
  'car',
  'cat',
  'chair',
  'cow',
  'diningtable',
  'dog',
  'horse',
  'motorbike',
  'person',
  'pottedplant',
  'sheep',
  'sofa',
  'train',
  'tvmonitor')


class Dataset:
  def __init__(self, voc_data_dir, size):
    self.db = VOCBboxDataset(voc_data_dir)
    min_size, max_size = size
    self.tsf = Transform(min_size, max_size)
  
  def __getitem__(self, idx):
    ori_img, bbox, label, difficult = self.db.get_example(idx)
    
    img, bbox, label, scale = self.tsf((ori_img, bbox, label))
    # TODO: check whose stride is negative to fix this instead copy all
    # some of the strides of a given numpy array are negative.
    return img.copy(), bbox.copy(), label.copy(), scale
  
  def __len__(self):
    return len(self.db)


class TestDataset:
  def __init__(self, voc_data_dir, split='test', use_difficult=True):
    self.db = VOCBboxDataset(voc_data_dir, split=split, use_difficult=use_difficult)
  
  def __getitem__(self, idx):
    ori_img, bbox, label, difficult = self.db.get_example(idx)
    img = preprocess(ori_img)
    return img, ori_img.shape[1:], bbox, label, difficult
  
  def __len__(self):
    return len(self.db)


# if __name__=="__main__":
#   from tqdm import tqdm
#   from torch.utils.data import DataLoader
#
#   # dataset = Dataset(voc_data_dir=['/dataset/VOCdevkit/VOC2007','/dataset/VOCdevkit/VOC2012'],size=(600,1000))
#   # count=0
#   # dataloader =DataLoader(dataset,batch_size=1,shuffle=True)#,  pin_memory=True,num_workers=opt.num_workers)
#   # for ii, (img, bbox_, label_,scale) in tqdm(enumerate(dataloader)):
#   #   count+=1
#   # print(count)
#
#   dataset = TestDataset(voc_data_dir=['/dataset/VOCdevkit/VOC2007'], )
#   count = 0
#   dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # ,  pin_memory=True,num_workers=opt.num_workers)
#   for ii, (img, size, bbox, label, difficult) in tqdm(enumerate(dataloader)):
#     count += 1
#   print(count)


