import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from .hzy import loadmat
import random

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __call__(self, target,num_obj):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in range(num_obj):
            '''
            width = bbox[obj][2]-bbox[obj][0] + 1
            height = bbox[obj][3] - bbox[obj][1] + 1
            for i in range(64):
                if i % 2 == 0 :
                    target[obj][i] = target[obj][i] / width
                else:
                    target[obj][i] = target[obj][i] / height
                    '''
                # scale height or width
            res += [target[obj]]  # [x1,y1,x2,y2,...]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCSegmentation(data.Dataset):

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform # SSDAugmentation
        self.target_transform = target_transform # Annotation_Transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()          # ('/home/zeyih/data/VOCdevkit/VOC2012', '2011_003276')
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Segmentation', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        self.AllImg = loadmat('/home/zeyih/project/ssd.pytorch-master/mat/allImgCollect.mat')



    def __getitem__(self, index):
        im, gt,  h, w = self.pull_item(index)

        return im, gt, h, w

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        if index > 3326:
            index = random.randint(1,3326)

        img_id = self.ids[index]

        #a = self.ids[3000]
        #a1 = self.AllImg['allimg_collect'][3000]
        #b = self.ids[3325]
        #b1 = self.AllImg['allimg_collect'][3325]


        contour = []
        labels = []
        #contour_bboxSize = []
        num_obj = len(self.AllImg['allimg_collect'][index]) - 1
        for i in range(1,num_obj + 1):
            contour.append( self.AllImg['allimg_collect'][index][i]['normal_contour'] )
            labels.append( self.AllImg['allimg_collect'][index][i]['classnum'])
            #contour_bboxSize.append( self.AllImg['img_collect'][index][i]['bbox'] )

        #target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id) # img type?
        height, width, channels = img.shape

        if self.target_transform is not None:
            contour = self.target_transform(contour, num_obj)

        if self.transform is not None:
            #contour = contour[:,:64]
            contour = np.array(contour) #[n,64]
            labels = np.array(labels)
            #img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = self.transform(img)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            ## img = img.transpose(2, 0, 1)
            #target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            contour = np.hstack((contour, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), contour, height, width

    def pull_item1(self, index):
        img_id = self.ids[index]

        target = [1,1,1]
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.transform is not None:
            target = np.array(target)
            img = self.transform(img)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            #target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
