from .data_seg import VOCSegmentation, VOC_CLASSES, detection_collate, AnnotationTransform
from .config import *
from .hzy import loadmat
import cv2
import numpy as np

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean)
