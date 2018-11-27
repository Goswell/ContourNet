# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageDraw

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    box_b = box_b[:,:64]
    num_object = box_a.size(0)
    all_sub = torch.FloatTensor(num_object, 23280)
    for i in range(num_object):
        object = box_a[i,:]
        sub_obj = box_b.sub(object)
        all_sub[i,:] = torch.sum(sub_obj.abs(), 1)

    return all_sub.cuda()  # [A,B]



def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].         [n,4 ]
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].     [n]
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.          [16,8732,4]
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.    [16,8732]
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index

    overlaps = jaccard(
        truths,
        priors
    )   #[n,8732]
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.min(1, keepdim=True) #[n,1] [n,1]
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.min(0, keepdim=True) #[1,8732] [1,8732]
    best_truth_idx.squeeze_(0) #8732
    best_truth_overlap.squeeze_(0) #8732
    best_prior_idx.squeeze_(1) #n
    best_prior_overlap.squeeze_(1) #n
    best_truth_overlap.index_fill_(0, best_prior_idx, 0)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    #print ('best_truth_ind', torch.min(best_truth_idx), torch.max(best_truth_idx))
    #best_truth_idx = best_truth_idx.cuda()
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]         # Shape: [num_priors]

    best_truth_overlap_comp = best_truth_overlap > 2.0
    #best_truth_overlap_comp = best_truth_overlap_comp.cuda()
    conf[best_truth_overlap_comp] = 0  # label as background
    loc = encode(matches, priors, variances) # [num_priors,4]
    #print ('loc', loc.size())
    #print ('loct', loc_t[idx].size())
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center

    scale = torch.unsqueeze(priors[:,64],1).expand(-1,64)
    g = (matched[:] - priors[:,:64]) / scale

    # return target for smooth_l1_loss
    return g  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    scale = torch.unsqueeze(priors[:, 64], 1).expand(-1, 64)
    boxes = loc[:] * scale + priors[:,:64]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    #x_max = torch.FloatTensor([x_max])
    #x_max = Variable(x_max, requires_grad=False)
    #x_max = x_max.cuda()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long() #[2]
    if boxes.numel() == 0:
        return keep


    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals


    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view


        IoU = idx.new(idx.size(0)).zero_().float()
        boxes_temp = boxes[i,:]
        for j in range(idx.numel()):
                iou = CalulteIou(boxes_temp,boxes[idx[j],:])
                IoU[j] = iou

        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def CalulteIou(topBox, box):
    topBox = torch.clamp(topBox, min = 0.0, max = 1.0)
    box = torch.clamp(box, min = 0.0, max = 1.0)
    topBox = topBox.cpu()
    box = box.cpu()
    topBox = topBox.numpy().flatten() * 300
    topBox = np.around(topBox).tolist()
    box = box.numpy().flatten() * 300
    box = np.around(box).tolist()

    width = 300
    height = 300

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(topBox, outline=1, fill=1)
    mask1 = np.array(img)
    ImageDraw.Draw(img).polygon(box, outline=1, fill=1)
    mask2 = np.array(img)

    jiao = np.logical_and(mask1,mask2)
    bin = np.logical_or(mask1,mask2)
    iou = np.count_nonzero(jiao)/np.count_nonzero(bin)

    return iou