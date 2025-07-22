

import torch

def to_one_hot(gt, number_class):
    shp_x = gt.shape
    shp_x=torch.tensor([shp_x[0],number_class,*shp_x[2:]])

    gt = gt.long()
    y_onehot = torch.zeros(*shp_x)
    if gt.device.type == "cuda":
        y_onehot = y_onehot.cuda(gt.device.index)
    y_onehot.scatter_(1, gt, 1)
    return y_onehot

