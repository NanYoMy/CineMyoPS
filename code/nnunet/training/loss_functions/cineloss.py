import numpy as np
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = (predict * target).sum(1)
        den = predict.sum(1) + target.sum(1)

        loss = 1 - (2 * num + 1e-10) / (den + 1e-10)

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        dice = BinaryDiceLoss()
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            total_loss += dice_loss

        return total_loss / target.shape[1]


class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def weight_function(self, target):
        mask = torch.argmax(target, dim=1)
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        weights = []
        for i in range(mask.max() + 1):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights.append(w_i)
        weights = torch.from_numpy(np.array(weights)).cuda()

        return weights

    def forward(self, predict, target):
        ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0, 2, 3))
        weights = self.weight_function(target)
        loss = weights * ce_loss

        return loss.sum()


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, predict, target):
        ce_loss = -target * torch.log(predict + 1e-10)

        return ce_loss.mean()


class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2, J2, IJ = I * I, J * J, I * J
        I_sum = torch.nn.functional.conv2d(I, filt, stride=stride, padding=padding).to(torch.float32)
        J_sum = torch.nn.functional.conv2d(J, filt, stride=stride, padding=padding).to(torch.float32)
        I2_sum = torch.nn.functional.conv2d(I2, filt, stride=stride, padding=padding).to(torch.float32)
        J2_sum = torch.nn.functional.conv2d(J2, filt, stride=stride, padding=padding).to(torch.float32)
        IJ_sum = torch.nn.functional.conv2d(IJ, filt, stride=stride, padding=padding).to(torch.float32)
        win_size = torch.prod(torch.tensor(win, dtype=torch.float32))
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def forward(self, predict, target, win=None):
        ndims = len(list(predict.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        sum_filt = torch.ones([1, 1, *win]).cuda()
        pad_no = math.floor(win[0] / 2)
        stride = [1] * ndims
        padding = [pad_no] * ndims
        I_var, J_var, cross = self.compute_local_sums(predict, target, sum_filt, stride, padding, win)

        # Clip or scale cc to avoid inf in mean
        cc = cross * cross / (I_var * J_var + 1e-6)
        return -1 * torch.mean(cc) + 1


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.dice_loss = DiceLoss()
        self.ce_loss = CELoss()

    def forward(self, predict, target):
        dice_loss = self.dice_loss(predict, target)
        ce_loss = self.ce_loss(predict, target)
        loss = ce_loss + dice_loss

        return loss


class Grad2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(Grad2d, self).__init__()
        self.penalty = penalty

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad


class Grad3d(nn.Module):
    def __init__(self, penalty='l1'):
        super(Grad3d, self).__init__()
        self.penalty = penalty

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad


class BendingEnergy2d(nn.Module):
    def __init__(self, energy_type):
        super(BendingEnergy2d, self).__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

    def gradient_dy(self, fv):
        return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy)
        else:
            norms = dTdx ** 2 + dTdy ** 2
        return torch.mean(norms) / 2.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        return torch.mean(dTdxx ** 2 + dTdyy ** 2 + 2 * dTdxy ** 2)

    def forward(self, disp):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class BendingEnergy3d(nn.Module):
    def __init__(self, energy_type):
        super(BendingEnergy3d, self).__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

    def forward(self, disp):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, seg_1, seg_2):
        consistency_loss = 1 - F.cosine_similarity(seg_1, seg_2, dim=1)

        return consistency_loss.mean()


class MotionLoss(nn.Module):
    def __init__(self):
        super(MotionLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        # self.ncc_loss = NCCLoss()
        self.STN = STN()

    def forward(self, motion_field, train_img, layers):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]

        motion_loss = 0

        for layer in layers:
            transformed_train_img = self.STN(train_img[:, :, layer, :, :], motion_field[:, :, layer, :, :])
            motion_loss += self.mse_loss(transformed_train_img, train_img[:, :, 0, :, :])

        motion_loss = motion_loss / len(layers)

        return motion_loss


class CardiacLoss(nn.Module):
    def __init__(self):
        super(CardiacLoss, self).__init__()

        self.con_loss = ConsistencyLoss()
        self.STN = STN()

    def forward(self, motion_field, cardiac_seg, layers):
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        cardiac_con_loss = 0

        for layer in layers:
            transformed_cardiac_seg = self.STN(cardiac_seg[:, :, layer, :, :], motion_field[:, :, layer, :, :])
            cardiac_con_loss += self.con_loss(transformed_cardiac_seg, cardiac_seg[:, :, 0, :, :])

        cardiac_con_loss = cardiac_con_loss / len(layers)

        return cardiac_con_loss


class CardiacLossV2(nn.Module):
    def __init__(self):
        super(CardiacLossV2, self).__init__()

        self.con_loss = ConsistencyLoss()
        self.STN = STN()

    def forward(self, motion_field, deva_seg, layers):
        cardiac_con_loss = 0

        for layer in layers:
            transformed_deva_seg = self.STN(deva_seg[:, :, layer, :, :], motion_field[:, :, layer, :, :])
            cardiac_con_loss += self.con_loss(transformed_deva_seg, deva_seg[:, :, 0, :, :])

        cardiac_con_loss = cardiac_con_loss / len(layers)

        return cardiac_con_loss


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.STN = STN()

    def forward(self, motion_field, seg, gd):
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        # cardiac_loss = self.seg_loss(seg[0][:,:,0,:,:], gd[0]) + \
        #                self.seg_loss(seg[0][:,:,9,:,:], gd[2])
        cardiac_loss = self.seg_loss(seg[0][:, :, 0, :, :], gd[0])

        # transformed_pathology_seg = self.STN(seg[1], motion_field[:,:,0,:,:])# @DWB pathology seg ??????
        # pathology_loss = self.seg_loss(transformed_pathology_seg, gd[1])
        pathology_loss = self.seg_loss(seg[1], gd[1])

        seg_loss = cardiac_loss + pathology_loss  # TODO

        return seg_loss


class SegLossV2(nn.Module):
    def __init__(self):
        super(SegLossV2, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.STN = STN()

    def forward(self, motion_field, seg, gd):
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        # cardiac_loss = self.seg_loss(seg[0][:,:,0,:,:], gd[0]) + \
        #                self.seg_loss(seg[0][:,:,9,:,:], gd[2])
        cardiac_loss = self.seg_loss(seg[:, :, 0, :, :], gd)

        # transformed_pathology_seg = self.STN(seg[1], motion_field[:,:,0,:,:])# @DWB pathology seg ??????
        # pathology_loss = self.seg_loss(transformed_pathology_seg, gd[1])

        seg_loss = cardiac_loss  # TODO

        return seg_loss


class SegLossV3(nn.Module):
    def __init__(self):
        super(SegLossV3, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.STN = STN()

    def forward(self, motion_field, seg, gd):
        seg_loss = 0

        for layer in range(1, seg.shape[2]):
            transformed_deva_seg = self.STN(seg[:, :, layer, :, :], motion_field[:, :, layer, :, :])
            seg_loss += self.seg_loss(transformed_deva_seg, gd.squeeze())

        seg_loss /= (seg.shape[2] - 1)

        return seg_loss


class SegLossV4(nn.Module):
    def __init__(self):
        super(SegLossV4, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.STN = STN()

    def forward(self, seg, gd):
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        # cardiac_loss = self.seg_loss(seg[0][:,:,0,:,:], gd[0]) + \
        #                self.seg_loss(seg[0][:,:,9,:,:], gd[2])
        seg_loss = self.seg_loss(seg.squeeze(), gd.squeeze())

        return seg_loss


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

        self.bending_energy = BendingEnergy2d('bending')

    def forward(self, motion_field):
        # motion_field [bs,2,T,dim,dim]

        smooth_loss = 0

        for f in range(motion_field.shape[2]):
            motion_field_f = motion_field[:, :, f, :, :]
            smooth_loss += self.bending_energy(motion_field_f)

        smooth_loss = smooth_loss / motion_field.shape[2]

        return smooth_loss


class CineSegLoss(nn.Module):
    def __init__(self):
        super(CineSegLoss, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLoss()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLoss()
        self.Dice_loss = DiceLoss()

    def forward(self, motion_field, train_img, seg, gd, ds_seg, ds_gd, layers, num_board):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        motion_loss = self.Motion_loss(motion_field, train_img, layers, num_board)
        cardiac_loss = self.Cardiac_loss(motion_field, seg[0], layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        seg_loss = self.Segmentation_loss(motion_field, seg, gd)
        ds_loss = 0
        for i in range(3):
            ds_loss += self.Dice_loss(ds_seg[i], ds_gd[i])

        loss = motion_loss + 2 * cardiac_loss + 1000 * smooth_loss + 5 * seg_loss + ds_loss

        return motion_loss, cardiac_loss, smooth_loss, seg_loss, ds_loss, loss


class CineSegLossV3(nn.Module):
    def __init__(self):
        super(CineSegLossV3, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLoss()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV2()
        self.Dice_loss = DiceLoss()

    def forward(self, motion_field, train_img, seg, gd, layers):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        motion_loss = self.Motion_loss(motion_field, train_img, layers)
        cardiac_loss = self.Cardiac_loss(motion_field, seg, layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        seg_loss = self.Segmentation_loss(motion_field, seg, gd)

        loss = motion_loss + 2 * cardiac_loss + 100 * smooth_loss + 5 * seg_loss

        return motion_loss, cardiac_loss, smooth_loss, seg_loss, loss


class CineSegLossV3WOCL(nn.Module):
    def __init__(self):
        super(CineSegLossV3WOCL, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLoss()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV3()
        self.Dice_loss = DiceLoss()

    def forward(self, motion_field, train_img, seg, gd, layers):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        motion_loss = self.Motion_loss(motion_field, train_img, layers)
        # cardiac_loss = self.Cardiac_loss(motion_field, seg, layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        seg_loss = self.Segmentation_loss(motion_field, seg, gd)

        loss = motion_loss + 100 * smooth_loss + 5 * seg_loss

        return motion_loss, smooth_loss, seg_loss, loss


class LossWithoutMF(nn.Module):
    def __init__(self):
        super(LossWithoutMF, self).__init__()

        # self.Motion_loss = MotionLoss()
        # self.Cardiac_loss = CardiacLoss()
        # self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV3()
        self.Dice_loss = DiceLoss()

    def forward(self, seg, gd):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        # motion_loss = self.Motion_loss(motion_field, train_img, layers)
        # cardiac_loss = self.Cardiac_loss(motion_field, seg, layers)  # ED
        # smooth_loss = self.Smooth_loss(motion_field)
        seg_loss = self.Segmentation_loss(0, seg, gd)

        # loss = motion_loss + 2 * cardiac_loss + 1000 * smooth_loss + 5 * seg_loss
        loss = 5 * seg_loss

        return seg_loss, loss


class LossOnlyMF(nn.Module):
    def __init__(self):
        super(LossOnlyMF, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLoss()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV3()
        self.Dice_loss = DiceLoss()

    def forward(self, motion_field, train_img, layers):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        motion_loss = self.Motion_loss(motion_field, train_img, layers)
        # cardiac_loss = self.Cardiac_loss(motion_field, seg, layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        # seg_loss = self.Segmentation_loss(motion_field, seg, gd)

        loss = motion_loss + 100 * smooth_loss

        return motion_loss, smooth_loss, loss


class EDSegLoss(nn.Module):
    def __init__(self):
        super(EDSegLoss, self).__init__()

        self.Segmentation_loss = SegmentationLoss()
        self.Dice_loss = DiceLoss()

    def forward(self, pred, gd, ds_seg, ds_gd, weight):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        loss = self.Segmentation_loss(pred, gd) * weight[0]
        ds_loss = 0
        for i in range(3):
            ds_loss += weight[i + 1] * self.Segmentation_loss(ds_seg[i], ds_gd[i])

        return ds_loss, loss


class CineAnaLoss(nn.Module):
    def __init__(self):
        super(CineAnaLoss, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLossV2()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV2()
        self.Dice_loss = DiceLoss()

    def forward(self, motion_field, train_img, seg, layers):
        # motion_field [bs,2,T,dim,dim]
        # train_img [bs,1,T,dim,dim]
        # seg [cardiac_seg, pathology_seg] [bs,4,T,dim,dim],[bs,5,dim,dim]
        # gd [ED_gd, PA_gd, ES_gd] [bs,4,dim,dim],[bs,5,dim,dim],[bs,4,dim,dim]

        motion_loss = self.Motion_loss(motion_field, train_img, layers)
        cardiac_loss = self.Cardiac_loss(motion_field, seg, layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        # seg_loss = self.Segmentation_loss(motion_field, seg, gd)

        loss = motion_loss + 100 * smooth_loss + cardiac_loss

        return motion_loss, cardiac_loss, smooth_loss, loss


class CineAnaLossV2(nn.Module):
    def __init__(self):
        super(CineAnaLossV2, self).__init__()

        self.Motion_loss = MotionLoss()
        self.Cardiac_loss = CardiacLossV2()
        self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV3()

    def forward(self, motion_field, train_img, gd, deva_res, layers):
        """
               motion_field : B * 2 * T * H * W
               train_img    : B * 1 * T * H * W
               gd           : B * num_objects * 1 * H * W
               unet_res     : B * num_objects * T * H * W
               deva_res     : B * num_objects * T * H * W
               layers       : random numbers
        """
        motion_loss = self.Motion_loss(motion_field, train_img, layers)
        # cardiac_loss = self.Cardiac_loss(motion_field, deva_res, layers)  # ED
        smooth_loss = self.Smooth_loss(motion_field)
        seg_loss = self.Segmentation_loss(motion_field, deva_res, gd)

        # loss = motion_loss + 100 * smooth_loss + cardiac_loss + 5 * seg_loss
        loss = motion_loss + 100 * smooth_loss + 5 * seg_loss

        # return motion_loss, cardiac_loss, smooth_loss, seg_loss, loss
        return motion_loss, smooth_loss, seg_loss, loss


class CineAnaLossV3(nn.Module):  # don't
    def __init__(self):
        super(CineAnaLossV3, self).__init__()

        # self.Motion_loss = MotionLoss()
        # self.Cardiac_loss = CardiacLossV2()
        # self.Smooth_loss = SmoothLoss()
        self.Segmentation_loss = SegLossV4()

    def forward(self, masks, gd, loss_weights):
        """
               masks        : B * 1 * num_objects * H * W
               gd           : B * num_objects * 1 * H * W
        """
        # motion_loss = self.Motion_loss(motion_field, train_img, layers)
        # cardiac_loss = self.Cardiac_loss(motion_field, deva_res, layers)  # ED
        # smooth_loss = self.Smooth_loss(motion_field)

        loss = 0
        for i in range(len(masks)):
            seg_loss = self.Segmentation_loss(masks[i], gd)
            # loss += seg_loss * loss_weights[len(masks) - i - 1]
            loss += seg_loss * loss_weights[i]

        return loss


class STN(object):
    def __init__(self, mode='bilinear', isCUDA=True):
        self.mode = mode
        self.isCUDA = isCUDA
        # when input is 5D the mode='bilinear' is used as trilinear

    def __call__(self, source, offset):
        # source: NCHW
        # grid: NHW2

        x_shape = source.size()
        grid_w, grid_h = torch.meshgrid(
            [torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (w, h)

        if self.isCUDA == True:
            grid_w = grid_w.float().cuda()
            grid_h = grid_h.float().cuda()
        else:
            grid_w = grid_w.float()
            grid_h = grid_h.float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        offset_w, offset_h = torch.split(offset, 1, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, w, h)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, w, h)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        grid = torch.stack((offset_h, offset_w), 3)  # should have the same order as offset

        out = F.grid_sample(source, grid, align_corners=True, mode=self.mode)

        return out


class BootstrappedCE(nn.Module):
    def __init__(self, top_p=0.3):
        super().__init__()

        self.start_warm = 2500
        self.end_warm = 8000
        self.top_p = top_p

    def forward(self, input, target, it) -> (torch.Tensor, float):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) /
                                                      (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self):
        super().__init__()
        self.bce = BootstrappedCE()

    def compute(self, data, num_objects, it) -> Dict[str, torch.Tensor]:
        losses = defaultdict(int)

        b, t = data['img'].shape[0], data['img'].shape[2]

        losses['XMem_Loss'] = 0
        losses['XMem_Dice_Loss'] = 0
        for ti in range(1, t):
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1],
                                   data['cls_gt'][bi:bi + 1, ti, 0], it)

                aux_loss = F.cross_entropy(
                    data[f'aux_logits_{ti}'][bi:bi + 1, :num_objects[bi] + 1, 0],
                    data['cls_gt'][bi:bi + 1, ti, 0])

                losses['p'] += p / b / (t - 1)
                losses[f'ce_loss_{ti}'] += loss / b
                losses[f'aux_loss_{ti}'] += aux_loss / b

            losses['XMem_Loss'] += losses['ce_loss_%d' % ti]
            losses['XMem_Loss'] += losses['aux_loss_%d' % ti] * 0.1
            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:, ti, 0])
            losses['XMem_Dice_Loss'] += losses[f'dice_loss_{ti}']
            losses['XMem_Loss'] += losses[f'dice_loss_{ti}']

        losses['XMem_Dice_Loss'] /= t
        # losses['XMem_Loss'] /= t
        return losses


def dice_loss(input_mask, cls_gt) -> torch.Tensor:
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:, i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt == (i + 1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()
