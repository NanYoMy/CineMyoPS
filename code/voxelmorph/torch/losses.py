import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad2D:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])


        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx


        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

def gradient_dx(fv):
    return (fv[:,:, 2:, 1:-1] - fv[:,:, :-2, 1:-1]) / 2

def gradient_dy(fv):
    return (fv[:,:, 1:-1, 2:] - fv[:,:, 1:-1, :-2]) / 2

class BendEng:
    def __init__(self, penalty='l1'):
        self.penalty = penalty


    def loss(self, y_pred):
        dFdx=gradient_dx(y_pred)
        dFdy=gradient_dy(y_pred)

        dFdxx=torch.abs(gradient_dx(dFdx))
        dFdyy=torch.abs(gradient_dy(dFdy))

        dFdxy=torch.abs(gradient_dy(dFdx))
        dFdyx=torch.abs(gradient_dx(dFdy))

        if self.penalty == 'l2':
            dFdxx = dFdxx * dFdxx
            dFdyy = dFdyy * dFdyy
            dFdxy = dFdxy * dFdxy
            dFdyx = dFdyx * dFdyx

        eng=dFdxx+dFdxy+dFdyy+dFdyx
        eng=torch.mean(eng)

        return eng

class LocalSmooth():
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        self.win = win

    def loss(self, flow):

        input_channal=flow.size()[1]
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(flow.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([input_channal, 1, *win]).to("cuda")
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        flow2=flow*flow

        flow_sum = conv_fn(flow, sum_filt, stride=stride, padding=padding,groups=input_channal)
        flow2_sum=conv_fn(flow2, sum_filt, stride=stride, padding=padding,groups=input_channal)

        win_size = np.prod(win)
        # u_I = I_sum / win_size
        # u_J = J_sum / win_size
        u_flow=flow_sum/win_size
        u2_flow=flow2_sum/win_size

        flow_var=flow2_sum+u2_flow*win_size-2*u_flow*flow_sum

        return torch.mean(flow_var)

        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # cc = cross * cross / (I_var * J_var + 1e-5)
        #
        # return -torch.mean(cc)