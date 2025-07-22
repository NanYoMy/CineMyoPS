import torch

from kornia.filters.kernels import get_spatial_gradient_kernel2d, get_spatial_gradient_kernel3d
from kornia.filters.kernels import normalize_kernel2d
from torch import nn as nn
from torch.nn import functional as F

from tools.torch_op_util2 import GaussianSmoothing


class SpatialGradient(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.
    Return:
        torch.Tensor: the sobel edges of the input feature map.
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`
    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode: str = 'sobel',
                 order: int = 1,
                 normalized: bool = True) -> None:
        super(SpatialGradient, self).__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel2d(mode, order)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'normalized=' + str(self.normalized) + ', ' + \
            'mode=' + self.mode + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype).detach()
        kernel: torch.Tensor = tmp_kernel.unsqueeze(1).unsqueeze(1)

        # convolve input tensor with sobel kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [self.kernel.size(1) // 2,
                       self.kernel.size(1) // 2,
                       self.kernel.size(2) // 2,
                       self.kernel.size(2) // 2]
        out_channels: int = 3 if self.order == 2 else 2
        padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]
        return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)


class SpatialGradient3d(nn.Module):
    r"""Computes the first and second order volume derivative in x, y and d using a diff
    operator.
    Return:
        torch.Tensor: the spatial gradients of the input feature map.
    Shape:
        - Input: :math:`(B, C, D, H, W)`. D, H, W are spatial dimensions, gradient is calculated w.r.t to them.
        - Output: :math:`(B, C, 3, D, H, W)` or :math:`(B, C, 6, D, H, W)`
    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode: str = 'diff',
                 order: int = 1):
        super(SpatialGradient3d, self).__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'mode=' + self.mode + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 5:
            raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, d, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype).detach()
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with grad kernel
        kernel_flip: torch.Tensor = kernel.flip(-3)
        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [self.kernel.size(2) // 2,
                       self.kernel.size(2) // 2,
                       self.kernel.size(3) // 2,
                       self.kernel.size(3) // 2,
                       self.kernel.size(4) // 2,
                       self.kernel.size(4) // 2]
        out_ch: int = 6 if self.order == 2 else 3
        # return F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c).view(b, c, out_ch, d, h, w)
        res=F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c)

        return res


class GaussNGILoss(nn.Module):
    def __init__(self,sigma):
        super(GaussNGILoss, self).__init__()
        self.gradient=SpatialGradient3d()
        # kernel_size=5
        # kernel_size=(sigma*3+1)//2
        kernel_size=int(3*sigma)*2+1
        print(f'kernel_size:{kernel_size}')
        self.gauss= GaussianSmoothing(3, kernel_size, sigma, dim=3)

    def forward(self, fix,moving):
        fix_grad_img=self.gradient(fix)
        fix_smooth_grad_img=self.gauss(fix_grad_img)

        mv_grad_img=self.gradient(moving)
        mv_smooth_grad_img=self.gauss(mv_grad_img)

        nu=torch.abs(fix_smooth_grad_img*mv_smooth_grad_img)
        de=torch.sqrt(fix_smooth_grad_img*fix_smooth_grad_img)*torch.sqrt(mv_smooth_grad_img*mv_smooth_grad_img)

        return fix_grad_img,fix_smooth_grad_img,nu/de

        # grad_img_squre=torch.square(fix_smooth_grad_img)
        # de=torch.sum(grad_img_squre,dim=1,keepdim=True).repeat(1,3,1,1,1)
        # de=torch.sqrt(de)
        #
        #
        #
        # # abs_grad_img=torch.abs(grad_img)
        # # nomal=torch.sum(abs_grad_img,dim=1,keepdim=True)+0.000001
        # normlized_grad_image=torch.abs(fix_smooth_grad_img)/de
        # # nomal = filter3D(nomal,kernel=torch.ones(1,3,3,3)/27)
        # # smooth_grad_img=self.gauss(normlized_grad_image)
        # return fix_grad_img,normlized_grad_image,fix_smooth_grad_img


import numpy as np
import  SimpleITK as sitk
if __name__=="__main__":
    device=torch.device('cuda')
    img=sitk.ReadImage("../data/test/2_t2SPIR_mr_image.nii.gz")
    array=sitk.GetArrayFromImage(img).astype(np.float32)
    tensor=torch.tensor(array,device=device).float().unsqueeze_(dim=0).unsqueeze_(dim=0)
    crit=GaussNGILoss(sigma=0.6)
    crit.to(device)
    grad, gau_grad, loss_grad=crit(tensor, tensor)

    img=sitk.GetImageFromArray(array)
    sitk.WriteImage(img,'../data/test/test.nii.gz')

    img=sitk.GetImageFromArray(grad.cpu().numpy())
    sitk.WriteImage(img,'../data/test/grad.nii.gz')
    img = sitk.GetImageFromArray(loss_grad.cpu().numpy())
    sitk.WriteImage(img, '../data/test/loss_grad.nii.gz')
    img=sitk.GetImageFromArray(gau_grad.cpu().numpy())
    sitk.WriteImage(img,'../data/test/gau_grad.nii.gz')
