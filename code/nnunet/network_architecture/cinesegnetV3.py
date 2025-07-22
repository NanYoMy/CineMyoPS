import os
import torch
import numpy as np
import SimpleITK as sitk

from nnunet.training.loss_functions.cineloss import STN
from nnunet.network_architecture.cinesegnetV2 import CineSegNetV2


class CineSegNetV3(CineSegNetV2):

    def __init__(self, in_chs, out_chs):
        super(CineSegNetV3, self).__init__(in_chs, out_chs)
        self.pathology_seg = None  # [bs,2T+1,dim,dim] -> [bs,5,dim,dim] bg,myo,lv,scar,edema
        self.STN = STN()

    def forward(self, cine):
        cine = cine.permute(0, 2, 1, 3, 4)
        t = cine.shape[2]

        # motion field
        features = []
        for i in range(0, t):
            features.append(self.encoder(cine[:, :, i, :, :]))

        motion_fields = []
        for i in range(0, t):
            motion_fields.append(self.motion_decoder(features[0], features[i]))
            # motion_fields.append(self.motion_decoder(features[i], features[0]))

        anatomys = []
        for i in range(0, t):
            anatomys.append(self.antomy_decoder(features[i]))

        anatomys = torch.stack(anatomys, dim=2)  # [bs,4,T,dim,dim]
        motion_fields = torch.stack(motion_fields, dim=2)  # [bs,2,T,dim,dim]

        # pathology segmentation and deep supervision
        concat = torch.cat([cine[:, :, 0, :, :], motion_fields[:, 0, :, :, :], motion_fields[:, 1, :, :, :],
                            anatomys[:, 1:2, 0, :, :]], dim=1)  # [bs,2T+1+1,dim,dim]
        concat = concat.unsqueeze(2)    # [bs,2T+2,1,dim,dim]
        pathology_seg = self.pathology_seg(concat)  # [bs,5,dim,dim]    channel: 0:17

        if self.do_ds:
            return motion_fields, anatomys, pathology_seg
        else:   # TODO
            # global global_counter
            # save_path = f'../outputs/nnunet/raw/nnUNet_raw_data/Task021_Cine_Seg/result'    # attention
            # mf_path = save_path + '/motion_field'
            # regis_path = save_path + '/regis_img'
            # os.makedirs(mf_path, exist_ok=True)
            # os.makedirs(regis_path, exist_ok=True)
            #
            # if global_counter % 1 == 0:
            #     global_counter = int(global_counter)
            #     for i in range(t):
            #         mf_array = motion_fields[0, :, i].detach().cpu().numpy()
            #         mf_array = mf_array.transpose(2, 1, 0).astype(np.float32)   # [dim, dim, 2]
            #         mf_img = sitk.GetImageFromArray(mf_array, isVector=True)
            #         sitk.WriteImage(mf_img, f'{mf_path}/{global_counter:02d}_{i:02d}.nii.gz')
            #
            #     gd_array = cine[0, 0, 0, :, :].detach().cpu()   # [dim, dim]
            #     gd_img = sitk.GetImageFromArray(gd_array)
            #     in_array = cine[0, 0, 9, :, :].detach().cpu()
            #     in_img = sitk.GetImageFromArray(in_array)
            #     regis_array = self.STN(cine[:, :, 9, :, :], motion_fields[:, :, 9, :, :])
            #     regis_array = regis_array[0, 0].detach().cpu()
            #     regis_img = sitk.GetImageFromArray(regis_array)
            #
            #     sitk.WriteImage(gd_img, f'{regis_path}/{global_counter:02d}_gd.nii.gz')
            #     sitk.WriteImage(in_img, f'{regis_path}/{global_counter:02d}_input.nii.gz')
            #     sitk.WriteImage(regis_img, f'{regis_path}/{global_counter:02d}_regis.nii.gz')
            #
            # global_counter += 0.125     # do mirror
            return pathology_seg[0]  # return full resolution when test


class NetWithoutMF(CineSegNetV3):

    def __init__(self, in_chs, out_chs):
        super(CineSegNetV3, self).__init__(in_chs, out_chs)
        self.pathology_seg = None  # [bs,2T+1,dim,dim] -> [bs,5,dim,dim] bg,myo,lv,scar,edema
        self.STN = STN()

    def forward(self, cine):
        cine = cine.permute(0, 2, 1, 3, 4)
        t = cine.shape[2]

        # motion field
        features = []
        for i in range(0, t):
            features.append(self.encoder(cine[:, :, i, :, :]))

        # motion_fields = []
        # for i in range(0, t):
        #     motion_fields.append(self.motion_decoder(features[0], features[i]))
            # motion_fields.append(self.motion_decoder(features[i], features[0]))

        anatomys = []
        for i in range(0, t):
            anatomys.append(self.antomy_decoder(features[i]))

        anatomys = torch.stack(anatomys, dim=2)  # [bs,4,T,dim,dim]
        # motion_fields = torch.stack(motion_fields, dim=2)  # [bs,2,T,dim,dim]

        # pathology segmentation and deep supervision
        concat = torch.cat([cine[:, :, 0, :, :], anatomys[:, 0:1, 0, :, :]], dim=1)  # [bs,2,dim,dim]
        concat = concat.unsqueeze(2)    # [bs,2T+2,1,dim,dim]
        pathology_seg = self.pathology_seg(concat)  # [bs,5,dim,dim]    channel: 0:17

        if self.do_ds:
            return anatomys, pathology_seg
        else:
            return pathology_seg[0]  # return full resolution when test


class NetWithoutAS(CineSegNetV3):

    def __init__(self, in_chs, out_chs):
        super(CineSegNetV3, self).__init__(in_chs, out_chs)
        self.pathology_seg = None  # [bs,2T+1,dim,dim] -> [bs,5,dim,dim] bg,myo,lv,scar,edema
        self.STN = STN()

    def forward(self, cine):
        cine = cine.permute(0, 2, 1, 3, 4)
        t = cine.shape[2]

        # motion field
        features = []
        for i in range(0, t):
            features.append(self.encoder(cine[:, :, i, :, :]))

        motion_fields = []
        for i in range(0, t):
            motion_fields.append(self.motion_decoder(features[0], features[i]))

        anatomys = []
        for i in range(0, t):
            anatomys.append(self.antomy_decoder(features[i]))

        anatomys = torch.stack(anatomys, dim=2)  # [bs,4,T,dim,dim]
        motion_fields = torch.stack(motion_fields, dim=2)  # [bs,2,T,dim,dim]

        # pathology segmentation and deep supervision
        concat = torch.cat([cine[:, :, 0, :, :], motion_fields[:, 0, :, :, :], motion_fields[:, 1, :, :, :]], dim=1)  # [bs,2+1,dim,dim]
        concat = concat.unsqueeze(2)  # [bs,2T+2,1,dim,dim]
        pathology_seg = self.pathology_seg(concat)  # [bs,5,dim,dim]    channel: 0:17

        if self.do_ds:
            return motion_fields, anatomys, pathology_seg
        else:
            return pathology_seg[0]  # return full resolution when test

