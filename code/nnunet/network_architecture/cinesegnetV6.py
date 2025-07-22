import os
import torch
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk

from nnunet.training.loss_functions.cineloss import STN
from nnunet.network_architecture.cinesegnetV3 import CineSegNetV3


class V6WithoutIMG(CineSegNetV3):

    def __init__(self, in_chs, out_chs):
        super(V6WithoutIMG, self).__init__(in_chs, out_chs)
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
        anatomys = []
        for i in range(0, t):
            motion_fields.append(self.motion_decoder(features[0], features[i]))
            anatomys.append(self.antomy_decoder(features[i]))

        anatomys = torch.stack(anatomys, dim=2)  # [bs,4,T,dim,dim]
        motion_fields = torch.stack(motion_fields, dim=2)  # [bs,2,T,dim,dim]

        # (pathology segmentation and deep supervision) * 17
        resolution = [[], [], [], []]
        for i in range(0, t):
            concat = torch.cat([motion_fields[:, 0, i:i+1, :, :], motion_fields[:, 1, i:i+1, :, :],
                                self.STN(anatomys[:, 1:2, i, :, :], motion_fields[:, :, i, :, :])], dim=1)      # [bs,4,dim,dim]
            concat = concat.unsqueeze(2)  # [bs,4,1,dim,dim]
            pathology = list(self.pathology_seg(concat))
            for m in range(4):
                resolution[m].append(pathology[m])

        # get average
        pathology_seg = []
        for i in range(4):
            stacked_tensor = torch.stack(resolution[i], dim=0)
            average_tensor = torch.mean(stacked_tensor, dim=0)
            pathology_seg.append(average_tensor)

        pathology_seg = tuple(pathology_seg)

        if self.do_ds:
            return motion_fields, anatomys, pathology_seg
        else:
            return pathology_seg[0]  # return full resolution when test

