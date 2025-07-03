'''
这个文件用于读取myops20比赛的数据集合，并且利用数据进行affine变化，从而达到训练网络的目的
'''
import os
import numpy as np
import torch
from torch.utils import  data
from tools.dir import sort_glob
import SimpleITK as sitk
from dataloader.util import SkimageOP_MyoPS20
from baseclass.medicalimage import MyoPSLabelIndex
class MyoPS20DataSet(data.Dataset):
    def __init__(self, args, type, augo=True, task="myo", ret_path=True):
        self.args=args
        self.augo = augo
        self.task = task
        self.has_path=ret_path
        self.op = SkimageOP_MyoPS20()
        self.ret_path=ret_path
        self.type=type
        if type == "train":
            subjects = sort_glob(f"{args.dataset_dir}/train20/*")

        elif type == "valid":
            subjects = sort_glob(f"{args.dataset_dir}/valid5/*")

        elif type== "test":
            subjects = sort_glob(f"{args.dataset_dir}/test20/*")
        elif type=='trainall':
            subjects = sort_glob(f"{args.dataset_dir}/train25/*")
        else:
            print("unrecognized type")
            exit(-10)

        print(f"{__class__.__name__}")
        self.c0=self._getsample(subjects, "c0")
        self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        assert len(self.c0["img"]) == len(self.de["img"]) and  len(self.c0["img"]) == len(self.t2["img"])
        self.cur_index=-1
    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.upper()}_[0-9].nii.gz"))
            dict["lab"].extend(sort_glob(f"{s}/*{modality.upper()}_gd_[0-9].nii.gz"))
        print(dict)
        return dict

    def _readimglab(self, index):
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.int16)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["lab"][index])).astype(np.int16)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.int16)
        return img_c0, img_t2,  img_de,lab_c0, lab_t2,lab_de

    def _readtestimg(self,index):
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)
        return img_c0, img_t2, img_de

    def _create_label(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype('uint16')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo  2:edema 3:scar 4:其他
        myo_mask = np.zeros_like(ori_label)
        myo_mask[ori_label == MyoPSLabelIndex.myo_nn.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.scar_nn.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.edema_nn.value] = 1

        # 只有edema的信息
        edema_mask = np.zeros_like(ori_label)
        edema_mask[ori_label == MyoPSLabelIndex.edema_nn.value] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(ori_label)
        scar_mask[ori_label == MyoPSLabelIndex.scar_nn.value] = 1

        # 只有lv_p的信息
        lv_pool_mask = np.zeros_like(ori_label)
        lv_pool_mask[ori_label == MyoPSLabelIndex.lv_p_nn.value] = 1
        lv_mask=lv_pool_mask+myo_mask

        # 只有rv的信息
        rv_mask = np.zeros_like(ori_label)
        rv_mask[ori_label == MyoPSLabelIndex.rv_nn.value] = 1

        # scar 与 myo
        myo_scar_mask = np.zeros_like(ori_label)
        myo_scar_mask[ori_label == MyoPSLabelIndex.myo_nn.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.edema_nn.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.scar_nn.value] = 2

        #scar 与 myo
        myo_ede_mask = np.zeros_like(ori_label)
        myo_ede_mask[ori_label == MyoPSLabelIndex.myo_nn.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.scar_nn.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.edema_nn.value] = 2

        #scar+edem 与 myo
        myo_scar_ede_mask = np.zeros_like(ori_label)
        myo_scar_ede_mask[ori_label == MyoPSLabelIndex.myo_nn.value] = 1
        myo_scar_ede_mask[ori_label == MyoPSLabelIndex.scar_nn.value] = 2
        myo_scar_ede_mask[ori_label == MyoPSLabelIndex.edema_nn.value] = 3

        #background
        bg_mask = np.where(ori_label == 0, 1, 0)

        mmc_label = np.concatenate([bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask,myo_scar_ede_mask,ori_label], axis=0)

        mmc_label = torch.from_numpy(mmc_label).float()

        return mmc_label

    # def _create_label(self, gd_lab):
    #     ori_label = gd_lab[np.newaxis, :, :]
    #     ori_label = np.round(ori_label).astype(np.float32)
    #     mmc_label = torch.from_numpy(ori_label).float()
    #     return mmc_label

    def __len__(self):
        #  or len(self.c0["img"])==len(self.de["img"])
        return len(self.c0["img"])

    def __getitem__(self, index):
        if self.type=='test':
            return self.__get_test_item(index)
        else:
            return self.__get_train_item(index)

    def __get_train_item(self, index):
        self.cur_index = index
        img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self._readimglab(index)
        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)


        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseq_myops(img_c0, img_t2, img_de, lab_c0,
                                                                                  lab_t2, lab_de)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)
            img_t2 = self.op.gamma_correction(img_t2)
        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)
        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)
        if self.ret_path:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd, self.c0["img"][index], self.t2["img"][
                index], self.de["img"][index]
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd

    def __get_test_item(self,index):
        self.cur_index = index
        img_c0,img_t2, img_de  = self._readtestimg(index)
        img_c0=self.op.normalize_intensity_image(img_c0,(256,256),True)
        img_t2=self.op.normalize_intensity_image(img_t2,(256,256),True)
        img_de=self.op.normalize_intensity_image(img_de,(256,256),True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)
        if self.augo:
            lab_c0=np.ones_like(img_c0)
            lab_t2=np.ones_like(img_c0)
            lab_de=np.ones_like(img_c0)
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseq_myops(img_c0, img_t2, img_de, lab_c0,  lab_t2, lab_de)
        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)
        if  self.ret_path:
            return img_c0,img_t2, img_de,self.c0["img"][index],self.t2["img"][index],self.de["img"][index]
        else:
            return img_c0,img_t2, img_de