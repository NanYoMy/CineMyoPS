import SimpleITK as sitk
import numpy as np
import torch

from tools.dir import sort_glob

from dataloader.util import SkimageOP_MSCMR
from baseclass.medicalimage import MyoPSLabelIndex

class MSCMRDataSet():
    def __init__(self, args, type="train", augo=True, task="myo",ret_path=True):
        self.args = args
        self.augo = augo
        self.task = task
        self.ret_path=ret_path
        self.op = SkimageOP_MSCMR()


        subjects=sort_glob(args.dataset_dir+"/*")

        if type == "train":
            self.train = True
            subjects=subjects[:25]

        elif type== "test" or type=='valid':
            self.train = False
            subjects=subjects[25:]
        else:
            self.train = False
            subjects=subjects

        print(f"{__class__.__name__}")
        print(subjects)

        self.c0=self._getsample(subjects, "c0")
        self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        self.cur_index=-1
    def get_cur_path(self):
        return self.c0[self.cur_index],self.t2[self.cur_index],self.de[self.cur_index]

    # def __getitem__(self, index):
    #     self.cur_index=index
    #     # path
    #     # cur_path = self.data_paths[index]
    #     # print(f'{cur_path}')
    #     # get images
    #     # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
    #     img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)
    #     # 在数据预处理的时候就进行归一化处理
    #     # mysize=(self.args.image_size,self.args.image_size,)
    #     # img_c0,lab_c0 = self.op.normalize_image(img_c0.astype("float"),lab_c0.astype("float"),size=mysize,clip=True) #对于结构的配准与分割可以考虑clip
    #     # img_t2,lab_t2 = self.op.normalize_image(img_t2.astype("float"),lab_t2.astype("float"),size=mysize,clip=True)
    #     # img_de,lab_de = self.op.normalize_image(img_de.astype("float"),lab_de.astype("float"),size=mysize,clip=True)
    #
    #     if self.augo:
    #         img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseq(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de)
    #
    #     c0_mmc_gd = self._create_label(lab_c0)
    #     t2_mmc_gd = self._create_label(lab_t2)
    #     lge_mmc_gd = self._create_label(lab_de)
    #
    #     img_c0 = self.op.convert_array_2_torch(img_c0)
    #     img_t2 = self.op.convert_array_2_torch(img_t2)
    #     img_de = self.op.convert_array_2_torch(img_de)
    #
    #     if self.train:
    #         return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
    #     else:
    #         return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["lab"][index],self.t2["lab"][index],self.de["lab"][index]

    def __getitem__(self, index):
        self.cur_index = index

        img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self._readimg(index)

        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)

        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)

        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseqV2(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de)
            img_c0 = self.op.gamma_correction(img_c0)
            img_t2 = self.op.gamma_correction(img_t2)
            img_de = self.op.gamma_correction(img_de)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if self.ret_path:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd, self.c0["img"][index], self.t2["img"][index], self.de["img"][index]
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd

    def _create_label(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype('uint16')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo 3:scare 2:edema 4:其他
        myo_mask = np.zeros_like(ori_label)
        myo_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.scar.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.edema.value] = 1

        # 只有edema的信息
        edema_mask = np.zeros_like(ori_label)
        edema_mask[ori_label == MyoPSLabelIndex.edema.value] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(ori_label)
        scar_mask[ori_label == MyoPSLabelIndex.scar.value] = 1

        # 只有lv_p的信息
        lv_pool_mask = np.zeros_like(ori_label)
        lv_pool_mask[ori_label == MyoPSLabelIndex.lv_p.value] = 1
        lv_mask=lv_pool_mask+myo_mask

        # 只有rv的信息
        rv_mask = np.zeros_like(ori_label)
        rv_mask[ori_label == MyoPSLabelIndex.rv.value] = 1

        bg_mask = np.where(ori_label == 0, 1, 0)


        # scar 与 myo
        myo_scar_mask = np.zeros_like(ori_label)
        myo_scar_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.edema.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.scar.value] = 2

        #scar 与 myo
        myo_ede_mask = np.zeros_like(ori_label)
        myo_ede_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.scar.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.edema.value] = 2

        mmc_label = np.concatenate([bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask], axis=0)

        mmc_label = torch.from_numpy(mmc_label).float()

        return mmc_label

    def __len__(self):
        assert len(self.c0["img"])==len(self.t2["img"])
        assert len(self.c0["img"])==len(self.de["img"])
        return len(self.c0["img"])

    def _getsample(self, subjects, type):
        dict = {"img": [], "lab": []}
        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*img_{type}*"))
            if type=="c0":
                dict["lab"].extend(sort_glob(f"{s}/*ana_{type}*"))
            else:
                dict["lab"].extend(sort_glob(f"{s}/*ana_patho_{type}*"))
        return dict

    def _readimg(self,index):
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.float)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["lab"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.float)
        return img_c0,  img_t2, img_de, lab_c0,lab_t2, lab_de

