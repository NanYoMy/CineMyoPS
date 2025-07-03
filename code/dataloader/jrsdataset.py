#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
这个文件用于读取mscmr比赛的数据集合，
'''
import os
from torch.utils import data
import numpy as np
import torch

# def data_augmentation(self, img1, img2, img3, mask):
#     #
#     # img1, img2, img3, mask = self.random_rotate(img1, img2, img3, mask)
#     # img1, img2, img3, mask = self.random_flip(img1, img2, img3, mask)
#     # img1, img2, img3, mask = self.random_step(img1, img2, img3, mask)
#     rd_scale = np.random.uniform(0.9, 1.2)
#     rd_translate_x = np.random.uniform(-0.1, 0.1) * img1.shape[0]
#     rd_translate_y = np.random.uniform(-0.1, 0.1) * img1.shape[1]
#     rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
#
#     transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y), rotation=rd_rotate)
#
#     img1 = self.masked_normalize(img1.astype("float"), mask)
#     img2 = self.masked_normalize(img2.astype("float"), mask)
#     img3 = self.masked_normalize(img3.astype("float"), mask)
#     return img1, img2, img3, mask
from dataloader.util import make_numpy_one_hot, SkimageOP_Base, SkimageOP_jrs_Pathology


def myops_dataset(dirname, filter="C0"):
    result = []  # 含有filter的所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            # ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
            ext = apath.split("_")[-2]
            if ext in filter:
                result.append(apath)
    return result
import SimpleITK as sitk
def load_slicer(path):
    C0_path = path
    DE_path = C0_path.replace("C0", "DE")
    T2_path = C0_path.replace("C0", "T2")
    c0_gdpath = C0_path.replace("C0", "C0_gd")
    t2_gdpath = C0_path.replace("C0", "T2_gd")
    de_gdpath = C0_path.replace("C0", "DE_gd")


    # p, gdname = os.path.split(gdpath);
    # preadname = gdname.replace("gd", "pred")
    # img_C0 = nib.load(C0_path).get_data()
    # img_DE = nib.load(DE_path).get_data()
    # img_T2 = nib.load(T2_path).get_data()
    # img_gd = nib.load(gdpath).get_data()
    img_C0 =sitk.GetArrayFromImage(sitk.ReadImage(C0_path)).astype(np.float)
    img_DE =sitk.GetArrayFromImage(sitk.ReadImage(DE_path)).astype(np.float)
    img_T2 =sitk.GetArrayFromImage(sitk.ReadImage(T2_path)).astype(np.float)
    C0_gd =sitk.GetArrayFromImage(sitk.ReadImage(c0_gdpath)).astype(np.float)
    LGE_gd =sitk.GetArrayFromImage(sitk.ReadImage(de_gdpath)).astype(np.float)
    T2_gd =sitk.GetArrayFromImage(sitk.ReadImage(t2_gdpath)).astype(np.float)

    return img_C0, img_DE, img_T2, C0_gd,LGE_gd,T2_gd#, preadname, gdname

'''
this dataloader returns bg,myo,edema,scar
'''


class MyoPSdataset(data.Dataset):

    def __init__(self, args, train=True, task="myo"):
        self.args=args
        self.augo = train
        self.data_paths = myops_dataset(args.dataset_dir)
        self.op= SkimageOP_Base()
        self.task = task


    def __getitem__(self, index):
        # path
        cur_path = self.data_paths[index]
        # print(f'{cur_path}')
        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        img_C0, img_LGE, img_T2, img_mask = load_slicer(cur_path)
        # img_C0, img_LGE, img_T2, img_mask = self.op.normalize_image(img_C0.astype("float"), img_LGE.astype("float"), img_T2.astype("float"), img_mask.astype("float"))
        # print(img_C0.dtype)
        img_C0,  C0_gd, img_LGE,  LGE_gd, img_T2, T2_gd=self.op.aug_multiseq(img_C0, img_LGE, img_T2, img_mask)
        C0_Myo_mask, _, _, C0_gd = self.convert_lab_2_torch(C0_gd)
        LGE_Myo_mask, _, _, LGE_gd = self.convert_lab_2_torch(LGE_gd)
        T2_Myo_mask, _, _, T2_gd = self.convert_lab_2_torch(T2_gd)

        img_C0 = self.convert_img_2_torch(img_C0)
        img_LGE = self.convert_img_2_torch(img_LGE)
        img_T2 = self.convert_img_2_torch(img_T2)

        if self.task.lower()=='pathology':
            return img_C0, img_LGE, img_T2, C0_Myo_mask,LGE_Myo_mask,T2_Myo_mask,C0_gd, LGE_gd, T2_gd
        elif self.task.lower()=='myo':
            return img_C0, img_LGE, img_T2, C0_Myo_mask,LGE_Myo_mask,T2_Myo_mask
        else:
            exit(-999)

    def __len__(self):
        return len(self.data_paths)

    def convert_img_2_torch(self, img):
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img).float()
        return img

    def convert_lab_2_torch(self, gd_lab):
        myo_scar_edema_mask = gd_lab[np.newaxis, :, :]
        myo_scar_edema_mask = myo_scar_edema_mask.astype('uint8')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo 3:scare 2:edema 4:其他
        myo_mask = np.zeros_like(myo_scar_edema_mask)
        myo_mask[myo_scar_edema_mask != 0] = 1
        myo_mask[myo_scar_edema_mask == 4] = 0

        #只有edema的信息
        edema_mask = np.zeros_like(myo_scar_edema_mask)
        edema_mask[myo_scar_edema_mask == 2] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(myo_scar_edema_mask)
        scar_mask[myo_scar_edema_mask == 3] = 1

        myo_scar_edema_mask[myo_scar_edema_mask == 4] = 0

        if self.args.nb_class>1:
            myo_mask=make_numpy_one_hot(myo_mask,2)
            scar_mask=make_numpy_one_hot(scar_mask,2)
            edema_mask=make_numpy_one_hot(edema_mask,2)
            # print(np.unique(myo_scar_edema_mask))
            myo_scar_edema_mask=make_numpy_one_hot(myo_scar_edema_mask,self.args.nb_class)

        myo_mask = torch.from_numpy(myo_mask).float()
        scar_mask = torch.from_numpy(scar_mask).float()
        edema_mask = torch.from_numpy(edema_mask).float()
        myo_scar_edema_mask = torch.from_numpy(myo_scar_edema_mask).float()
        return myo_mask, scar_mask, edema_mask, myo_scar_edema_mask


    def make_one_hot(self,labels, C=2):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.data, 1)

        return target


'''
this dataloader returns bg,myo，edema_scar,scar
'''
class MyoPSdatasetV2(MyoPSdataset):
    def __init__(self,args, train=True, task="myo"):
        super().__init__(args,train,task)

    '''
    normal_myo+edema+scar,edema+scar,scar
    '''
    def convert_lab_2_torch(self, gd_lab):
        myo_scar_edema_mask = gd_lab[np.newaxis, :, :]
        myo_scar_edema_mask = myo_scar_edema_mask.astype('uint8')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo 3:scare 2:edema 4:其他
        myo_mask = np.zeros_like(myo_scar_edema_mask)
        myo_mask[myo_scar_edema_mask == 1] = 1
        myo_mask[myo_scar_edema_mask == 2] = 1
        myo_mask[myo_scar_edema_mask == 3] = 1

        #只有edema的信息
        edema_mask = np.zeros_like(myo_scar_edema_mask)
        edema_mask[myo_scar_edema_mask == 2] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(myo_scar_edema_mask)
        scar_mask[myo_scar_edema_mask == 3] = 1


        edema_scar_mask=edema_mask+scar_mask

        bg_mask=np.where(myo_mask==1,0,1)

        myo_scar_edema_mask=np.concatenate([bg_mask,myo_mask,edema_scar_mask,scar_mask],axis=0)
        # myo_scar_edema_mask[myo_scar_edema_mask == 4] = 0



        if self.args.nb_class>1:
            myo_mask=make_numpy_one_hot(myo_mask,2)
            scar_mask=make_numpy_one_hot(scar_mask,2)
            edema_mask=make_numpy_one_hot(edema_mask,2)
            # print(np.unique(myo_scar_edema_mask))
            # myo_scar_edema_mask=make_numpy_one_hot(myo_scar_edema_mask,self.args.nb_class)

        myo_mask = torch.from_numpy(myo_mask).float()
        scar_mask = torch.from_numpy(scar_mask).float()
        edema_mask = torch.from_numpy(edema_mask).float()
        myo_scar_edema_mask = torch.from_numpy(myo_scar_edema_mask).float()
        return myo_mask, scar_mask, edema_mask, myo_scar_edema_mask

'''
the pre-aligned version of MyoPsdatasetV2. this dataloader returns **pre-aligned** bg,myo，edema_scar,scar
'''
class MyoPSdataset_wo_aff(MyoPSdatasetV2):
    def __init__(self,args, train=True, task="myo"):
        super().__init__(args,train,task)

    def aug_data(self, c0, t2, lge, mask):
        c0, lge, t2, mask = self.random_rotate(c0, lge, t2, mask)
        c0, lge, t2, mask = self.random_flip(c0, lge, t2, mask)
        c0, lge, t2, mask = self.random_affine_all(c0, lge, t2, mask)
        return c0, mask, lge, mask, t2, mask


from baseclass.medicalimage import MyoPSLabelIndex
class MyoPSDataSet_MultiChannel(MyoPSdataset):
    def __init__(self,args, train=True, task="myo"):
        super().__init__(args,train,task)
        self.augo=train

    def __getitem__(self, index):
        # path
        cur_path = self.data_paths[index]
        # print(f'{cur_path}')
        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        img_C0, img_LGE, img_T2, C0_gd,LGE_gd,T2_gd = load_slicer(cur_path)
        # img_C0, img_LGE, img_T2, img_mask = self.op.normalize_image(img_C0, img_LGE, img_T2, img_mask)


        if self.augo:
            img_mask=C0_gd
            img_C0,  C0_gd, img_LGE,  LGE_gd, img_T2, T2_gd=self.op.aug_multiseq(img_C0, img_LGE, img_T2, img_mask)
        else:
            pass
        # img_mask = C0_gd
        # img_C0, C0_gd, img_LGE, LGE_gd, img_T2, T2_gd = self.op.aug_data(img_C0, img_LGE, img_T2, img_mask)

        c0_mmc_gd = self.convert_lab_2_torch(C0_gd)
        t2_mmc_gd = self.convert_lab_2_torch(T2_gd)
        lge_mmc_gd = self.convert_lab_2_torch(LGE_gd)

        img_C0 = self.convert_img_2_torch(img_C0)
        img_LGE = self.convert_img_2_torch(img_LGE)
        img_T2 = self.convert_img_2_torch(img_T2)

        return img_C0, img_T2, img_LGE, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,cur_path

    def convert_lab_2_torch(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype('uint8')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo 3:scare 2:edema 4:其他
        myo_mask = np.zeros_like(ori_label)
        myo_mask[ori_label == 1] = 1
        myo_mask[ori_label == 2] = 1
        myo_mask[ori_label == 3] = 1

        #只有edema的信息
        edema_mask = np.zeros_like(ori_label)
        edema_mask[ori_label == 2] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(ori_label)
        scar_mask[ori_label == 3] = 1

        # 只有lv_p的信息
        lv_pool_mask = np.zeros_like(ori_label)
        lv_pool_mask[ori_label == MyoPSLabelIndex.lv_p_nn.value] = 1
        lv_mask=lv_pool_mask+myo_mask

        # 只有rv的信息
        rv_mask = np.zeros_like(ori_label)
        rv_mask[ori_label == MyoPSLabelIndex.rv_nn.value] = 1

        bg_mask=np.where(ori_label==0,1,0)

        edema_scar_mask=edema_mask+scar_mask

        mmc_label=np.concatenate([bg_mask,myo_mask,edema_scar_mask,scar_mask,lv_mask,rv_mask],axis=0)


        # myo_mask = torch.from_numpy(myo_mask).float()
        # scar_mask = torch.from_numpy(scar_mask).float()
        # edema_mask = torch.from_numpy(edema_mask).float()
        mmc_label = torch.from_numpy(mmc_label).float()

        return mmc_label
from tools.dir import sort_glob
class MyoPSDataSet_MultiChannelV2(MyoPSDataSet_MultiChannel):
    def __init__(self,args, train=True, task="myo"):
        self.args=args
        self.augo = train
        self.task = task
        self.op= SkimageOP_Base()
        if train==True:
            self.data_paths = sort_glob(args.dataset_dir + "/train/*/*C0_[0-9]*")
        else :
            self.data_paths = sort_glob(args.dataset_dir + "/test/*/*C0_[0-9]*")


class MyoPSDataSet_unliagned(data.Dataset):
    def __init__(self, args, type="train", augo=True, task="myo"):
        self.args = args
        self.augo = augo
        self.task = task
        self.op = SkimageOP_Base()


        subjects=sort_glob(args.dataset_dir+"/*")

        if type == "train":
            self.train = True
            subjects=subjects[:25]

        elif type== "test":
            self.train = False
            subjects=subjects[25:]
        elif type=='all':
            self.train = False
            subjects = subjects
        else:
            self.train = False
            subjects=subjects

        # print(f"{__class__.__name__}")
        print(subjects)

        self.c0=self._getsample(subjects, "c0")
        self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        self.cur_index=-1
    def get_cur_path(self):
        return self.c0[self.cur_index],self.t2[self.cur_index],self.de[self.cur_index]

    def __getitem__(self, index):
        self.cur_index=index
        # path
        # cur_path = self.data_paths[index]
        # print(f'{cur_path}')
        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)
        # 在数据预处理的时候就进行归一化处理
        # mysize=(self.args.image_size,self.args.image_size,)
        # img_c0,lab_c0 = self.op.normalize_image(img_c0.astype("float"),lab_c0.astype("float"),size=mysize,clip=True) #对于结构的配准与分割可以考虑clip
        # img_t2,lab_t2 = self.op.normalize_image(img_t2.astype("float"),lab_t2.astype("float"),size=mysize,clip=True)
        # img_de,lab_de = self.op.normalize_image(img_de.astype("float"),lab_de.astype("float"),size=mysize,clip=True)

        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseq(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if self.train:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["lab"][index],self.t2["lab"][index],self.de["lab"][index]

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
        # print(self.t2["img"][index])
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.float)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["lab"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.float)
        return img_c0,  img_t2, img_de, lab_c0,lab_t2, lab_de

class MyoPSDataSet_aff_aligned(MyoPSDataSet_unliagned):
    def _getsample(self, subjects, type):
        dict = {"img": [], "lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*subject[0-9]*_{type.upper()}*"))
            if type=="c0":
                dict["lab"].extend(sort_glob(f"{s}/*ana_{type}*"))
            else:
                dict["lab"].extend(sort_glob(f"{s}/*ana_patho_{type}*"))
        print(dict)
        return dict

'''
large and small spatial augmentation
'''
class DataSetRJ(MyoPSDataSet_unliagned):
    def __getitem__(self, index):
        self.cur_index=index

        # print(f"current {self.cur_index}")
        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)

        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)


        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseqV3(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de,0.2)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)
            img_t2 = self.op.gamma_correction(img_t2)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if self.train:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["img"][index],self.t2["img"][index],self.de["img"][index]

from baseclass.medicalimage import MedicalImage
from dataloader.util import SkimageOP_RJ_PSN
class DataSetRJ_PSN(data.Dataset):
    def __init__(self, args, type="train", augo=True, task="myo"):
        self.args = args
        self.augo = augo
        self.task = task
        self.op = SkimageOP_RJ_PSN()

        subjects=sort_glob(args.dataset_dir+"/*[0-9]")

        if type == "train":
            self.train = True
            subjects=subjects[:25]

        elif type== "test":
            self.train = False
            subjects=subjects[25:]
        elif type=='all':
            self.train = False
            subjects = subjects
        else:
            self.train = False
            subjects=subjects

        # print(f"{__class__.__name__}")
        print(subjects)

        self.c0=self._getsample(subjects, "c0")
        self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        self.cur_index=-1
    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"prior": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["prior"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))

        return dict
    def get_cur_path(self):
        return self.c0[self.cur_index],self.t2[self.cur_index],self.de[self.cur_index]
    def _readimg(self,index):
        # print(self.t2["img"][index])
        self._check_name(self.c0,index)
        self._check_name(self.t2,index)
        self._check_name(self.de,index)
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["gt_lab"][index])).astype(np.float)
        prior_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["prior"][index])).astype(np.float)
        c0_data=MedicalImage(img_c0,prior_c0,lab_c0,path=self.c0["img"][index])


        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        prior_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["prior"][index])).astype(np.float)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["gt_lab"][index])).astype(np.float)
        t2_data=MedicalImage(img_t2,prior_t2,lab_t2,path=self.t2["img"][index])

        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)
        prior_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["prior"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["gt_lab"][index])).astype(np.float)
        de_data=MedicalImage(img_de,prior_de,lab_de,path=self.de["img"][index])


        return c0_data,t2_data ,de_data

    def _check_name(self,modality_data, index):
        p_img = os.path.basename(modality_data["img"][index])
        p_gt = os.path.basename(modality_data["gt_lab"][index])
        p_prior = os.path.basename(modality_data["prior"][index])
        assert p_img.split("_")[-1] == p_gt.split("_")[-1] and p_img.split("_")[-1] == p_prior.split("_")[-1]
        assert p_img.split("_")[-1] == p_gt.split("_")[-1] and p_img.split("_")[-1] == p_prior.split("_")[-1]

    def __getitem__(self, index):
        self.cur_index=index
        # path
        # cur_path = self.data_paths[index]
        # print(f'{cur_path}')
        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        c0_data, t2_data, de_data = self._readimg(index)

        datas = self.op.prep_normalize({"c0":c0_data,"t2":t2_data,"de":de_data},size=[self.args.image_size,self.args.image_size])

        if self.augo:
            datas = self.op.augment(datas,0.3)
            # img_c0 = self.op.gamma_correction(img_c0)
            # img_de = self.op.gamma_correction(img_de)

        for k in datas.keys():
            # print(f"processing {k}")
            datas[k].gt_lab = self._create_label(datas[k].gt_lab)
            datas[k].img = self._convert_img_2_torch(datas[k].img)
            datas[k].prior = self._convert_img_2_torch(datas[k].prior)

        return self.to_dict(datas["c0"]),self.to_dict(datas["t2"]),self.to_dict(datas["de"])

    def to_dict(self,data):
        return {"img":data.img,"prior":data.prior,"gt_lab":data.gt_lab,"path":data.path}

    def _convert_img_2_torch(self,img):
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img).float()
        return img

    def _create_label(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype('uint16')


        # scar 与 myo
        scar_edema = np.zeros_like(ori_label)

        scar_edema[ori_label == MyoPSLabelIndex.edema.value] = 1
        scar_edema[ori_label == MyoPSLabelIndex.scar.value] = 1

        ori_label = np.round(scar_edema).astype(np.float32)
        mmc_label = torch.from_numpy(ori_label).float()

        return mmc_label

    def __len__(self):
        assert len(self.c0["img"])==len(self.t2["img"])
        assert len(self.c0["img"])==len(self.de["img"])
        return len(self.c0["img"])


'''
large and mid spatial augmentation
'''

class MyoPSDataSet_unliagnedV3(MyoPSDataSet_unliagned):
    def __getitem__(self, index):
        self.cur_index=index

        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)

        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)


        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseqV3(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)
            img_t2 = self.op.gamma_correction(img_t2)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if self.train:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["img"][index],self.t2["img"][index],self.de["img"][index]



class MyoPSDataSet_aff_aligendV2(DataSetRJ):
    def _getsample(self, subjects, type):
        dict = {"img": [], "lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*subject[0-9]*_{type.upper()}*"))
            if type=="c0":
                dict["lab"].extend(sort_glob(f"{s}/*ana_{type}*"))
            else:
                dict["lab"].extend(sort_glob(f"{s}/*ana_patho_{type}*"))
        print(dict)
        return dict

class MyoPSDataSet_aligned(DataSetRJ):
    def _getsample(self, subjects, type):
        dict = {"img": [], "lab": []}
        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*subject[0-9]*_{type.upper()}_[0-9]*.nii.gz"))
            dict["lab"].extend(sort_glob(f"{s}/*subject[0-9]*_{type.upper()}_manual*.nii.gz"))

        print(dict)
        return dict


class jsr2MPathologyDataset(data.Dataset):

    def __init__(self, args, type,augo=True,  task="myo"):
        self.args=args
        self.augo = augo
        self.op= SkimageOP_jrs_Pathology()
        self.task = task
        subjects=sort_glob(f"{args.dataset_dir}/*")
        if type == "train":
            self.train = True
            subjects=subjects[:25]

        elif type== "test":
            self.train = False
            subjects=subjects[25:]
        else:
            exit(-997)

        print(f"{__class__.__name__}")
        print(subjects)
        self.c0=self._getsample(subjects, "c0")
        # self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        assert len(self.c0["img"]) == len(self.de["img"])
        self.cur_index=-1
    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],'mask':[], "lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*subject_[0-9]*_img_{modality.lower()}*"))
            dict["mask"].extend(sort_glob(f"{s}/*subject_[0-9]*_lab_{modality.lower()}_pred_*"))
            dict["lab"].extend(sort_glob(f"{s}/*subject_[0-9]*_lab_{modality.lower()}_ori_*"))
        print(dict)
        return dict

    def _readimg(self,index):
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        mask_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["mask"][index])).astype(np.float)
        mask_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["mask"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.float)
        return img_c0,   img_de,mask_c0,mask_de, lab_c0, lab_de

    def _create_label(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype(np.float32)
        mmc_label = torch.from_numpy(ori_label).float()
        return mmc_label

    def __len__(self):
        #  or len(self.c0["img"])==len(self.de["img"])
        return len(self.c0["img"])

    def __getitem__(self, index):
        self.cur_index=index

        img_c0,  img_de, mask_c0,mask_de, lab_c0, lab_de = self._readimg(index)

        img_c0 = self.op.usm(img_c0)
        img_de = self.op.usm(img_de)

        if self.augo:
            img_c0,  img_de, mask_c0,mask_de, lab_c0, lab_de = self.op.aug_multiseqV3(img_c0, img_de,mask_c0,mask_de,lab_c0, lab_de)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)

        c0_mmc_gd = self._create_label(lab_c0)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_de = self.op.convert_array_2_torch(img_de)

        mask_c0 = self.op.convert_array_2_torch(mask_c0)
        mask_de = self.op.convert_array_2_torch(mask_de)

        if self.train:
            return img_c0, img_de,mask_c0,mask_de, c0_mmc_gd,  lge_mmc_gd
        else:
            return img_c0, img_de,mask_c0,mask_de, c0_mmc_gd,  lge_mmc_gd,self.c0["lab"][index],self.de["lab"][index]



'''
large and small spatial augmentation
'''
from config.warped_c0_unet_config import WarpedC0Config
class DataSetWarpedC0RJ(MyoPSDataSet_unliagned):

    def __init__(self, args:WarpedC0Config, type="train", augo=True, task="myo"):
        self.args = args
        self.augo = augo
        self.task = task
        self.op = SkimageOP_Base()


        subjects=sort_glob(args.dataset_dir+"/*")

        if type == "train":
            self.train = True
            subjects=subjects[:25]

        elif type== "test":
            self.train = False
            subjects=subjects[25:]
        elif type=='all':
            self.train = False
            subjects = subjects
        else:
            self.train = False
            subjects=subjects

        # print(f"{__class__.__name__}")
        print(subjects)

        self.c0=self._getsample(subjects, "C0")
        self.t2=self._getsample(subjects, "T2")
        self.de=self._getsample(subjects, "DE")
        self.cur_index=-1
    def get_cur_path(self):
        return self.c0[self.cur_index],self.t2[self.cur_index],self.de[self.cur_index]

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
            dict["img"].extend(sort_glob(f"{s}/*_{type}_[0-9].nii.gz"))
            dict["lab"].extend(sort_glob(f"{s}/*_{type}_manual_[0-9].nii.gz"))

        return dict

    def _readimg(self,index):
        # print(self.t2["img"][index])
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.float)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["lab"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.float)
        return img_c0,  img_t2, img_de, lab_c0,lab_t2, lab_de


    def __getitem__(self, index):
        self.cur_index=index

        # print(f"current {self.cur_index}")
        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)

        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)


        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseqV3(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de,0.2)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)
            img_t2 = self.op.gamma_correction(img_t2)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if self.train:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["img"][index],self.t2["img"][index],self.de["img"][index]
