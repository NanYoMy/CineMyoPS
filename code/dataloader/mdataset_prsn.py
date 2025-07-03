
import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch
import scipy.io as scio
import nibabel as nib
from skimage import transform,exposure

def make_one_hot(x, n): # 对输入的volume数据x，对每个像素值进行one-hot编码
    # print(x.max())
    return x
    # one_hot = np.zeros([n,x.shape[1], x.shape[2]]) # 创建one-hot编码后shape的zero张量
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         for v in range(x.shape[2]):
    #             one_hot[int(x[i, j, v]),j, v] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape
    # return one_hot


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


def load_slicer(path):
    C0_path = path
    DE_path = C0_path.replace("C0", "DE")
    T2_path = C0_path.replace("C0", "T2")
    gdpath = C0_path.replace("C0", "gd")
    p, gdname = os.path.split(gdpath);
    preadname = gdname.replace("gd", "pred")
    img_C0 = nib.load(C0_path).get_data()
    img_DE = nib.load(DE_path).get_data()
    img_T2 = nib.load(T2_path).get_data()
    img_gd = nib.load(gdpath).get_data()
    # exposure.equalize_hist()
    return img_C0,img_T2, img_DE,  img_gd, preadname, gdname


from tools.dir import sort_glob
from skimage.filters import gaussian
import  logging
class PRSNMultiModalityDataLoader(data.Dataset):

    def __init__(self, cfg, train=True, test=False,valid=False,task="multi"):

        self.test = test
        self.train = train
        self.valid = valid
        self.data_dir = sort_glob(cfg.dataset_dir + "/*")
        val_percent = cfg.validation / 100
        n_val = int(len(self.data_dir) * val_percent)
        n_train = len(self.data_dir) - n_val
        # logging.info(self.data_dir)


        if train == True:
            if valid==False:
                self.data_dir = self.data_dir[:n_train]
            else:
                self.data_dir = self.data_dir[n_train:]
        if test==True:
            self.data_dir = self.data_dir[n_train:]
        self.data_paths=[]
        for d in self.data_dir:
            self.data_paths.extend(sort_glob(d+"/*C0*"))

        # logging.info(f'numb_train:{len(self.data_paths)}')

        self.task=task



    def __getitem__(self, index):

        # path
        cur_path = self.data_paths[index]

        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)

        img_C0,img_T2, img_DE, img_mask, preadname, gdname = load_slicer(cur_path)
        # print(img_C0.dtype)
        if self.train:
            img_C0,img_T2, img_DE, img_mask = self.transform_multiseq(img_C0.astype("float"), img_T2.astype("float"), img_DE.astype("float"),img_mask.astype("float"))
            # split into patches (128*128)
            img_C0 = self.convert(img_C0.astype("float"))
            img_T2 = self.convert(img_T2.astype("float"))
            img_DE = self.convert(img_DE.astype("float"))
            # print(np.max(img_mask))
            myo_mask, scar_edema_mask, scar_mask, img_mask = self.convert(img_mask.astype("float"), mask=True)
            # print(torch.max(img_mask))
            # print(make_one_hot(img1,2).shape)
            # return img_C0, img_DE, img_T2, make_one_hot(img1,2), make_one_hot(img2,3), make_one_hot(img3,3), make_one_hot(img_mask,4)
            return img_C0, img_T2, img_DE,  myo_mask, scar_edema_mask, scar_mask, img_mask
        else:
            img_C0, img_T2,img_DE,  img_mask2 = self.transform_multiseq(img_C0.astype("float"),  img_T2.astype("float"),img_DE.astype("float"), img_mask.astype("float"), train=False)
            img_C0 = self.convert(img_C0)
            # print(torch.max(img_C0))
            img_T2 = self.convert(img_T2)
            img_DE = self.convert(img_DE)
            x,y=img_mask.shape[0],img_mask.shape[1]
            # print(x,y)
            # img_mask = transform.resize(img_mask, (128,128), order=0)
            _, _, _, img_mask = self.convert(img_mask.astype("float"), mask=True)
            # print(img_mask.shape)
            if self.valid:
                return img_C0,  img_T2, img_DE,img_mask
            else:
                return img_C0, img_T2, img_DE, img_mask, x,y,preadname, gdname

    def __len__(self):
        return len(self.data_paths)

    def convert(self, img, mask=False ):

        img = img[np.newaxis, :, :]
        if mask:
            img = img.astype('uint8')
            # myo img0
            myo_mask = np.zeros_like(img)
            myo_mask[img ==1] = 1
            myo_mask[img ==2] = 1
            myo_mask[img ==3] = 1

            # scar img2
            scar_mask = np.zeros_like(img)
            scar_mask[img == 3] = 1

            # scar edema
            scar_edema_mask = np.zeros_like(img)
            scar_edema_mask[img == 2] = 1
            scar_edema_mask[img == 3] = 1


            myo_mask = torch.from_numpy(myo_mask).float()
            scar_mask = torch.from_numpy(scar_mask).float()
            scar_edema_mask = torch.from_numpy(scar_edema_mask).float()
            img = torch.from_numpy(img).float()
            return myo_mask,scar_edema_mask, scar_mask,  img
        else:
            img = torch.from_numpy(img).float()
            return img

    def transform_multiseq(self, C0, T2, DE, mask, size=(128, 128), train=True):
        # img1=exposure.equalize_hist(img1,mask=(mask!=0))
        # img2 = exposure.equalize_hist(img2, mask=(mask != 0))
        # img3 = exposure.equalize_hist(img3, mask=(mask != 0))

        C0 = self.usm(C0)
        T2 = self.usm(T2)
        DE = self.usm(DE)
        # img1=exposure.equalize_hist(img1)
        # img2 = exposure.equalize_hist(img2)
        # img3 = exposure.equalize_hist(img3)
        C0 = transform.resize(C0, size)
        T2 = transform.resize(T2, size)
        DE = transform.resize(DE, size)
        mask = transform.resize(mask, size, order=0)

        if train==True:

            C0, T2, DE, mask = self.random_rotate(C0, T2, DE, mask)
            C0, T2, DE, mask = self.random_flip(C0, T2, DE, mask)
            C0, T2, DE, mask = self.random_transform(C0, T2, DE, mask)
            C0 = self.gamma_correction(C0)
            T2 = self.gamma_correction(T2)
            DE = self.gamma_correction(DE)
            # img1, img2, img3, mask = self.random_step(img1, img2, img3, mask)

        # img1 = self.normalization(img1.astype("float"))###########
        # img2 = self.normalization(img2.astype("float"))
        # img3 = self.normalization(img3.astype("float"))
        # img1=exposure.equalize_adapthist(img1)
        # img2 = exposure.equalize_adapthist(img2)
        # img3 = exposure.equalize_adapthist(img3)

        # img1 = self.standardization(img1.astype("float"))###########
        # img2 = self.standardization(img2.astype("float"))
        # img3 = self.standardization(img3.astype("float"))
        # img1 = self.standardization2(img1.astype("float"),mask)###########
        # img2 = self.standardization2(img2.astype("float"),mask)
        # img3 = self.standardization2(img3.astype("float"),mask)
        return C0, T2, DE, mask


    def usm(self,img):

        img = img * 1.0
        gauss_out = gaussian(img, sigma=5, multichannel=True)

        # alpha 0 - 5
        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / np.max(img)

        # 饱和处理
        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        return img_out

    def gamma_correction(self,img,rang=(7,15),prob=0.3):
        rand1 = np.random.rand()
        if rand1<prob:
            rand2=np.random.randint(7,15)/10.0
            img = exposure.adjust_gamma(img,rand2)
        return img

    def standardization2(self, data,mask):
        indices = np.where(mask > 0)
        mean = data[indices].mean()
        std = data[indices].std()
        data[indices] = (data[indices] - mean) / std
        # 其他的值保持为0
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def normalization2(self, data,mask):
        indices = np.where(mask > 0)
        # mean = data[indices].mean()
        max = data[indices].max()
        min = data[indices].min()
        range = max-min
        data[indices] = (data[indices] - min) / range
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def standardization(self, data):
        mu = np.mean(data)
        sigma = np.std(data)
        data = (data - mu) / sigma
        # print(np.max(data))
        return data

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / range
        return data

    def random_rotate(self, img1, img2, img3, mask):
        randa = np.random.randint(1, 360)
        img1 = transform.rotate(img1, randa)
        img2 = transform.rotate(img2, randa)
        img3 = transform.rotate(img3, randa)
        mask = transform.rotate(mask, randa,order=0)
        return img1, img2, img3, mask

    def random_flip(self, img1, img2, img3, mask):
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        if rand1 > 0.5:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            img3 = np.flipud(img3)
            mask = np.flipud(mask)
        if rand2 > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            img3 = np.fliplr(img3)
            mask = np.fliplr(mask)
        return img1, img2, img3, mask

    def random_crop(self, img1, img2, img3, mask, size=(128, 128)):
        shape = img1.shape
        s1 = np.random.randint(0, shape[0] - size[0])
        s2 = np.random.randint(0, shape[1] - size[1])
        img1 = img1[s1:s1 + size[0], s2:s2 + size[1]]
        img2 = img2[s1:s1 + size[0], s2:s2 + size[1]]
        img3 = img3[s1:s1 + size[0], s2:s2 + size[1]]
        mask = mask[s1:s1 + size[0], s2:s2 + size[1]]
        return img1, img2, img3, mask

    def random_step(self, img1, img2, img3, mask, size=(123, 123)):
        shape = img1.shape
        s1 = np.random.randint(0, shape[0] - size[0])
        s2 = np.random.randint(0, shape[1] - size[1])
        nimg1 = np.zeros_like(img1)
        nimg2 = np.zeros_like(img2)
        nimg3 = np.zeros_like(img2)
        nmask = np.zeros_like(mask)
        img1, img2, img3, mask = self.random_crop(img1, img2, img3, mask, size)
        nimg1[s1: s1 + size[0], s2: s2 + size[1]] = img1
        nimg2[s1: s1 + size[0], s2: s2 + size[1]] = img2
        nimg3[s1: s1 + size[0], s2: s2 + size[1]] = img3
        nmask[s1: s1 + size[0], s2: s2 + size[1]] = mask

        return nimg1, nimg2, nimg3, nmask


    def random_transform(self, img1, img2, img3, mask):
        rot = 0  # np.random.randint(0, 360)
        tra = np.random.randint(-5, 5, size=2)
        she = np.random.uniform(-0.1, 0.1)
        img1 = transform.warp(img1, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img2 = transform.warp(img2, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img3 = transform.warp(img3, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        mask = transform.warp(mask, transform.AffineTransform(translation=tra, rotation=rot, shear=she), order=0)
        return img1, img2, img3, mask