
import SimpleITK as sitk
from enum import Enum
class DataInfo(Enum):
    version=0
    img=1
    lab=2
    img_path=3
    lab_path=4
    C0_path=5
    T2_path=6
    LGE_path=7
    C0_img = 8
    T2_img = 9
    LGE_img = 10
    C0_lab = 11
    T2_lab = 12
    LGE_lab = 13
class MyoPSLabelIndex(Enum):
    rv=600
    rv_nn=6
    lv_p=500
    lv_p_nn=5

    myo=200
    myo_nn=1
    scar=2221
    scar_nn=3
    edema=1220
    edema_nn=2

class Modality(Enum):
    c0="c0"
    t2="t2"
    de="de"

class ImgType(Enum):
    img='img'
    lab="gt_lab"
    prior="prior"
    pred='pred'


class MedicalData():
    def __int__(self):
        self.data = {}

    def set_data(self,key,value):
        self.data[key]=value

    def get_data(self,key):
        return self.data[key]

class MedicalImage():

    def __init__(self,img,prior,gt_lab,pred=None,path=None):
        self.img=img
        self.prior=prior
        self.gt_lab=gt_lab
        self.pred=pred
        self.path=path


class MyoPS20MultiModalityImage(MedicalImage):

    def __init__(self, c0_path_img,t2_path_img,lge_path_img, path_lab):
        self.data={}
        self.data[DataInfo.C0_path]=c0_path_img
        self.data[DataInfo.C0_img]=sitk.ReadImage(c0_path_img)

        self.data[DataInfo.T2_path] = t2_path_img
        self.data[DataInfo.T2_img] = sitk.ReadImage(t2_path_img)

        self.data[DataInfo.LGE_path] = lge_path_img
        self.data[DataInfo.LGE_img] = sitk.ReadImage(lge_path_img)

        self.data[DataInfo.lab_path]=path_lab
        self.data[DataInfo.lab]=sitk.ReadImage(path_lab)




