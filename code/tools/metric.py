
import numpy as np
smooth = 0.01
from medpy.metric import hd95,assd
'''
0/1的二值化mask可以用这个求解
'''
def calculate_binary_dice(y_true, y_pred,thres=0.5):
    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_true=np.where(y_true>thres,1,0)
    y_pred=np.where(y_pred>thres,1,0)
    return  dc(y_pred,y_true)

def calculate_binary_hd(y_true, y_pred,thres=0.5,spacing=[1,1,1]):
    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_true=np.where(y_true>thres,1,0)
    y_pred=np.where(y_pred>thres,1,0)
    return  assd(y_pred,y_true,spacing)
def dice_compute(groundtruth,pred,labs ):           #batchsize*channel*W*W
    dice=[]
    # for i in [1]:
    for i in labs:
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32))/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]
    if dice[0]>1:
        print("error!!!! dice >1 ")
    return np.array(dice,dtype=np.float32)


def dice_multi_class(gd,pred,nb_class=3):
    gd=gd.astype(np.int16)
    pred=pred.astype(np.int16)
    lables=np.unique(gd)
    res=[0.0]*nb_class
    for lab in lables:
        tmp_gd=np.where(gd==lab,1,0)
        tmp_pred=np.where(pred==lab,1,0)
        res[lab]=(dc(tmp_gd,tmp_pred))
    return res

import SimpleITK as sitk

def neg_jac(flow):
    flow_img = sitk.GetImageFromArray(flow, isVector=True)
    jac_det_filt = sitk.DisplacementFieldJacobianDeterminant(flow_img)
    jac_det = sitk.GetArrayFromImage(jac_det_filt)
    mean_grad_detJ = np.mean(np.abs(np.gradient(jac_det)))
    negative_detJ = np.sum((jac_det < 0))
    return jac_det,mean_grad_detJ,negative_detJ

def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality

from medpy.metric import dc,hd
def dice_and_asd(target_lab, predict_lab, spacing=[1, 1, 1]):
    dice=calculate_binary_dice(target_lab,predict_lab)
    haus=calculate_binary_hd(target_lab,predict_lab,spacing=spacing)
    return dice,haus

from tools.variablename import retrieve_name

def print_mean_and_std(array, info="info",detail=True):
    print("=====%s===="%retrieve_name(array))
    if detail:
        print(array)
    else:
        print(len(array))
    print("mean:%f"%np.mean(array))
    print("std:%f"%np.std(array))

from tools.ttest import ttest_alt
def cal_different_ttest(a,b):
    diff=np.mean(a)-np.mean(b)
    tt,tp=ttest_alt(a,b)
    return diff,tp


#image metric
def sad(x, y):
    """Sum of Absolute Differences (SAD) between two images."""
    return np.sum(np.abs(x - y))


def ssd(x, y):
    """Sum of Squared Differences (SSD) between two images."""
    return np.sum((x - y) ** 2)


def ncc(x, y):
    """Normalized Cross Correlation (NCC) between two images."""
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())


def mi(x, y):
    """Mutual Information (MI) between two images."""
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(x.ravel(), y.ravel())


from tools.excel import read_excel,write_array
def cross_validate(args):
    dict=read_excel(args.res_excel)
    stastic_all(args,dict, '_DS')
    stastic_all(args,dict, '_HD')

def stastic_all(args, dict, type):
    all_fold = []
    for id in range(1, 5):
        list=dict.get((args.MOLD_ID_TEMPLATE + type).replace("#", str(id)))
        if not list is None:
            all_fold = all_fold + list
    write_array(args.res_excel, args.MOLD_ID_TEMPLATE + type, [np.mean(all_fold), np.std(all_fold)])


from medpy.metric import dc
def dice_multi_label(logits, targets, class_index):
    logits=np.resize(logits,-1)
    targets=np.resize(targets,-1)
    x=np.zeros_like(logits)
    y = np.zeros_like(targets)
    for i in class_index[0]:
        x[logits==i]=1
    for i in class_index[1]:
        y[targets == i] = 1
    dice=dc(x,y)
    # inter = np.sum(x * y)
    # union = np.sum(x) + np.sum(y)
    # dice = (2. * inter + 0.00001) / (union + 1)
    return dice