import glob
import os

import SimpleITK as sitk

from tools.dir import sort_glob, mk_or_cleardir, mkdir_if_not_exist
from tools.np_sitk_tools import get_bounding_box_by_ids, crop_by_bbox, binarize_img, sitkResize, sitkRespacing, \
    get_bounding_boxV2
from tools.itkdatawriter import sitk_write_image
from tools.parse import parse_arg_list
from tools.dir import mkdir_if_not_exist
import os

def extrate_myo_pathology_slices(config, input_dir, output_dir, ):
    mk_or_cleardir(output_dir)
    input_img_dir=input_dir+"/img/"
    input_lab_dir=input_dir+"/lab/"
    output_img_dir=output_dir+"/img/"
    output_lab_dir=output_dir+"/lab/"

    files = sort_glob(input_lab_dir+"/*.nii.gz")
    for p in files:
        lab = sitk.ReadImage(p)
        #先转化成统一space,保证crop的大小一致
        # lab=sitkResample3DV2(lab,sitk.sitkNearestNeighbor,[1,1,1])
        ids=parse_arg_list(config.components,"int")
        bbox=get_bounding_box_by_ids(lab,padding=10,ids=ids)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=binarize_img(crop_lab,ids)
        crop_lab=sitkResize(crop_lab, [config.img_size, config.img_size, crop_lab.GetSize()[-1]], sitk.sitkNearestNeighbor)
        for i in range(crop_lab.GetSize()[-1]):
            sitk_write_image(crop_lab[:,:,i],dir=output_lab_dir,name="%s_%d"%(os.path.basename(p).split('.')[0],i))
            img_file=glob.glob("%s/*%s*.nii.gz"%(input_img_dir,os.path.basename(p).split("_")[2]))
            img_file.sort()
            for j in img_file:
                img = sitk.ReadImage(j)
                # img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
                crop_img=crop_by_bbox(img,bbox)
                crop_img = sitkResize(crop_img, [config.img_size, config.img_size, crop_img.GetSize()[-1]], sitk.sitkLinear)
                sitk_write_image(crop_img[:,:,i], dir=output_img_dir, name="%s_%d"%(os.path.basename(j).split('.')[0],i))

def crop_img_by_label(input,output):
    '''
    :return:
    '''
    input_img_dir = input + "/img/"
    input_lab_dir = input + "/lab/"
    output_img_dir = output + '/img/'
    output_lab_dir = output + '/lab/'

    files = glob.glob(input_lab_dir+"/*.nii.gz")
    mkdir_if_not_exist(output_lab_dir)
    for i in files:
        lab = sitk.ReadImage(i)
        #先转化成统一space,保证crop的大小一致
        lab=sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, 1])
        bbox=get_bounding_boxV2(sitk.GetArrayFromImage(lab),padding=10)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=sitkResize(crop_lab, [256, 256, crop_lab.GetSize()[-1]], sitk.sitkNearestNeighbor)
        sitk_write_image(crop_lab,dir=output_lab_dir,name=os.path.basename(i))
        img_file=glob.glob("%s/*%s*.nii.gz"%(input_img_dir,os.path.basename(i).split("_")[2]))
        for j in img_file:
            img = sitk.ReadImage(j)
            img = sitkRespacing(img, sitk.sitkLinear, [1, 1, 1])
            crop_img=crop_by_bbox(img,bbox)
            crop_img = sitkResize(crop_img, [256, 256, crop_img.GetSize()[-1]], sitk.sitkLinear)
            sitk_write_image(crop_img, dir=output_img_dir, name=os.path.basename(j))

def prepare_myo_data(config):
    input_dir="../../dataset/%s"%(config.task)
    # if os.path.exists(config.dataset_dir):
    #     return
    mkdir_if_not_exist(config.dataset_dir)
    extrate_myo_pathology_slices(config,input_dir,config.dataset_dir)