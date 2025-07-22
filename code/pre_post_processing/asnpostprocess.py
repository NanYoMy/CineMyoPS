import os

import SimpleITK as sitk

from tools.dir import sort_glob, mkdir_if_not_exist
from tools.itkdatawriter import sitk_write_multi_lab
from tools.np_sitk_tools import merge_dir
from tools.sitkPasteTool import paste_roi_image

'''
合并分割结果，并可以把分割结果重新转换到原始的图像空间
'''


def merge_slice_to_ref(slice_dir,refdir,outdir):
    subject_dir=sort_glob(f"{slice_dir}/*/")
    mkdir_if_not_exist(outdir)
    for sub in (subject_dir):
        name=sub.split('\\')[-2]
        for type in ['C0','DE','T2']:
            ref=f'{refdir}/{name}_{type}.nii.gz'
            parameter=sitk.ReadImage(ref)
            array=merge_dir(sort_glob(f'{sub}/*{type}*lab*'))
            sitk_write_multi_lab(array,parameter,outdir,os.path.basename(ref))


def paste_to_ref(crop_dir,ref_dir,out_dir):
    mkdir_if_not_exist(out_dir)
    for type in ['C0',"DE","T2"]:
        subject_crop=sort_glob(f"{crop_dir}/*{type}*.nii.gz")
        subject_ori=sort_glob(f"{ref_dir}/*_{type}.nii.gz")
        for s_crop,s_ori in zip(subject_crop,subject_ori):
            crop=sitk.ReadImage(s_crop)
            ori=sitk.ReadImage(s_ori)
            ori=sitk.Cast(ori,crop.GetPixelID())
            pasted_img=paste_roi_image(ori,crop)
            sitk_write_multi_lab(pasted_img,dir=out_dir,name=os.path.basename(s_crop))


if __name__=='__main__':

    #手工或者利用labelcrop 出来的数据
    crop_ref= r"E:\consistent_workspace\myops20\data\gen_myops20\valid5_croped"
    #网络输出的数据
    input=r"E:\consistent_workspace\myops20\outputs\MyoPS20Consis3MROISegConfig_tps_4_0.1\gen_res"
    # 合成的3D-crop数据
    merge_out= r"E:\consistent_workspace\myops20\outputs\MyoPS20Consis3MROISegConfig_tps_4_0.1\merge"

    merge_slice_to_ref(input, crop_ref, merge_out)

    #原始数据
    paste_ref=r"E:\consistent_workspace\myops20\data\myops\valid5"
    #最终的输出结果
    paste_out=r"E:\consistent_workspace\myops20\outputs\MyoPS20Consis3MROISegConfig_tps_4_0.1\paste"
    paste_to_ref(merge_out, paste_ref, paste_out)

