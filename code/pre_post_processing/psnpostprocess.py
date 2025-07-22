import SimpleITK as sitk
import numpy as np

from tools.dir import sort_glob, mkdir_if_not_exist
from tools.itkdatawriter import sitk_write_multi_lab
from tools.np_sitk_tools import merge_dir
from tools.sitkPasteTool import paste_roi_image

'''
合并分割结果，并可以把分割结果重新转换到原始的图像空间
'''


def merge_slice_to_ref(slice_dir,refdir,outdir):

    mkdir_if_not_exist(outdir)
    for sub in range(201, 221):
        scar_slices=sort_glob(f"{slice_dir}/scar/*{sub}*")
        edema_slices=sort_glob(f"{slice_dir}/edemascar/*{sub}*")


        assert  len(scar_slices)==len(edema_slices)
        scar_3d=merge_dir(scar_slices)
        edema_3d=merge_dir(edema_slices)

        ref = sort_glob(f'{refdir}/*{sub}_gd.nii.gz')
        assert len(ref)==1
        parameter = sitk.ReadImage(ref[0])

        print(f"sub:{sub}-{ref[0]}")

        array=np.zeros_like(sitk.GetArrayFromImage(parameter))
        #scar
        array[np.where(1==edema_3d)]=1220
        #edema
        array[np.where(1==scar_3d)]=2221

        sitk_write_multi_lab(array, parameter, outdir, os.path.basename(ref[0]).replace("gd",'pred'))

def merge_slice_to_ref_MSCMR(slice_dir,refdir,outdir):

    mkdir_if_not_exist(outdir)
    for sub in range(26, 46):
        scar_slices=sort_glob(f"{slice_dir}/scar/*{sub}*")
        edema_slices=sort_glob(f"{slice_dir}/edemascar/*{sub}*")


        assert  len(scar_slices)==len(edema_slices)
        scar_3d=merge_dir(scar_slices)
        edema_3d=merge_dir(edema_slices)

        ref = sort_glob(f'{refdir}/*{sub}_ana_patho_de*.nii.gz')
        assert len(ref)==1
        parameter = sitk.ReadImage(ref[0])

        print(f"sub:{sub}-{ref[0]}")

        array=np.zeros_like(sitk.GetArrayFromImage(parameter))
        #scar
        array[np.where(1==edema_3d)]=1220
        #edema
        array[np.where(1==scar_3d)]=2221

        sitk_write_multi_lab(array, parameter, outdir, os.path.basename(ref[0]).replace("gd",'pred'))

from tools.sitkPasteTool import resample_segmentations
def paste_to_ref_MSCMR(merge_dir, ref_dir, out_dir):
    mkdir_if_not_exist(out_dir)

    subject_crop=sort_glob(f"{merge_dir}/**.nii.gz")
    tmp=sort_glob(f"{ref_dir}/*_img_de.nii.gz")
    subject_ori=[]
    for item in tmp:
        for i in range(26,46):
            if item.find(str(i))>0:
                subject_ori.append(item)
                break



    for s_crop,s_ori in zip(subject_crop,subject_ori):
        print(f"f{s_crop}--{s_ori}")
        crop=sitk.ReadImage(s_crop)
        ori=sitk.ReadImage(s_ori)
        ori=sitk.Cast(ori,crop.GetPixelID())
        crop=resample_segmentations(ori,crop)
        pasted_img=paste_roi_image(ori,crop)
        sitk_write_multi_lab(pasted_img,dir=out_dir,name=os.path.basename(s_crop))

def paste_to_ref(crop_dir,ref_dir,out_dir):
    mkdir_if_not_exist(out_dir)

    subject_crop=sort_glob(f"{crop_dir}/**.nii.gz")
    subject_ori=sort_glob(f"{ref_dir}/*_gd.nii.gz")
    for s_crop,s_ori in zip(subject_crop,subject_ori):
        crop=sitk.ReadImage(s_crop)
        ori=sitk.ReadImage(s_ori)
        ori=sitk.Cast(ori,crop.GetPixelID())
        pasted_img=paste_roi_image(ori,crop)
        sitk_write_multi_lab(pasted_img,dir=out_dir,name=os.path.basename(s_crop))

from tools.np_sitk_tools import reindex_label_array_by_dict
from medpy.metric import dc
#


from tools.dir import mk_or_cleardir
def msmcr_main(task_id):
    #手工或者利用labelcrop 出来的数据
    crop_ref= "../data/gen_unaligned/croped"
    #网络输出的数据
    input=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_mscmr_MS/labelsTs_pre"
    # 合成的3D-crop数据
    merge_out= f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_mscmr_MS/labelsTs_3D"

    mk_or_cleardir(merge_out)
    merge_slice_to_ref_MSCMR(input, crop_ref, merge_out)

    #原始数据
    paste_ref="../data/unalignedmsmcr_rerank/"
    #最终的输出结果
    paste_out=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_mscmr_MS/labelsTs_PSN"
    mk_or_cleardir(paste_out)
    paste_to_ref_MSCMR(merge_out, paste_ref, paste_out)

    cal_dice(paste_ref,paste_out)



def cal_dice(paste_ref,paste_out):
    ds=[]
    for sub in range(26,46):
        gt=f"{paste_ref}/subject_{sub}_ana_patho_de_scar.nii.gz"
        pred=f"{paste_out}/subject_{sub}_ana_patho_de_scar.nii.gz"
        gt_arry=sitk.GetArrayFromImage(sitk.ReadImage(gt))
        pred_array=sitk.GetArrayFromImage(sitk.ReadImage(pred))
        gt_arry=reindex_label_array_by_dict(gt_arry,{1:[2221]})
        pred_array=reindex_label_array_by_dict(pred_array,{1:[2221]})
        ds.append(dc(gt_arry,pred_array))
    print_mean_and_std(ds)




def myops_main(task_id):
    #手工或者利用labelcrop 出来的数据
    crop_ref= "../data/gen_myops20/test20_croped"
    #网络输出的数据
    input=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_myops2020_MS/labelsTs"
    # 合成的3D-crop数据
    merge_out= f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_myops2020_MS/labelsTs_3D"

    merge_slice_to_ref(input, crop_ref, merge_out)

    #原始数据
    paste_ref="../data/myops/test20"
    #最终的输出结果
    paste_out=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_id}_myops2020_MS/labelsTs_PSN"
    paste_to_ref(merge_out, paste_ref, paste_out)

import os
from tools.metric import print_mean_and_std

if __name__=='__main__':
    msmcr_main()