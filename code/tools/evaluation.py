from tools.np_sitk_tools import extract_label_bitwise, merge_dir
from medpy.metric import dc, asd,assd,specificity as spec, sensitivity as sens, precision as prec,ravd
from medpy.metric import hd95,hd
from tools.dir import sort_glob
import numpy as np
from tools.excel import write_array
import SimpleITK as sitk
from tools.dir import mkcleardir,mkdir_if_not_exist
from visulize.pngutils import SaveNumpy2Png
import os

def cal_biasis(arrayA,arrayB,voxelspacing=None):
    if np.sum(arrayA)>np.sum(arrayB):
        return ravd(arrayA,arrayB)
    else:
        return -ravd(arrayA,arrayB)


def extract_scar(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [4]})

def extract_edema(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [2]})

def extract_scar_edema(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [4]})+extract_label_bitwise(gt_3d, {1: [2]})-extract_label_bitwise(gt_3d, {1: [6]})


from tools.np_sitk_tools import reindex_label_by_dict,reindex_label_array_by_dict
'''
For PSF only
'''
def evaluation_by_dir(task_id,pre_dir,gt_dir,ref_3D,indicator=None):

    res = {}
    for it in ['scar','edema','de','t2']:
        res[it] = {'dice':[], 'hd':[],'hd95':[], 'asd':[], 'sens':[], 'prec':[],'biasis':[]}

    for subj in range(1026, 1051):
        for type in ['de','t2']:
            gts = sort_glob(f"{gt_dir}/*{subj}*")
            # gts = sort_glob(f"{gt_dir}/*{subj}*/*{type}*assn_gt_lab*")
            if type=='de':
                preds = sort_glob(f"{pre_dir}/scar/*{subj}*")
            elif type=='t2':
                preds = sort_glob(f"{pre_dir}/edema/*{subj}*")
            else:
                exit(-999)
            if len(preds)==0:
                continue
            assert len(preds) == len(gts)

            spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
            assert len(spacing_para_3D) == 1
            spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

            gt_3d = merge_dir(gts)
            pred_3d = merge_dir(preds)
            if type=='de':
                gt_3d = extract_scar(gt_3d)  # scar
                # gt_3d = reindex_label_array_by_dict(gt_3d,{1:[2221]})  # scar

            elif type=='t2':
                gt_3d = extract_edema(gt_3d)  # scar
                # gt_3d = reindex_label_array_by_dict(gt_3d,{1:[1220]})  # scar
            else:
                exit(-999)

            res[type]['dice'].append(dc(pred_3d, gt_3d))
            res[type]['sens'].append(sens(pred_3d, gt_3d))
            res[type]['prec'].append(prec(pred_3d, gt_3d))
            res[type]['hd'].append(hd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['hd95'].append(hd95(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['asd'].append(asd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['biasis'].append(cal_biasis(pred_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_id}.xls', f"{t}-{k}", res[t][k])

# def evaluation_by_dir_V2(task_id,pre_dir,gt_dir,ref_3D,indicator=None):
#
#     res = {}
#     for it in ['scar','edema','de','t2']:
#         res[it] = {'dice':[], 'hd':[],'hd95':[], 'asd':[], 'sens':[], 'prec':[]}
#
#     for subj in range(1026, 1051):
#         for type in ['de','t2']:
#             gts = sort_glob(f"{gt_dir}/*{subj}*")
#             preds = sort_glob(f"{pre_dir}/*{subj}*")
#             assert len(preds) == len(gts)
#
#             spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
#             assert len(spacing_para_3D)==1
#             spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()
#
#             gt_3d = merge_dir(gts)
#             scar_3d = merge_dir(preds)
#             if type=='de':
#                 gt_3d_scar = extract_scar(gt_3d)  # scar
#             elif type=='t2':
#                 gt_3d_scar = extract_scar(gt_3d)  # scar
#
#             res[type]['dice'].append(dc(scar_3d, gt_3d_scar))
#             res[type]['sens'].append(sens(scar_3d, gt_3d_scar))
#             res[type]['prec'].append(prec(scar_3d, gt_3d_scar))
#             res[type]['hd'].append(hd(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#             res[type]['hd95'].append(hd95(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#             res[type]['asd'].append(asd(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#
#     for t in res.keys():
#         print(f"===={t}=======")
#         for k in res[t].keys():
#             # print(res[t][k])
#             if len(res[t][k]) <= 0:
#                 continue
#             print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
#             write_array(f'../outputs/result/{task_id}.xls', f"{t}-{k}", res[t][k])


'''
For nnunet only
'''
def evaluation_by_dir_V2(task_name,input_dir,gt_dir, ref_3D,type):
    # 网络输出的数据


    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[]}

    for subj in range(1026, 1051):
        # if subj in [26,28,38]:
        #     continue
        preds = sort_glob(f"{input_dir}/subject*{subj}*")
        gts = sort_glob(f"{gt_dir}/subject*{subj}*")
        assert len(preds) == len(gts)
        gt_3d = merge_dir(gts)
        preds_3d = merge_dir(preds)
        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
        spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])

from visulize.color import colorGD,colorC0,colorT2,colorDe
def save_png_dir(pre_dir,img_dir,gt_dir,out_dir):
    #网络输出的数据


    op=SaveNumpy2Png()
    mkcleardir(out_dir)
    for subj in range(1026,1051):
        # if subj in [26,28,38]:
        #     continue
        for type in ['de', 't2']:

            gts = sort_glob(f"{gt_dir}/*{subj}*")
            if type=='de':
                preds = sort_glob(f"{pre_dir}/scar/*{subj}*")
                imgs = sort_glob(f"{img_dir}/subject_{subj}*0002.nii.gz")
            elif type=='t2':
                preds = sort_glob(f"{pre_dir}/edema/*{subj}*")
                imgs = sort_glob(f"{img_dir}/subject_{subj}*0001.nii.gz")
            else:
                exit(-999)
            if len(preds)==0:
                continue
            assert len(preds)==len(gts)
            assert len(imgs)==len(gts)
            for img,gt,pred in zip(imgs,gts,preds):
                name=f"{os.path.basename(img).split('.')[0]}.png"
                img_array=sitk.GetArrayFromImage(sitk.ReadImage(img))
                gt_array=sitk.GetArrayFromImage(sitk.ReadImage(gt))
                sub_dir=f"{out_dir}/{subj}"

                mkdir_if_not_exist(sub_dir)
                pred_array=sitk.GetArrayFromImage(sitk.ReadImage(pred))
                if type=='de':
                    # gt_array=np.bitwise_and(gt_array,4)
                    # gt_array=np.where(gt_array==4,1,0)
                    gt_array=extract_scar(gt_array)
                    op.save_img(img_array,sub_dir,f"img_scar_{name}")
                    op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_scar_{name}",colors=colorDe,mash_alpha=1)
                    op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_scar_{name}", colors=colorDe,mash_alpha=1)
                else:
                    # gt_array=np.bitwise_and(gt_array,2)
                    # gt_array=np.where(gt_array==2,1,0)
                    gt_array = extract_edema(gt_array)
                    op.save_img(img_array,sub_dir,f"img_edema_{name}")
                    op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_edema_{name}",colors=colorT2,mash_alpha=1)
                    op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_edema_{name}", colors=colorT2,mash_alpha=1)


def save_png_dir_v2(task_name, pred_dir,img_dir,gt_dir,out_dir,type):
    #网络输出的数据
    # pred_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre"
    # gt_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs"
    # img_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/imagesTs"
    # out_dir= f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre_png"

    op=SaveNumpy2Png()
    mkcleardir(out_dir)
    for subj in range(1026,1051):
        # if subj in [26,28,38]:
        #     continue
        gts=sort_glob(f"{gt_dir}/subject*{subj}*")
        preds = sort_glob(f"{pred_dir}/subject*{subj}*")
        imgs=sort_glob(f"{img_dir}/subject*{subj}*0000.nii.gz")

        assert len(preds)==len(gts)
        assert len(imgs)==len(gts)
        for img,gt,pred in zip(imgs,gts,preds):
            name=f"{os.path.basename(img).split('.')[0]}.png"
            img_array=sitk.GetArrayFromImage(sitk.ReadImage(img))
            gt_array=sitk.GetArrayFromImage(sitk.ReadImage(gt))
            sub_dir=f"{out_dir}/{subj}"

            mkdir_if_not_exist(sub_dir)
            pred_array=sitk.GetArrayFromImage(sitk.ReadImage(pred))
            if type == "de" or type=="DE":
                op.save_img(img_array,sub_dir,f"img_scar_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_scar_{name}",colors=colorDe,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_scar_{name}", colors=colorDe,mash_alpha=1)
            elif type=="t2" or type=="T2":
                op.save_img(img_array,sub_dir,f"img_edema_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_edema_{name}",colors=colorT2,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_edema_{name}", colors=colorT2,mash_alpha=1)

def save_png_dir_v3(task_name, pred_dir,img_dir,gt_dir,out_dir,type):
    #网络输出的数据
    # pred_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre"
    # gt_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs"
    # img_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/imagesTs"
    # out_dir= f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre_png"

    op=SaveNumpy2Png()
    mkcleardir(out_dir)
    for subj in range(1026,1051):
        # if subj in [26,28,38]:
        #     continue
        gts=sort_glob(f"{gt_dir}/subject*{subj}*")
        preds = sort_glob(f"{pred_dir}/subject*{subj}*")
        if type == 'de' or type=='DE':
            imgs=sort_glob(f"{img_dir}/subject*{subj}*0000.nii.gz")
        elif type == 't2' or type=='T2':
            imgs=sort_glob(f"{img_dir}/subject*{subj}*0001.nii.gz")
        else:
            exit(-999)


        assert len(preds)==len(gts)
        assert len(imgs)==len(gts)
        for img,gt,pred in zip(imgs,gts,preds):
            name=f"{os.path.basename(img).split('.')[0]}.png"
            img_array=sitk.GetArrayFromImage(sitk.ReadImage(img))
            gt_array=sitk.GetArrayFromImage(sitk.ReadImage(gt))
            sub_dir=f"{out_dir}/{subj}"

            mkdir_if_not_exist(sub_dir)
            pred_array=sitk.GetArrayFromImage(sitk.ReadImage(pred))
            if type == "de" or type=="DE":
                op.save_img(img_array,sub_dir,f"img_scar_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_scar_{name}",colors=colorDe,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_scar_{name}", colors=colorDe,mash_alpha=1)
            elif type=="t2" or type=="T2":
                op.save_img(img_array,sub_dir,f"img_edema_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_edema_{name}",colors=colorT2,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_edema_{name}", colors=colorT2,mash_alpha=1)

def save_png_dir_v4(task_name, pred_dir,img_dir,gt_dir,out_dir,type):
    #网络输出的数据
    # pred_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre"
    # gt_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs"
    # img_dir=f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/imagesTs"
    # out_dir= f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre_png"

    op=SaveNumpy2Png()
    mkcleardir(out_dir)
    for subj in range(1026,1051):
        # if subj in [26,28,38]:
        #     continue
        gts=sort_glob(f"{gt_dir}/subject*{subj}*")
        preds = sort_glob(f"{pred_dir}/subject*{subj}*")
        imgs=sort_glob(f"{img_dir}/subject*{subj}*0001.nii.gz")


        assert len(preds)==len(gts)
        assert len(imgs)==len(gts)
        for img,gt,pred in zip(imgs,gts,preds):
            name=f"{os.path.basename(img).split('.')[0]}.png"
            img_array=sitk.GetArrayFromImage(sitk.ReadImage(img))
            gt_array=sitk.GetArrayFromImage(sitk.ReadImage(gt))
            sub_dir=f"{out_dir}/{subj}"

            mkdir_if_not_exist(sub_dir)
            pred_array=sitk.GetArrayFromImage(sitk.ReadImage(pred))
            if type == "de" or type=="DE":
                op.save_img(img_array,sub_dir,f"img_scar_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_scar_{name}",colors=colorDe,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_scar_{name}", colors=colorDe,mash_alpha=1)
            elif type=="t2" or type=="T2":
                op.save_img(img_array,sub_dir,f"img_edema_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_edema_{name}",colors=colorT2,mash_alpha=1)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_edema_{name}", colors=colorT2,mash_alpha=1)
