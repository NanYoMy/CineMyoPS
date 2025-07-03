# from nnunet.inference.predict_simple import main_pathology as predict_pathology
from nnunet.inference.predict_simple import main as predict_pathology
from nnunet import gl
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import argparse

# nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
# nnUNet_train 2d nnUNetTrainer3MV2 Task701_myops2020_MS 0
# ~/anaconda3/envs/dwb_pytorch/bin/python
# -i $nnUNet_raw_data_base/nnUNet_raw_data/Task805_mmm_SA/imagesTs -o $nnUNet_raw_data_base/nnUNet_raw_data/Task805_mmm_SA/labelsTs -t 805 -m 2d --chk model_best --overwrite_existing
from tools.dir import sort_glob
import numpy as np
# import SimpleITK as sitk
from visulize.pngutils import SaveNumpy2Png
from tools.excel import write_array
# from tools.np_sitk_tools import reindex_label_array_by_dict, merge_dir
from medpy.metric import dc, asd, specificity as spec, sensitivity as sens, precision as prec
from medpy.metric import hd95 as hd
import os
# from pre_post_processing.psnpostprocess import myops_main
# from pre_post_processing.psnpostprocess import msmcr_main as psnpost
# from RJ_psn_4_test import extract_scar,extract_edema

# def evaluate_validation(output_dir, gd_dir):
#     ds = {"scar": [], "edemascar": [], "myo": []}
#     for type in ['scar', 'edemascar', 'myo']:
#
#         for subject in range(121,126):
#             preds = sort_glob(f"{os.path.dirname(output_dir)}/{type}/*{subject}*nii.gz")
#             gds = sort_glob(f"{gd_dir}/*{subject}*/*C0_gd*nii.gz")
#             gds_list = []
#             preds_list = []
#             assert len(gds) == len(preds)
#
#             for gd, pred in zip(gds, preds):
#                 # print(f"{gd}-{pred}")
#                 gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)), axis=0))
#                 preds_list.append(sitk.GetArrayFromImage(sitk.ReadImage(pred)))
#             gds_arr = np.concatenate(gds_list, axis=0)
#             preds_arr = np.concatenate(preds_list, axis=0)
#
#             if type=="scar":
#                 ids={1:[3]}
#             elif type=="edemascar":
#                 ids = {1: [2,3]}
#             elif type=="myo":
#                 ids = {1: [1,2,3]}
#
#             ds[type].append(dc(preds_arr,reindex_label_array_by_dict(gds_arr,ids)))
#
#     for k in ds.keys():
#         if (len(ds[k])) > 0:
#             print(f"{k}:{np.mean(ds[k])}")

# import  platform
# from myops_psn_5_evaluation import evaluation_myops
from tools.np_sitk_tools import extract_label_bitwise, merge_dir

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def extract_scar(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [4]})


def extract_edema(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [2]})


def msmcr_eval_ps(task_name, type):
    # 网络输出的数据
    input_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre"
    gt_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs"
    # scar edemascar myo
    # type = "scar"
    if type == "de":
        type = "scar"
    elif type == 't2':
        type = "edema"

    res = {}
    for it in ['dice', 'hd', 'asd', 'sens', 'prec']:
        res[it] = {"scar": [], 'edemascar': [], 'edema': []}

    for subj in range(1026, 1051):
        # if subj in [26,28,38]:
        #     continue
        preds = sort_glob(f"{input_dir}/subject_{subj}*")
        gts = sort_glob(f"{gt_dir}/subject_{subj}*")
        assert len(preds) == len(gts)
        gt_3d = merge_dir(gts)
        preds_3d = merge_dir(preds)
        # gt_3d_scar = extract_scar(gt_3d)  # scar
        # https://blog.csdn.net/tangxianyu/article/details/102454611

        if task_name == '763':
            spacing_para_3D = sort_glob(f"../data/gen_RJ_unaligned/croped/*{subj}*de.nii.gz")
            spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        if task_name == '764':
            spacing_para_3D = sort_glob(f"../data/gen_RJ_unaligned/croped/*{subj}*t2.nii.gz")
            spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        res['dice'][type].append(dc(preds_3d, gt_3d))
        res['asd'][type].append(asd(preds_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
        res['hd'][type].append(hd(preds_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
        res['sens'][type].append(sens(preds_3d, gt_3d))
        res['prec'][type].append(prec(preds_3d, gt_3d))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])


import SimpleITK as sitk
from tools.dir import mkcleardir, mkdir_if_not_exist
from visulize.pngutils import SaveNumpy2Png


def save_png_res(task_name, type):
    # 网络输出的数据
    pred_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre"
    gt_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs"
    img_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/imagesTs"

    out_mask_in_img_dir = f"../outputs/nnunet/raw/nnUNet_raw_data/Task{task_name}_RJ_ms_psnnunet_{type}/labelsTs_pre_png"
    op = SaveNumpy2Png()
    mkcleardir(out_mask_in_img_dir)
    for subj in range(1026, 1051):
        # if subj in [26,28,38]:
        #     continue
        gts = sort_glob(f"{gt_dir}/subject_{subj}*")

        if task_name == "763":
            preds = sort_glob(f"{pred_dir}/subject_{subj}*")
            imgs = sort_glob(f"{img_dir}/subject_{subj}*0000.nii.gz")

        else:
            preds = sort_glob(f"{pred_dir}/subject_{subj}*")
            imgs = sort_glob(f"{img_dir}/subject_{subj}*0000.nii.gz")

        assert len(preds) == len(gts)
        assert len(imgs) == len(gts)
        for img, gt, pred in zip(imgs, gts, preds):
            name = f"{os.path.basename(img).split('.')[0]}.png"
            img_array = sitk.GetArrayFromImage(sitk.ReadImage(img))
            gt_array = sitk.GetArrayFromImage(sitk.ReadImage(gt))
            sub_dir = f"{out_mask_in_img_dir}/{subj}"

            mkdir_if_not_exist(sub_dir)
            pred_array = sitk.GetArrayFromImage(sitk.ReadImage(pred))
            if task_name == "763":
                op.save_img(img_array, sub_dir, f"img_scar_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_scar_{name}", colors=[(0, 255, 255)],
                                      mash_alpha=0.2)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_scar_{name}", colors=[(255, 255, 0)],
                                      mash_alpha=0.2)
            else:
                op.save_img(img_array, sub_dir, f"img_edema_{name}")
                op.save_img_with_mask(img_array, gt_array, sub_dir, f"gt_edema_{name}", colors=[(0, 255, 255)],
                                      mash_alpha=0.2)
                op.save_img_with_mask(img_array, pred_array, sub_dir, f"pred_edema_{name}", colors=[(0, 255, 0)],
                                      mash_alpha=0.2)


from tools.evaluation import evaluation_by_dir_V2, save_png_dir_v2

if __name__ == "__main__":

    # evaluation for mscmr
    # psnpost()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-tr', '--trainer_class_name',
                        help='Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution '
                             'U-Net. The default is %s. If you are running inference with the cascade and the folder '
                             'pointed to by --lowres_segmentations does not contain the segmentation maps generated by '
                             'the low resolution U-Net then the low resolution segmentation maps will be automatically '
                             'generated. For this case, make sure to set the trainer class here that matches your '
                             '--cascade_trainer_class_name (this part can be ignored if defaults are used).'
                             % default_trainer,
                        required=False,
                        default=default_trainer)

    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % default_cascade_trainer, required=False,
                        default=default_cascade_trainer)

    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)

    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)

    parser.add_argument('-f', '--folds', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None',
                        help="if model is the highres stage of the cascade then you can use this folder to provide "
                             "predictions from the low resolution 3D U-Net. If this is left at default, the "
                             "predictions will be generated automatically (provided that the 3D low resolution U-Net "
                             "network weights are present")

    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument('--whichsubnet', "--whichsubnet", required=False, default='scar',
                        help="predicted via which subnet")
    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you "
                             "want to use n GPUs to predict this "
                             "folder you need to run this command "
                             "n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different "
                             "GPU (via "
                             "CUDA_VISIBLE_DEVICES=X)")

    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")

    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")

    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations, has no effect if mode=fastest. Do not touch this.")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z is z is done differently. Do not touch this.")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest. "
    #                          "Do not touch this.")
    parser.add_argument('--chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that yhis is not recommended (mixed precision is ~2x faster!)')

    args = parser.parse_args()
    gl._init()

    # if args.whichsubnet=="all":
    #     subnets= ['scar','edema']
    # else:
    #     subnets=[args.whichsubnet]
    # base_dir=args.output_folder

    # for subnet in subnets:
    #     args.output_folder=f"{base_dir}/{subnet}"
    #     gl.set_value("subnet",subnet)
    predict_pathology()

    # evaluation for mscmr
    if args.task_name == '763' or args.task_name == '766':
        tmp = 'de'
        evaluation_by_dir_V2(args.task_name,
                             f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre",
                             f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs",
                             "../data/gen_RJ_unaligned/croped",
                             tmp)
        save_png_dir_v2(args.task_name,
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/imagesTs",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre_png",
                        tmp)

    elif args.task_name == '764' or args.task_name == '765':
        tmp = 't2'
        evaluation_by_dir_V2(args.task_name,
                             f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre",
                             f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs",
                             "../data/gen_RJ_unaligned/croped",
                             't2')

        save_png_dir_v2(args.task_name,
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/imagesTs",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs",
                        f"../outputs/nnunet/raw/nnUNet_raw_data/Task{args.task_name}_RJ_ms_psnnunet_{tmp}/labelsTs_pre_png",
                        tmp)





    else:
        print("error")

    # evaluation for mscmr
    # psnpost(args.task_name)

    # evalutaion for mscmr
