import matplotlib.pyplot
from torch.utils.tensorboard  import SummaryWriter
import torch
from visulize.helper import extract_semantic_contour,draw_coutours,draw_mask
from tools.dir import sort_glob
from medpy.metric import dc,assd,asd
# from medpy.metric import hd95 as hd
from medpy.metric import hd

import logging
import cv2
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.tps_painter import save_image_with_tps_points
from tools.nii_lab_to_png import save_img
from tools.np_sitk_tools import clipseScaleSitkImage,clipseScaleSArray
from skimage.util.compare import compare_images
from skimage.exposure import rescale_intensity
from tools.excel import write_array
from visulize.color import my_color
from numpy import ndarray
import cv2
from tools.dir import mkdir_if_not_exist
import os
import numpy as np
from tools.set_random_seed import setup_seed
from tools.itkdatawriter import sitk_write_image,sitk_write_images,sitk_write_labs,sitk_write_lab
from medpy.metric import dc,asd,assd
from tools.np_sitk_tools import sitkResize3D
from baseclass.medicalimage import Modality,MyoPSLabelIndex
import SimpleITK as sitk
from skimage import  segmentation,color
from matplotlib.pyplot import plot
import seaborn
class BaseExperiment():
    def write_dict_to_tb(self,dict,step):
        for k in dict.keys():
            self.eval_writer.add_scalar(f"{k}",dict[k],step)

    def __init__(self,args):
        self.args=args
        self.eval_writer = SummaryWriter(log_dir=f"{args.log_dir}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from matplotlib import pyplot as plt
class BaseMyoPSExperiment(BaseExperiment):

    def plotbox(self,offset_list,output,filename):
        # seaborn.set(style='whitegrid')
        # tip = seaborn.load_dataset('tips')
        # seaborn.boxplot(x='day', y='tip', data=tip)
        res=[]  #72 16 2
        # offset_list=offset_list)
        data=np.transpose(offset_list, [1, 0]).tolist()
        data=[np.array(d) for d in data]

        plot=seaborn.boxplot( data=data)
        # plot=seaborn.violinplot( data=data)
        plot.set(ylim=(0, 0.25))
        plot.yaxis.grid(True)
        plot.xaxis.grid(False)
        fig=plot.get_figure()
        fig.savefig(f"{output}/{filename}",dpi=400)
        fig.clf()

    def save_img_with_tps(self,img,control_points,dir,name):
        mkdir_if_not_exist(dir)
        source_array = np.squeeze(img)
        control_points=(control_points.data[0])
        tmp=sitk.GetImageFromArray(source_array)
        tmp=clipseScaleSitkImage(tmp,0,100)
        source_array=sitk.GetArrayFromImage(tmp).astype('uint8')

        save_image_with_tps_points(control_points, source_array, dir, name, self.args.grid_size, 500, 0)

    def save_img(self,img,dir,name):
        mkdir_if_not_exist(dir)
        source_array = np.squeeze(img)
        tmp = sitk.GetImageFromArray(source_array)
        tmp = clipseScaleSitkImage(tmp, 0, 100)
        source_array = sitk.GetArrayFromImage(tmp).astype('uint8')
        save_img(source_array,dir,name,img_size=500,border=0)


    def save_diff_img(self,array1,array2,dir,name):
        mkdir_if_not_exist(dir)
        array1=np.squeeze(array1).astype(np.float32)
        array2=np.squeeze(array2).astype(np.float32)
        array1=cv2.resize(array1,(500,500))
        array2=cv2.resize(array2,(500,500))
        diff = compare_images(rescale_intensity(array1),
                              rescale_intensity(array2),
                              method='checkerboard',n_tiles=(4,4))
        diff=(diff * 255).astype(np.uint8)
        cv2.imwrite(f"{dir}/{name}",diff )

    def save_tensor_with_parameter(self, tensor, parameter, outputdir, name, is_label=False):
        array=tensor.cpu().numpy()
        array=np.squeeze(array)
        target_size=parameter.GetSize()
        if is_label==True:
            array=self.op.resize(array,(target_size[1],target_size[0]),0)
        else:
            array=self.op.resize(array,(target_size[1],target_size[0]),1)

        array=np.expand_dims(array,axis=0)
        if is_label==True:
            array=np.round(array).astype(np.int16)

        img = sitk.GetImageFromArray(array)
        img.CopyInformation(parameter)
        sitk.WriteImage(img, os.path.join(outputdir, name+'.nii.gz'))



    def save_torch_img_lab(self, output_dir, img, lab, name, info_img,info_lab):
        self.save_torch_to_nii(output_dir,img,name,info_img,is_lab=False)
        self.save_torch_to_nii(output_dir,lab,name,info_lab)

    def save_torch_to_nii(self, output_dir, array, name, modality, is_lab=True):
        array = np.squeeze(array[0].detach().cpu().numpy())
        output_dir=f"{output_dir}/{os.path.basename(os.path.dirname(name[0]))}"
        mkdir_if_not_exist(output_dir)
        term=os.path.basename((name[0])).split("_")

        if is_lab==True:
            lab_name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{modality}_{term[4]}'
            sitk_write_image(np.round(array).astype(np.int16), None, output_dir, lab_name)
        else:

            img_name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{modality}_{term[4]}'
            sitk_write_image(array, None, output_dir, img_name)


    def save_diff_lab(self, output_dir, source, target, name, modality):
        source = np.squeeze(source[0].detach().cpu().numpy())
        target = np.squeeze(target[0].detach().cpu().numpy())
        diff=source-target
        print(f"{name}:{modality}: {dc(source,target)}")
        output_dir=f"{output_dir}/{os.path.basename(os.path.dirname(name[0]))}"
        mkdir_if_not_exist(output_dir)
        term=os.path.basename((name[0])).split("_")
        # img_name=f'{term[0]}_{term[1]}_img_{modality}_{term[-1]}'
        lab_name=f'{term[0]}_{term[1]}_lab_{modality}_{term[-1]}'
        # sitk_write_image(source, None, output_dir, img_name)
        sitk_write_image(np.round(diff), None, output_dir, lab_name)

    def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_scar_mask, scar_mask, lv_mask, rv_mask,ori_label
        c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
        c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=4, length=1)
        t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=4, length=1)
        de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=4, length=1)
        c0_roi_reg_mask_3 = lab_c0.narrow(dim=1, start=5, length=1)#rv
        t2_roi_reg_mask_3 = lab_t2.narrow(dim=1, start=5, length=1)#rv
        de_roi_reg_mask_3 = lab_de.narrow(dim=1, start=5, length=1)#rv
        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
        roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
        roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab1,roi_lab2
    def save_img_with_contorusV2(self, img, lab, dir, name,color=[(1, 0, 0)]):
        mkdir_if_not_exist(dir)


        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)
        img=cv2.resize(img, (500, 500))
        lab=cv2.resize(lab, (500, 500),interpolation=cv2.INTER_NEAREST)
        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        img = segmentation.mark_boundaries(img, lab.astype(np.uint8),color=color,outline_color=color, mode='thick')


        # img=sitk.GetImageFromArray(img)
        # img=clipseScaleSitkImage(img,0,100)
        # img=sitk.GetArrayFromImage(img).astype('uint8')
        # contours = extract_semantic_contour(lab)
        # img = draw_coutours(np.tile(np.expand_dims(img, -1), (1, 1, 3)), contours,
        #                     my_color)
        img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}",img*255 )

    def save_image_with_pred_gt_contousV2(self, img, pred_lab, gt_lab, dir, name,colora=[(0, 0, 1)],colorb=[(1, 0, 0)]):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        pred_lab = np.squeeze(pred_lab).astype(np.float32)
        gt_lab = np.squeeze(gt_lab).astype(np.float32)
        pred_lab=cv2.resize(pred_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
        gt_lab=cv2.resize(gt_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
        img=cv2.resize(img, (500, 500))
        img=clipseScaleSArray(img,0,100).astype(np.uint8)
        img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=colora,outline_color=colora, mode='thick')
        img = segmentation.mark_boundaries(img, gt_lab.astype(np.uint8), color=colorb,outline_color=colorb, mode='thick')
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img*255)

    def save_img_with_mv_fix_contorusV2(self, img, mv_lab, fix_lab, dir, name,colora=[(0, 0, 1)],colorb=[(1, 0, 0)]):
        self.save_image_with_pred_gt_contousV2(img, mv_lab, fix_lab, dir, name,colora,colorb)


    def __init__(self,args):
        super().__init__(args)



class BaseMSCMRExperiment(BaseMyoPSExperiment):
    def create_torch_tensor(self, img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask,myo_scar_ede_mask,ori_lab
        c0_roi_reg_mask_myo = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_myo = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_myo = lab_de.narrow(dim=1, start=1, length=1)

        c0_roi_reg_mask_lv = lab_c0.narrow(dim=1, start=4, length=1)
        t2_roi_reg_mask_lv = lab_t2.narrow(dim=1, start=4, length=1)
        de_roi_reg_mask_lv = lab_de.narrow(dim=1, start=4, length=1)

        c0_roi_reg_mask_rv = lab_c0.narrow(dim=1, start=5, length=1)
        t2_roi_reg_mask_rv = lab_t2.narrow(dim=1, start=5, length=1)
        de_roi_reg_mask_rv = lab_de.narrow(dim=1, start=5, length=1)


        lab_c0 = lab_c0.narrow(dim=1, start=-1, length=1)
        lab_t2 = lab_t2.narrow(dim=1, start=-1, length=1)
        lab_de = lab_de.narrow(dim=1, start=-1, length=1)

        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab_myo={Modality.c0:c0_roi_reg_mask_myo, Modality.t2:t2_roi_reg_mask_myo, Modality.de:de_roi_reg_mask_myo}
        roi_lab_lv={Modality.c0:c0_roi_reg_mask_lv, Modality.t2:t2_roi_reg_mask_lv, Modality.de:de_roi_reg_mask_lv}
        roi_lab_rv={Modality.c0:c0_roi_reg_mask_rv, Modality.t2:t2_roi_reg_mask_rv, Modality.de:de_roi_reg_mask_rv}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab_myo,roi_lab_lv,roi_lab_rv

    def create_test_torch_tensor(self, img_c0, img_t2, img_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        img = {Modality.c0: img_c0, Modality.t2: img_t2, Modality.de: img_de}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return img

    def save_img_with_tps(self,img,control_points,dir,name):
        mkdir_if_not_exist(dir)
        source_array = np.squeeze(img)
        control_points=(control_points.data[0])
        tmp=sitk.GetImageFromArray(source_array)
        tmp=clipseScaleSitkImage(tmp,0,100)
        source_array=sitk.GetArrayFromImage(tmp).astype('uint8')

        save_image_with_tps_points(control_points, source_array, dir, name, self.args.grid_size, 500, 0)

    def save_img(self,img,dir,name):
        mkdir_if_not_exist(dir)
        source_array = np.squeeze(img)
        tmp = sitk.GetImageFromArray(source_array)
        tmp = clipseScaleSitkImage(tmp, 0, 100)
        source_array = sitk.GetArrayFromImage(tmp).astype('uint8')
        save_img(source_array,dir,name,img_size=500,border=0)


    def renamepath(self, name, tag):
        term = os.path.basename((name[0])).split("_")
        name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{tag}_{term[4]}'
        return name

    def save_diff_img(self,array1,array2,dir,name):
        mkdir_if_not_exist(dir)
        array1=np.squeeze(array1).astype(np.float32)
        array2=np.squeeze(array2).astype(np.float32)
        array1=cv2.resize(array1,(500,500))
        array2=cv2.resize(array2,(500,500))
        diff = compare_images(rescale_intensity(array1,out_range=(0,1)),
                              rescale_intensity(array2,out_range=(0,1)),
                              method='checkerboard',n_tiles=(4,4))
        diff=(diff * 255).astype(np.uint8)
        # diff=cv2.flip(diff,0)#调整视角
        cv2.imwrite(f"{dir}/{name}",diff )

    def save_tensor_with_parameter(self, tensor, parameter, outputdir, name, is_label=False):

        if not isinstance(tensor,ndarray):
            array=tensor.cpu().numpy()
        else:
            array=tensor
        array=np.squeeze(array)
        target_size=parameter.GetSize()
        if is_label==True:
            array=self.op.resize(array,(target_size[1],target_size[0]),0)
        else:
            array=self.op.resize(array,(target_size[1],target_size[0]))

        array=np.expand_dims(array,axis=0)
        if is_label==True:
            array=np.round(array).astype(np.int16)

        img = sitk.GetImageFromArray(array)
        img.CopyInformation(parameter)
        sitk.WriteImage(img, os.path.join(outputdir, name+'.nii.gz'))

    def save_img_with_mask(self,img,lab,dir,name):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)
        img = cv2.resize(img, (500, 500))
        lab = cv2.resize(lab, (500, 500))
        img=clipseScaleSArray(img,0,100).astype('uint8')
        img = color.label2rgb(lab.astype(np.uint8), img, colors=[(255, 255, 0), (0, 255, 255)], alpha=0.001,
                                       bg_label=0, bg_color=None)
        # img = clipseScaleSArray(img, 0, 100)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}",img*255 )


    def save_img_with_contorus(self, img, lab, dir, name):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)
        img=cv2.resize(img, (500, 500))
        lab=cv2.resize(lab, (500, 500),interpolation=cv2.INTER_NEAREST)
        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        contours = extract_semantic_contour(lab)
        img = draw_coutours(np.tile(np.expand_dims(img, -1), (1, 1, 3)), contours,my_color)



        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}",img )


    def save_img_with_contorusV2(self, img, lab, dir, name,color=[(1, 0, 0)]):
        mkdir_if_not_exist(dir)


        img = np.squeeze(img).astype(np.float32)
        lab = np.squeeze(lab).astype(np.float32)
        img=cv2.resize(img, (500, 500))
        lab=cv2.resize(lab, (500, 500),interpolation=cv2.INTER_NEAREST)
        img = clipseScaleSArray(img, 0, 100).astype('uint8')
        img = segmentation.mark_boundaries(img, lab.astype(np.uint8),color=color,outline_color=color, mode='thick')


        # img=sitk.GetImageFromArray(img)
        # img=clipseScaleSitkImage(img,0,100)
        # img=sitk.GetArrayFromImage(img).astype('uint8')
        # contours = extract_semantic_contour(lab)
        # img = draw_coutours(np.tile(np.expand_dims(img, -1), (1, 1, 3)), contours,
        #                     my_color)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}",img*255 )

    def save_image_with_pred_gt_contous(self, img, pred_lab, gt_lab, dir, name):
        self.save_img_with_mv_fix_contorus(img, pred_lab, gt_lab, dir, name)

    def save_image_with_pred_gt_contousV2(self, img, pred_lab, gt_lab, dir, name,colora=[(0, 0, 1)],colorb=[(1, 0, 0)]):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        pred_lab = np.squeeze(pred_lab).astype(np.float32)
        gt_lab = np.squeeze(gt_lab).astype(np.float32)
        pred_lab=cv2.resize(pred_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
        gt_lab=cv2.resize(gt_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
        img=cv2.resize(img, (500, 500))
        img=clipseScaleSArray(img,0,100).astype(np.uint8)
        img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=colora,outline_color=colora, mode='thick')
        img = segmentation.mark_boundaries(img, gt_lab.astype(np.uint8), color=colorb,outline_color=colorb, mode='thick')
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}", img*255)

    def save_img_with_mv_fix_contorusV2(self, img, mv_lab, fix_lab, dir, name,colora=[(0, 0, 1)],colorb=[(1, 0, 0)]):
        self.save_image_with_pred_gt_contousV2(img, mv_lab, fix_lab, dir, name,colora,colorb)

    def save_img_with_mv_fix_contorus(self, img, mv_lab, fix_lab, dir, name):
        mkdir_if_not_exist(dir)
        img = np.squeeze(img).astype(np.float32)
        mv_lab = np.squeeze(mv_lab).astype(np.float32)
        img=cv2.resize(img, (500, 500))
        mv_lab=cv2.resize(mv_lab, (500, 500))
        fix_lab = np.squeeze(fix_lab).astype(np.float32)
        fix_lab = cv2.resize(fix_lab, (500, 500))
        img=sitk.GetImageFromArray(img)
        img=clipseScaleSitkImage(img,0,100)
        img=sitk.GetArrayFromImage(img).astype('uint8')
        mv_contours = extract_semantic_contour(mv_lab)
        fix_contous=extract_semantic_contour(fix_lab)
        img = draw_coutours(np.tile(np.expand_dims(img, -1), (1, 1, 3)),
                            mv_contours,
                            my_color)
        img = draw_coutours(img,
                            fix_contous,
                            my_color)
        # img = cv2.flip(img, 0)
        cv2.imwrite(f"{dir}/{name}",img )



    def print_res(self, seg_ds, seg_hds,task='seg'):
        for k in seg_ds.keys():
            if (len(seg_ds[k])) > 0:
                # print(ds[k])
                logging.info(f'subject level evaluation:  DS {k}: {np.mean(seg_ds[k])} {np.std(seg_ds[k])}')
                write_array(self.args.res_excel, f'myops_asn_{task}_{k}_ds', seg_ds[k])
                # print(hds[k])
                logging.info(f'subject level evaluation:  HD {k}: {np.mean(seg_hds[k])} {np.std(seg_hds[k])}')
                write_array(self.args.res_excel, f'myops_asn_{task}_{k}_hd', seg_hds[k])

    def cal_ds_hd(self, gd_paths, pred_paths, roi_labs={1: [1220, 2221, 200, 500]}):
        seg_gds_list = []
        seg_preds_list = []
        assert len(gd_paths) == len(pred_paths)
        for gd, pred in zip(gd_paths, pred_paths):
            # print(f"{gd}-{pred}")
            seg_gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)), axis=0))
            seg_preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)), axis=0))
        gds_arr = np.concatenate(seg_gds_list, axis=0)
        preds_arr = np.concatenate(seg_preds_list, axis=0)
        gds_arr = np.squeeze(reindex_label_array_by_dict(gds_arr, roi_labs))
        preds_arr = np.squeeze(preds_arr)
        ds_res = dc(gds_arr, preds_arr)
        if len(gds_arr.shape) == 2:
            gds_arr = np.expand_dims(gds_arr, axis=-1)
            preds_arr = np.expand_dims(preds_arr, axis=-1)


        suject="_".join(os.path.basename(gd_paths[0]).split('.')[0].split("_")[:-1])

        gd3d=sort_glob(f"../data/gen_{self.args.data_source}/croped/{suject}.nii.gz")
        para=sitk.ReadImage(gd3d[0])
        # hds[modality].append(hd95(gds_arr,preds_arr,para.GetSpacing()))

        hd_res = hd(gds_arr, preds_arr, (para.GetSpacing()[-1],para.GetSpacing()[1],para.GetSpacing()[0]))
        asd_res = asd(gds_arr, preds_arr, (para.GetSpacing()[-1],para.GetSpacing()[1],para.GetSpacing()[0]))
        return ds_res, hd_res,asd_res
    def __init__(self,args):
        super().__init__(args)