import logging

from torch.utils.data import DataLoader

from config.nSD import nSD_Config
from tools.compute_transmurality import get_path_s, get_path_sv2
from tools.dir import mkdir_if_not_exist

from dataloader.nsdsegdataset import  n_SDDataset as DataSetLoader
from dataloader.util import SkimageOP_MSCMR
from experiment.baseexperiment import BaseMSCMRExperiment
from jrs_networks.jrs_tps_seg_net import JRS3TpsSegNet

from tools.dir import sort_time_glob, sort_glob, mk_or_cleardir,parent_dir
from tools.evaluation import myo_infarct_size, myo_edema_size
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.set_random_seed import worker_init_fn
import skimage.morphology as sm
from skimage.morphology import square, erosion
import os
import itertools

from tools.itkdatawriter import sitk_write_array_as_nii
import SimpleITK as sitk
import numpy as np
import skimage
from tools.np_sitk_tools import clipseScaleSArray
import cv2

from skimage.morphology import disk, erosion, binary_dilation, skeletonize

def my_collate(batch):
    # real_batch=np.array(batch)
    # return real_batch
    batch=batch[0]
    return np.squeeze(batch[0]),np.squeeze(batch[1]),np.squeeze(batch[2]),np.squeeze(batch[3]),np.squeeze(batch[4]),np.squeeze(batch[5]),batch[6],batch[7],batch[8]


from conventional_seg.nSD_seg import get_seed_point,image_random_region,get_center_line,image_patch_region,show_cv_array

class Experiment_n_SD(BaseMSCMRExperiment):
    def __init__(self,args:nSD_Config):
        super().__init__(args)
        self.args=args

        # self.model = JRS3TpsSegNet(args)
        self.op = SkimageOP_MSCMR()
        # if args.load:
        #     if self.args.ckpt==-1:
        #         model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[-1]
        #     else:
        #         model_path = sort_time_glob(args.checkpoint_dir + f"/*{self.args.ckpt}*.pth")[-1]
        #     # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
        #     logging.info(f'Model loaded from : {args.load} {model_path}')
        #     self.model = self.model.load(model_path, self.device)
        # self.model.to(device= self.device)
        #
        # r1 = self.args.span_range_height
        # r2 = self.args.span_range_width
        # control_points = np.array(list(itertools.product(
        #     np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.args.grid_height - 1)),
        #     np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.args.grid_width - 1)),
        # )))
        # self.base_control_points=np.zeros_like(control_points)
        # self.base_control_points[:,0]=control_points[:,1]
        # self.base_control_points[:,1]=control_points[:,0]

        # self.val_loader = DataLoader(DataSetLoader(args, type="test",augo=False,  task='pathology'),
        #                              batch_size=1,
        #                              shuffle=False,
        #                              num_workers=4,
        #                              pin_memory=True,
        #                              worker_init_fn=worker_init_fn)

        # self.val_loader = DataLoader(DataSetLoader(args, type="all",augo=False,  task='pathology'),
        #                              batch_size=1,
        #                              shuffle=False,
        #                              num_workers=4,
        #                              pin_memory=True,
        #                              worker_init_fn=worker_init_fn)

        # r1 = self.args.span_range_height
        # r2 = self.args.span_range_width
        # control_points = np.array(list(itertools.product(
        #     np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.args.grid_height - 1)),
        #     np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.args.grid_width - 1)),
        # )))
        # self.base_control_points=np.zeros_like(control_points)
        # self.base_control_points[:,0]=control_points[:,1]
        # self.base_control_points[:,1]=control_points[:,0]

        logging.info(f'''Starting training:
            modality:          {self.args.modality}
            SD:          {self.args.nSD}

        ''')

    #TMI的论文方法
    def seg_res(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        print(f"Segmenting.....................................")
        self.val_loader = DataLoader(DataSetLoader(self.args, type="test",augo=False,  task='pathology',ret_name=True),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn,
                                     collate_fn=my_collate)




        for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
            #
            # if  c0_path[0].find('33')<0:
            #     continue

            if self.args.modality.lower()=='t2':
                # target_img = clipseScaleSArray(img_t2,0,100)
                target_img = img_t2
                print(t2_path)
                gt_lab = lab_t2
                pre_name =t2_path
            elif self.args.modality.lower()=="lge":
                # target_img=clipseScaleSArray(img_de,0,100)
                target_img=img_de
                print(de_path)
                gt_lab=lab_de
                pre_name=de_path
            else:
                exit("-932")

            if self.args.modality.lower()=="lge":
                patch=get_path_s(gt_lab,target_img,dialed=3)
            else:
                patch = get_path_s(gt_lab, target_img,dialed=2)
            # healthy_myo=reindex_label_array_by_dict(gt_lab,{2:[200]})
            #random nSD seed

            mean, sd = self.get_mean_sd_global(gt_lab, target_img)

            ind = np.argwhere(patch == 1)
            ind = ind[np.random.choice(ind.shape[0], ind.shape[0]), :]
            ind = tuple(map(tuple, np.transpose(ind)))
            health_pixels = []
            health_pixels.append(target_img[ind])
            # mean=   np.mean(health_pixels)
            sd=    np.std(health_pixels)

            myo = reindex_label_array_by_dict(gt_lab, {1: [1220,2221,200,1]})
            footprint = disk(1)
            ero_myo = erosion(myo, footprint)
            ind = np.argwhere(ero_myo == 1)

            # ind1 = tuple(map(tuple, np.transpose(ind)))

            new_array = np.zeros(target_img.shape, np.uint8)

            for j in ind:
                if (target_img[j[0], j[1]] - mean) > sd * self.args.nSD:
                    new_array[j[0], j[1]] = 1

            subdir=os.path.basename(os.path.dirname(pre_name))
            output=os.path.join(self.args.gen_dir,subdir)
            mkdir_if_not_exist(output)

            #filter small object
            if self.args.modality=='LGE' or self.args.modality=='lge':
                new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#scar [(0.5 70) 0.4]  0.2 70
            else:
                new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#edema 50

            print(f"output:{output}/{os.path.basename(pre_name)}")

            sitk_write_array_as_nii(np.expand_dims(new_array,0),sitk.ReadImage(pre_name),output,os.path.basename(pre_name),islabel=True)
    #
    # def seg_resV2(self):
    #     """
    #     Evaluation without the densecrf with the dice coefficient
    #     """
    #     print(f"Segmenting.....................................")
    #     self.val_loader = DataLoader(DataSetLoader(self.args, type="test",augo=False,  task='pathology',ret_name=True),
    #                                  batch_size=1,
    #                                  shuffle=False,
    #                                  num_workers=1,
    #                                  pin_memory=True,
    #                                  worker_init_fn=worker_init_fn,
    #                                  collate_fn=my_collate)
    #
    #
    #
    #
    #     for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
    #         #
    #         # if  c0_path[0].find('33')<0:
    #         #     continue
    #
    #
    #
    #         if self.args.modality.lower()=='t2':
    #             # target_img = clipseScaleSArray(img_t2,0,100)
    #             target_img = img_t2
    #             print(t2_path)
    #             gt_lab = lab_t2
    #             pre_name =t2_path
    #         elif self.args.modality.lower()=="lge":
    #             # target_img=clipseScaleSArray(img_de,0,100)
    #             target_img=img_de
    #             print(de_path)
    #             gt_lab=lab_de
    #             pre_name=de_path
    #         else:
    #             exit("-932")
    #
    #         mean, sd = self.get_mean_sd_global(gt_lab, target_img)
    #
    #         myo = reindex_label_array_by_dict(gt_lab, {1: [1220,2221,200,1]})
    #         ind = np.argwhere(myo == 1)
    #         # ind1 = tuple(map(tuple, np.transpose(ind)))
    #
    #         new_array = np.zeros(target_img.shape, np.uint8)
    #
    #         for j in ind:
    #             if (target_img[j[0], j[1]] - mean) > (sd-100) * self.args.nSD:
    #                 new_array[j[0], j[1]] = 1
    #
    #         subdir=os.path.basename(os.path.dirname(pre_name))
    #         output=os.path.join(self.args.gen_dir,subdir)
    #         mkdir_if_not_exist(output)
    #
    #         #filter small object
    #         if self.args.modality=='LGE':
    #             new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#scar [(0.5 70) 0.4]  0.2 70
    #         else:
    #             new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#edema 50
    #
    #         print(f"output:{output}/{os.path.basename(pre_name)}")
    #
    #         sitk_write_array_as_nii(np.expand_dims(new_array,0),sitk.ReadImage(pre_name),output,os.path.basename(pre_name),islabel=True)

    def seg_resV3(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        print(f"Segmenting.....................................")
        self.val_loader = DataLoader(DataSetLoader(self.args, type="test",augo=False,  task='pathology',ret_name=True),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn,
                                     collate_fn=my_collate)




        for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
            #
            # if  c0_path[0].find('33')<0:
            #     continue



            if self.args.modality.lower()=='t2':
                # target_img = clipseScaleSArray(img_t2,0,100)
                target_img = img_t2
                # print(t2_path)
                gt_lab = lab_t2
                pre_name =t2_path
            elif self.args.modality.lower()=="lge":
                # target_img=clipseScaleSArray(img_de,0,100)
                target_img=img_de
                # print(de_path)
                gt_lab=lab_de
                pre_name=de_path
            else:
                exit("-932")

            mean, sd = self.get_mean_sd_local(gt_lab, target_img)
            print(f"mean={mean} std:={sd}")
            myo = reindex_label_array_by_dict(gt_lab, {1: [1220,2221,200]})
            ind = np.argwhere(myo == 1)
            # ind1 = tuple(map(tuple, np.transpose(ind)))

            new_array = np.zeros(target_img.shape, np.uint8)

            for j in ind:
                if (target_img[j[0], j[1]] - mean) > (sd) * self.args.nSD:
                    new_array[j[0], j[1]] = 1

            subdir=os.path.basename(os.path.dirname(pre_name))
            output=os.path.join(self.args.gen_dir,subdir)
            mkdir_if_not_exist(output)

            #filter small object
            if self.args.modality=='lge':
                new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#scar [(0.5 70) 0.4]  0.2 70
            else:
                new_array=skimage.morphology.remove_small_objects(new_array>0,min_size=self.args.minsize)#edema 50

            print(f"output:{output}/{os.path.basename(pre_name)}")

            sitk_write_array_as_nii(np.expand_dims(new_array,0),sitk.ReadImage(pre_name),output,os.path.basename(pre_name),islabel=True)
            # sitk_write_array_as_nii(np.expand_dims(target_img,0),sitk.ReadImage(pre_name),output,"img"+os.path.basename(pre_name))


    def get_mean_sd_global(self, gt_lab, target_img):
        healthy_myo = reindex_label_array_by_dict(gt_lab, {2: [200]})
        # random nSD seed
        ind = np.argwhere(healthy_myo == 2)
        ind = ind[np.random.choice(ind.shape[0], 100), :]
        ind = tuple(map(tuple, np.transpose(ind)))
        health_pixels = []
        health_pixels.append(target_img[ind])
        mean = np.mean(health_pixels)
        sd = np.std(health_pixels)
        return mean, sd


    def get_min(self,cl,img):
        ind = np.argwhere(cl == 1)
        x,y=-1,-1
        min=99999
        for i,j in ind:
            if img[x][y]<min:
                x=i
                y=j
                min=img[i][j]
        return x,y
    def get_mean_sd_local(self, gt_lab, target_img):

        healthy_myo = reindex_label_array_by_dict(gt_lab, {1: [200,1220,2221]})
        show_cv_array(healthy_myo)
        cl=get_center_line(gt_lab,[2221,1220,200])
        show_cv_array(cl)
        cl=cl*healthy_myo
        x,y=self.get_min(cl,target_img)


        path_mask=image_patch_region(x,y,gt_lab,8)
        show_cv_array(gt_lab)
        show_cv_array(path_mask,'patch')
        ind = np.argwhere(path_mask == 1)
        ind = tuple(map(tuple, np.transpose(ind)))
        health_pixels = []
        health_pixels.append(target_img[ind])

        mean = np.mean(health_pixels)
        sd = np.std(health_pixels)
        return mean, sd



        # random nSD seed


    def evalue_res(self):
        pred_dir=sort_glob(f"{self.args.gen_dir}/*")
        gt_dir=sort_glob(f"{self.args.dataset_dir}/*")[20:]
        ref_3D=sort_glob(f"../data/gen_ZS_unaligned/croped/*img_de*")

        if self.args.modality.lower()=="lge" :
            myo_infarct_size(self.args.model_id,pred_dir,gt_dir,ref_3D,"de")
        else:
            myo_edema_size(self.args.model_id,pred_dir,gt_dir,ref_3D,"t2")
