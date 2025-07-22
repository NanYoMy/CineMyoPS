import logging
import os
from os import path as osp

import SimpleITK as sitk
import cv2
import numpy as np
import torch
from medpy.metric import dc, hd95
from skimage.exposure import rescale_intensity
from skimage.util import compare_images
from torch import optim
from torch.utils.data import DataLoader

from baseclass.medicalimage import Modality
from dataloader.jrsdataset import DataSetRJ as DataSetLoader
from dataloader.util import SkimageOP_MSCMR
from experiment.baseexperiment import BaseMyoPSExperiment
from jrs_networks.jrs_mid_seg_net import MIDROISegNet as MIDNet
from tools.dir import sort_time_glob, mk_or_cleardir, mkdir_if_not_exist, sort_glob
from tools.excel import write_array
from tools.np_sitk_tools import reindex_label_array_by_dict, clipseScaleSitkImage
from tools.set_random_seed import worker_init_fn
from tools.tps_painter import save_image_with_tps_points
from tools.nii_lab_to_png import save_img
from experiment.baseexperiment import BaseMSCMRExperiment
from tools.model_static import static_model
class Experiment_MID_RJ(BaseMSCMRExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.args=args

        self.model = MIDNet(args)
        self.op = SkimageOP_MSCMR()
        if args.load:
            model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[-1]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            logging.info(f'Model loaded from : {args.load} {model_path}')
            self.model = self.model.load(model_path, self.device)
        self.model.to(device= self.device)
        static_model(self.model)
        self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology'),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(DataSetLoader(args, type="test",augo=False,  task='pathology'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)

        self.val_loader = DataLoader(DataSetLoader(args, type="all",augo=False,  task='pathology'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)



        logging.info(f'''Starting training:
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Optimizer:       {self.args.optimizer}
            Checkpoints:     {self.args.save_cp}
            Device:          {self.device.type}
            load:           {self.args.load}
        ''')


    def validate_net(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        self.val_loader = DataLoader(DataSetLoader(self.args, type="all",augo=False,  task='pathology'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)


        size=(256,256)
        import os
        from tools.dir import mkdir_if_not_exist
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
                #
                # if  c0_path[0].find('33')<0:
                #     continue


                c0_lab_path=c0_path[0].replace("img_c0","ana_c0")
                t2_lab_path=t2_path[0].replace('img_t2','ana_patho_t2_edema')
                de_lab_path=de_path[0].replace('img_de','ana_patho_de_scar')

                img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv = self.create_torch_tensor(img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de)
                # warp_pred = {}
                pred={}
                para={}
                ori_img={}
                ori_lab={}
                # ums_img={}
                # warp_ori_img = {}
                # warp_ori_lab={}
                # train
                c0_seg,t2_seg,lge_seg= self.model(img[Modality.c0],img[Modality.t2], img[Modality.de])

                para['c0'] = sitk.ReadImage(c0_path)
                ori_img['c0'] = sitk.GetArrayFromImage(para['c0'])
                # ori_img['c0'] = self.op.usm(ori_img['c0'])
                ori_lab['c0']=sitk.GetArrayFromImage(sitk.ReadImage(c0_lab_path))

                para['t2'] = sitk.ReadImage(t2_path)
                ori_img['t2'] = sitk.GetArrayFromImage(para['t2'])
                # ori_img['t2'] = self.op.usm(ori_img['t2'])
                ori_lab['t2']=sitk.GetArrayFromImage(sitk.ReadImage(t2_lab_path))

                para['de'] = sitk.ReadImage(de_path)
                ori_img['de'] = sitk.GetArrayFromImage(para['de'])
                # ori_img['de'] = self.op.usm(ori_img['de'])
                ori_lab['de']=sitk.GetArrayFromImage(sitk.ReadImage(de_lab_path))


                # warp_pred[Modality.c0]=self.model.warp(c0_seg, theta_c0)
                # warp_pred[Modality.t2]=self.model.warp(t2_seg, theta_t2)
                # print(de_path)
                # print(theta_c0.cpu().numpy()[0,:,:]-self.base_control_points)
                # print(theta_t2.cpu().numpy()[0,:,:]-self.base_control_points)

                # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
                # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
                pred['de']=torch.argmax(lge_seg,dim=1,keepdim=True)
                pred['c0']=torch.argmax(c0_seg,dim=1,keepdim=True)
                pred['t2']=torch.argmax(t2_seg,dim=1,keepdim=True)

                # warp_pred[Modality.c0]=torch.argmax(warp_pred[Modality.c0],dim=1,keepdim=True)
                # warp_pred[Modality.t2]=torch.argmax(warp_pred[Modality.t2],dim=1,keepdim=True)


                # warp_ori_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_img['c0'], size), axis=0)).cuda(), theta_c0)
                # warp_ori_lab[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_lab['c0'], size,order=0), axis=0)).cuda(), theta_c0,mode='nearest')
                #
                # warp_ori_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_img['t2'], size), axis=0)).cuda(), theta_t2)
                # warp_ori_lab[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_lab['t2'], size,order=0), axis=0)).cuda(), theta_t2,mode='nearest')


                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                if self.args.save_imgs==True:
                    # self.save_diff_img(ori_img['c0'],ori_img['de'],output+"_diff",self.renamepath(c0_path, 'diff_img')+".png")
                    # self.save_diff_img(ori_img['t2'],ori_img['de'],output+"_diff",self.renamepath(t2_path, 'diff_img')+".png")
                    # self.save_diff_img(warp_ori_img[Modality.c0].cpu().numpy(),ori_img['de'],output+"_diff",self.renamepath(c0_path, 'diff_warp_img')+".png")
                    # self.save_diff_img(warp_ori_img[Modality.t2].cpu().numpy(),ori_img['de'],output+"_diff",self.renamepath(t2_path, 'diff_warp_img')+".png")

                    # self.save_img_with_mv_fix_contorus(ori_img['c0'],
                    #                                    reindex_label_array_by_dict(ori_lab['c0'],{2:[1,200,1220,2221,500]}),
                    #                                    reindex_label_array_by_dict(ori_lab['de'], {1: [1, 200, 1220,
                    #                                                                                    2221, 500]}),
                    #                                    output +"_contours", self.renamepath(c0_path, 'gt_con_img') +".png")
                    # self.save_img_with_mv_fix_contorus(ori_img['t2'],
                    #                             reindex_label_array_by_dict(ori_lab['t2'],{2:[1,200,1220,2221,500]}),
                    #                             reindex_label_array_by_dict(ori_lab['de'],{1: [1, 200, 1220, 2221, 500]}),
                    #                             output+"_contours",self.renamepath(t2_path, 'gt_con_img')+".png")
                    self.save_img_with_mask(ori_img['de'],
                                                reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
                                                output + "_contours", self.renamepath(de_path, 'gt_con_img') + ".png")

                    # self.save_img_with_mv_fix_contorus(warp_ori_img[Modality.c0].cpu().numpy(),
                    #                                    reindex_label_array_by_dict(warp_ori_lab[Modality.c0].cpu().numpy(),{2:[1,200,1220,2221,500]}),
                    #                                    reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
                    #                                    output+"_reg_contours",
                    #                                    self.renamepath(c0_path, 'warp_img_fix_mv')+".png")

                    # self.save_img_with_mv_fix_contorus(warp_ori_img[Modality.t2].cpu().numpy(),
                    #                                    reindex_label_array_by_dict(warp_ori_lab[Modality.t2].cpu().numpy(),{2:[1,200,1220,2221,500]}),
                    #                                    reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
                    #                                    output+"_reg_contours",
                    #                                    self.renamepath(t2_path, 'warp_img_fix_mv')+".png")

                    self.save_image_with_pred_gt_contousV2(ori_img['c0'],
                                                         reindex_label_array_by_dict(pred['c0'].cpu().numpy(),
                                                                                     {3: [1, 200, 1220, 2221, 500]}),
                                                         reindex_label_array_by_dict(ori_lab['c0'],
                                                                                     {1: [1, 200, 1220, 2221, 500]}),
                                                       output+"_seg_contours",
                                                       self.renamepath(c0_path, 'warp_img_pre')+".png")

                    self.save_image_with_pred_gt_contousV2(ori_img['t2'],
                                                         reindex_label_array_by_dict(pred['t2'].cpu().numpy(),
                                                                                     {3: [1, 200, 1220, 2221, 500]}),
                                                         reindex_label_array_by_dict(ori_lab['t2'],
                                                                                     {1: [1, 200, 1220, 2221, 500]}),
                                                       output+"_seg_contours",
                                                       self.renamepath(t2_path, 'warp_img_pre')+".png")

                    self.save_image_with_pred_gt_contousV2(ori_img['de'],
                                                       reindex_label_array_by_dict(pred['de'].cpu().numpy(),{3:[1,200,1220,2221,500]}),
                                                       reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
                                                       output+"_seg_contours",
                                                       self.renamepath(de_path, 'warp_img_pre')+".png")


                # self.save_tensor_with_parameter(warp_ori_img[Modality.c0], para['c0'],output , self.renamepath(c0_path, 'assn_img'))
                # self.save_tensor_with_parameter(warp_pred[Modality.c0], para['c0'],output, self.renamepath(c0_path, 'assn_lab'),is_label=True)
                # self.save_tensor_with_parameter(warp_ori_lab[Modality.c0], para['c0'],output, self.renamepath(c0_path, 'assn_gt_lab'),is_label=True)
                #
                # self.save_tensor_with_parameter(warp_ori_img[Modality.t2], para['t2'],output , self.renamepath(t2_path, 'assn_img'))
                # self.save_tensor_with_parameter(warp_pred[Modality.t2], para['t2'],output, self.renamepath(t2_path, 'assn_lab'),is_label=True)
                # self.save_tensor_with_parameter(warp_ori_lab[Modality.t2], para['t2'],output, self.renamepath(t2_path, 'assn_gt_lab'),is_label=True)
                #

                # self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_img['de']), para['de'],output, self.renamepath(de_path, 'assn_img'))
                # self.save_tensor_with_parameter(pred['de'], para['de'],output, self.renamepath(de_path, 'assn_lab'),is_label=True)
                # self.save_tensor_with_parameter(ori_lab['de'], para['de'],output, self.renamepath(de_path, 'assn_gt_lab'),is_label=True)

                self.save_tensor_with_parameter(pred['c0'], para['c0'],output, self.renamepath(c0_path, 'branch_lab'),is_label=True)
                self.save_tensor_with_parameter(pred['t2'], para['t2'],output, self.renamepath(t2_path, 'branch_lab'),is_label=True)
                self.save_tensor_with_parameter(pred['de'], para['de'],output, self.renamepath(de_path, 'branch_lab'),is_label=True)


        # logging.info(f'slice level evaluation reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
        # logging.info(f'slice level evaluation seg mv:{np.mean(seg_error["mv"])} | fix: {np.mean(seg_error["fix"])}')

        seg_ds = {"c0": [], "t2": [], "de": []}
        seg_hds = {"c0": [], "t2": [], "de": []}
        reg_ds = {"c0": [], "t2": [], "de": []}
        reg_hds = {"c0": [], "t2": [], "de": []}

        for dir in range(1026, 1051):
            for modality in ['c0','t2', 'de']:
                seg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
                seg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_branch_lab*nii.gz")

                # reg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*de_*nii.gz")
                # reg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")

                print(f"================>{dir}")
                if len(seg_gds) == 0:
                    continue
                try:
                    ds_res, hd_res = self.cal_ds_hd(seg_gds, seg_preds, {1: [1220, 2221, 200,500]})
                    seg_ds[modality].append(ds_res)
                    seg_hds[modality].append(hd_res)

                    # ds_res, hd_res = self.cal_ds_hd(reg_gds, reg_preds, {1: [1220, 2221, 200,500]})
                    # reg_ds[modality].append(ds_res)
                    # reg_hds[modality].append(hd_res)
                except Exception as e:
                    print(e)
                    logging.error(e)

        print("=========segmentation=================")
        self.print_res(seg_ds, seg_hds, 'seg')
        # print("=========registration=================")
        # self.print_res(reg_ds, reg_hds, 'reg')

    def train_eval_net(self):
        global_step = 0
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.args.lr)
        elif self.args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=self.args.lr,
                                      weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=self.args.nesterov)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_decay_milestones,
                                                   gamma = self.args.lr_decay_gamma)


        # crit_reg=BinaryDiceLoss()
        from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
        from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
        from nnunet.utilities.nd_softmax import softmax_helper
        # crit_reg=BinaryGaussianDice()
        regcrit=BinaryGaussianDice(sigma=1)
        # segcrit=DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
        self.model.train() # dropout, BN在train和test的行为不一样
        mk_or_cleardir(self.args.log_dir)
        # loss=0

        for epoch in range(self.args.init_epochs,self.args.epochs+1):
            sts_total_loss = 0
            np.random.seed(17 + epoch)
            #前1000个epoch只进行配准

            print("train.................")
            for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
                #load data
                img, lab, roi_lab_myo, roi_lab_lv,roi_lab_rv= self.create_torch_tensor(img_c0, img_t2,img_de, lab_c0, lab_t2, lab_de)
                # warp_img={}
                # warp_roi_lab_myo={}
                sts_loss = {}
                # warp_roi_lab_lv={}
                # warp_roi_lab_rv={}
                # train
                c0_seg,t2_seg,lge_seg= self.model(img[Modality.c0],img[Modality.t2], img[Modality.de])

                #loss
                # warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
                # warp_roi_lab_myo[Modality.t2] = self.model.warp(roi_lab_myo[Modality.t2], theta_t2)
                #
                # warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta_c0)
                # warp_roi_lab_lv[Modality.t2] = self.model.warp(roi_lab_lv[Modality.t2], theta_t2)
                #
                # warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta_c0)
                # warp_roi_lab_rv[Modality.t2] = self.model.warp(roi_lab_rv[Modality.t2], theta_t2)
                #
                #
                # warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta_c0)
                # warp_img[Modality.t2] = self.model.warp(img[Modality.t2], theta_t2)

                # loss_reg_myo = regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.c0])+regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.t2])
                # loss_reg_lv = regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.c0])+regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.t2])
                # loss_reg_rv = regcrit(roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.c0])+regcrit(roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.t2])
                #分割目标是myo
                # loss_seg_lge = segcrit(lge_seg, roi_lab_myo[Modality.de])
                # loss_seg_c0 = segcrit(c0_seg, roi_lab_myo[Modality.c0])
                # loss_seg_t2 = segcrit(t2_seg, roi_lab_myo[Modality.t2])

                loss_seg_lge = segcrit(lge_seg, roi_lab_lv[Modality.de])
                loss_seg_c0 = segcrit(c0_seg, roi_lab_lv[Modality.c0])
                loss_seg_t2 = segcrit(t2_seg, roi_lab_lv[Modality.t2])

                # loss_all=(loss_seg_lge+loss_seg_c0+2*loss_seg_t2)+self.args.weight*(loss_reg_myo+loss_reg_lv+loss_reg_rv)
                loss_all=(loss_seg_lge+loss_seg_c0+2*loss_seg_t2)
                # loss_all=(2*loss_seg_lge+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_myo+loss_reg_lv)
                #statistic

                # sts_loss["train/reg_myo"]=(loss_reg_myo.item())
                # sts_loss["train/reg_lv"]=(loss_reg_lv.item())
                sts_loss["train/loss_seg_lge"]=(loss_seg_lge.item())
                sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
                sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
                sts_loss["train/reg_total"]=(loss_all.item())

                optimizer.zero_grad()
                loss_all.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
                optimizer.step()
                global_step += 1

                self.write_dict_to_tb(sts_loss,global_step)

            scheduler.step()
            if (epoch)%self.args.save_freq==0:
                try:
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                    # self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)
                    self.eval_writer.add_images("train/labs/lv_de", roi_lab_lv[Modality.de], global_step)
                    self.eval_writer.add_images("train/labs/lv_t2", roi_lab_lv[Modality.t2], global_step)
                    self.eval_writer.add_images("train/labs/lv_c0", roi_lab_lv[Modality.c0], global_step)
                    self.eval_writer.add_images("train/labs/myo_de", roi_lab_myo[Modality.de], global_step)
                    self.eval_writer.add_images("train/labs/myo_t2", roi_lab_myo[Modality.t2], global_step)
                    self.eval_writer.add_images("train/labs/myo_c0", roi_lab_myo[Modality.c0], global_step)

                    # evaluation befor save checkpoints
                    self.model.eval()
                    self.validate_net()
                    self.model.train()
                    ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                    self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                except Exception as e:
                    print(e)
                    logging.error(f'Checkpoint {epoch + 1} {e} ')


        self.eval_writer.close()

    def gen(self):
        print("generating................")
        #
        # # reg_error = {"init": [], "reg": []}
        # # seg_error = {"mv": [], "fix": []}
        # # with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        # i = 0
        # with torch.no_grad():
        #     for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path, t2_path, de_path in self.val_loader:
        #         img, lab, roi_lab_myo, roi_lab_rv = self.create_torch_tensor(img_c0,  img_t2,img_de, lab_c0,lab_t2, lab_de)
        #         warp_img = {}
        #         warp_lab = {}
        #         warp_roi_lab_myo = {}
        #         warp_roi_lab_rv = {}
        #         # train
        #         c0_seg, t2_seg,lge_seg, theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
        #         c0_seg = torch.argmax(c0_seg, dim=1, keepdim=True)
        #         t2_seg = torch.argmax(t2_seg, dim=1, keepdim=True)
        #         lge_seg = torch.argmax(lge_seg, dim=1, keepdim=True)
        #
        #         warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta_c0)
        #         warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], theta_c0, mode='nearest')
        #
        #         warp_img[Modality.t2] = self.model.warp(img[Modality.t2], theta_t2)
        #         warp_lab[Modality.t2] = self.model.warp(lab[Modality.t2], theta_c0, mode='nearest')
        #
        #
        #         warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta_c0, mode='nearest')
        #         warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta_c0, mode='nearest')
        #
        #         # bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask
        #         #warped c0
        #         self.save_torch_to_nii(self.args.gen_dir,warp_img[Modality.c0],c0_path, 'c0_ori',is_lab=False)
        #         self.save_torch_to_nii(self.args.gen_dir,warp_roi_lab_myo[Modality.c0],c0_path, 'c0_ori',is_lab=True)
        #         self.save_torch_to_nii(self.args.gen_dir, c0_seg, c0_path, "c0_pred",is_lab=True)
        #         #de
        #         self.save_torch_to_nii(self.args.gen_dir,img[Modality.de],de_path, 'de_ori',is_lab=False)
        #
        #         myo_scar_mask = lab_de.narrow(dim=1, start=6, length=1)
        #         self.save_torch_to_nii(self.args.gen_dir,myo_scar_mask,de_path, 'de_ori',is_lab=True)
        #         self.save_torch_to_nii(self.args.gen_dir, lge_seg, c0_path, "de_pred",is_lab=True)
        #

#
# class Experiment_MID_MSCMR(BaseMyoPSExperiment):
#     def __init__(self,args):
#         super().__init__(args)
#         self.args=args
#
#         self.model = MIDNet(args)
#
#         if args.load:
#             model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[args.ckpt]
#             # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
#             self.model = self.model.load(model_path, self.device)
#             logging.info(f'Model loaded from {args.load}')
#         self.model.to(device= self.device)
#
#         self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology',ret_path=False),
#                                        batch_size=args.batch_size,
#                                        shuffle=True,
#                                        num_workers=4,
#                                        pin_memory=True,
#                                        worker_init_fn=worker_init_fn)
#         # self.trainall_loader = DataLoader(DataSetLoader(args, type="trainall",augo=True, task='pathology',ret_path=False),
#         #                                batch_size=args.batch_size,
#         #                                shuffle=True,
#         #                                num_workers=4,
#         #                                pin_memory=True,
#         #                                worker_init_fn=worker_init_fn)
#
#         self.val_loader = DataLoader(DataSetLoader(args, type="valid",augo=False,  task='pathology',ret_path=True),
#                                      batch_size=1,
#                                      shuffle=False,
#                                      num_workers=4,
#                                      pin_memory=True,
#                                      worker_init_fn=worker_init_fn)
#
#         self.test_loader = DataLoader(DataSetLoader(args, type="test",augo=False,  task='pathology',ret_path=True),
#                                      batch_size=1,
#                                      shuffle=False,
#                                      num_workers=4,
#                                      pin_memory=True,
#                                      worker_init_fn=worker_init_fn)
#         self.op = SkimageOP_MSCMR()
#         logging.info(f'''Starting training:
#             Epochs:          {self.args.epochs}
#             Batch size:      {self.args.batch_size}
#             Learning rate:   {self.args.lr}
#             Optimizer:       {self.args.optimizer}
#             Checkpoints:     {self.args.save_cp}
#             Device:          {self.device.type}
#             load:           {self.args.load}
#             weight:          {self.args.weight}
#             log:            {self.args.log_dir}
#         ''')
#
#     def validate_net(self):
#         """
#         Evaluation without the densecrf with the dice coefficient
#         """
#         self.gen_valid()
#     def train_eval_net(self):
#         global_step = 0
#         if self.args.optimizer == 'Adam':
#             optimizer = optim.Adam(self.model.parameters(),
#                                    lr=self.args.lr)
#         elif self.args.optimizer == 'RMSprop':
#             optimizer = optim.RMSprop(self.model.parameters(),
#                                       lr=self.args.lr,
#                                       weight_decay=self.args.weight_decay)
#         else:
#             optimizer = optim.SGD(self.model.parameters(),
#                                   lr=self.args.lr,
#                                   momentum=self.args.momentum,
#                                   weight_decay=self.args.weight_decay,
#                                   nesterov=self.args.nesterov)
#
#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                                    milestones=self.args.lr_decay_milestones,
#                                                    gamma = self.args.lr_decay_gamma)
#
#
#         # crit_reg=BinaryDiceLoss()
#         from nnunet.training.loss_functions.dice_loss import SoftDiceLoss,SoftConsistentDiceLoss
#         from nnunet.utilities.nd_softmax import softmax_helper
#         # crit_reg=BinaryGaussianDice()
#         regcrit=SoftConsistentDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
#         # segcrit=DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
#         segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
#         self.model.train() # dropout, BN在train和test的行为不一样
#         mk_or_cleardir(self.args.log_dir)
#         # loss=0
#
#         for epoch in range(self.args.init_epochs,self.args.epochs+1):
#             sts_total_loss = 0
#             np.random.seed(17 + epoch)
#             #前1000个epoch只进行配准
#
#             print("train.................")
#             for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
#                 #load data
#                 img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0,  img_t2, img_de,lab_c0, lab_t2, lab_de)
#
#                 # train
#                 c0_pred,t2_pred,de_pred= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
#                 # c0_pred,de_pred,t2_pred,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.de], img[Modality.t2])
#                 #loss
#                 # warp_c0_pred = self.model.warp(c0_pred, theta_c0)
#                 # warp_t2_pred = self.model.warp(t2_pred, theta_t2)
#                 # warp_c0_lab= self.model.warp(roi_lab_lv[Modality.c0], theta_c0)
#                 # warp_t2_lab= self.model.warp(roi_lab_lv[Modality.t2], theta_t2)
#
#                 # loss_consis_c0 = regcrit(warp_c0_pred, de_pred)
#                 # loss_consis_t2 = regcrit(warp_t2_pred, de_pred)
#                 #
#                 # loss_reg_c0 = regcrit(to_one_hot(warp_c0_lab,2), to_one_hot(roi_lab_lv[Modality.de],2))
#                 # loss_reg_t2 = regcrit(to_one_hot(warp_t2_lab,2), to_one_hot(roi_lab_lv[Modality.de],2))
#
#
#                 loss_seg_c0 = segcrit(c0_pred, roi_lab_lv[Modality.c0])
#                 loss_seg_t2 = segcrit(t2_pred, roi_lab_lv[Modality.t2])
#                 loss_seg_de = segcrit(de_pred, roi_lab_lv[Modality.de])
#
#                 loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)#+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
#                 # loss_all=loss_seg_t2+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
#                 #statistic
#                 sts_loss={}
#                 # sts_loss["train/reg_pre_c0"]=(loss_consis_c0.item())
#                 # sts_loss["train/reg_pre_t2"]=(loss_consis_t2.item())
#                 # sts_loss["train/reg_c0"]=(loss_reg_c0.item())
#                 # sts_loss["train/reg_t2"]=(loss_reg_t2.item())
#
#                 sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
#                 sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
#                 sts_loss["train/loss_seg_de"]=(loss_seg_de.item())
#
#                 sts_loss["train/loss_total"]=(loss_all.item())
#
#
#                 optimizer.zero_grad()
#                 loss_all.backward()
#                 torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
#                 optimizer.step()
#                 global_step += 1
#
#                 self.write_dict_to_tb(sts_loss,global_step)
#
#             scheduler.step()
#             if (epoch+1)%self.args.save_freq==0 or epoch==0:
#                 try:
#                     print(sts_loss)
#                     self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
#                     self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
#                     self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
#
#                     self.eval_writer.add_images("train/labs/de", (roi_lab_lv[Modality.de]), global_step)
#                     self.eval_writer.add_images("train/labs/t2", (roi_lab_lv[Modality.t2]), global_step)
#                     self.eval_writer.add_images("train/labs/c0", (roi_lab_lv[Modality.c0]), global_step)
#
#                     # evaluation befor save checkpoints
#                     self.model.eval()
#                     self.validate_net()
#                     self.model.train()
#                     ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
#                     self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
#                     logging.info(f'Checkpoint {epoch + 1} saved !')
#                 except OSError:
#                     logging.info(f'Checkpoint {epoch + 1} save failed !')
#                     exit(-999)
#
#
#         self.eval_writer.close()
#
#     def gen_valid(self):
#         """
#         Evaluation without the densecrf with the dice coefficient
#         """
#         mk_or_cleardir(self.args.gen_dir)
#         print("staget ASSN output")
#         size=[self.args.image_size,self.args.image_size]
#         with torch.no_grad():
#             for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
#                 img, lab, roi_lab_myo, roi_lab_lv = self.create_torch_tensor(img_c0,img_t2, img_de,  lab_c0, lab_t2, lab_de)
#                 warp_img = {}
#                 warp_seg={}
#                 para={}
#                 ori_array={}
#                 c0_pred,t2_pred,de_pred= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
#                 # c0_pred, de_pred, t2_pred, theta_c0, theta_t2 = self.model(img[Modality.c0], img[Modality.de], img[Modality.t2])
#                 para['c0']=sitk.ReadImage(c0_path)
#                 ori_array['c0']=sitk.GetArrayFromImage(para['c0'])
#
#                 para['t2'] = sitk.ReadImage(t2_path)
#                 ori_array['t2'] = sitk.GetArrayFromImage(para['t2'])
#
#                 para['de'] = sitk.ReadImage(de_path)
#                 ori_array['de'] = sitk.GetArrayFromImage(para['de'])
#
#
#
#                 # warp_seg[Modality.c0]=self.model.warp(c0_pred, theta_c0)
#                 # warp_seg[Modality.t2]=self.model.warp(t2_pred, theta_t2)
#
#                 # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
#                 # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
#                 de_pred=torch.argmax(de_pred,dim=1,keepdim=True)
#                 t2_pred=torch.argmax(t2_pred,dim=1,keepdim=True)
#                 c0_pred=torch.argmax(c0_pred,dim=1,keepdim=True)
#                 # warp_seg[Modality.c0]=torch.argmax(warp_seg[Modality.c0],dim=1,keepdim=True)
#                 # warp_seg[Modality.t2]=torch.argmax(warp_seg[Modality.t2],dim=1,keepdim=True)
#                 #
#                 # warp_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(), theta_c0)
#                 # warp_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(), theta_t2)
#
#                 subdir=os.path.basename(os.path.dirname(c0_path[0]))
#                 output=os.path.join(self.args.gen_dir,subdir)
#                 mkdir_if_not_exist(output)
#
#                 # self.save_diff_img(ori_array['c0'],ori_array['de'],output+"_diff",self._rename(c0_path, 'diff_img')+".png")
#                 # self.save_diff_img(ori_array['t2'],ori_array['de'],output+"_diff",self._rename(t2_path, 'diff_img')+".png")
#                 # self.save_diff_img(warp_img[Modality.c0].cpu().numpy(),ori_array['de'],output+"_diff",self._rename(c0_path, 'diff_warp_img')+".png")
#                 # self.save_diff_img(warp_img[Modality.t2].cpu().numpy(),ori_array['de'],output+"_diff",self._rename(t2_path, 'diff_warp_img')+".png")
#
#                 # self.save_img_with_tps(ori_array['c0'],theta_c0,output+"_tps", self._rename(c0_path, 'tps_img')+".png")
#                 # self.save_img_with_tps(ori_array['t2'],theta_t2,output+"_tps", self._rename(t2_path, 'tps_img')+".png")
#                 # self.save_img(ori_array['de'],output+"_tps",self._rename(de_path, 'tps_img')+".png")
#
#                 # self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'],output , self._rename(c0_path, 'assn_img'))
#                 # self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'],output, self._rename(c0_path, 'assn_lab'),is_label=True)
#
#                 # self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'],output, self._rename(t2_path, 'assn_img'))
#                 # self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'],output, self._rename(t2_path, 'assn_lab'),is_label=True)
#
#                 # self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'],output, self._rename(de_path, 'assn_img'))
#                 self.save_tensor_with_parameter(c0_pred, para['c0'],output, self._rename(c0_path, 'assn_lab'),is_label=True)
#                 self.save_tensor_with_parameter(t2_pred, para['t2'],output, self._rename(t2_path, 'assn_lab'),is_label=True)
#                 self.save_tensor_with_parameter(de_pred, para['de'],output, self._rename(de_path, 'assn_lab'),is_label=True)
#
#                 # self.save_torch_img_lab(self.args.gen_dir, self.op.convert_img_2_torch(ori_array['de']),de_seg, de_path, 'assn','pred')
#                 # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.c0],  warp_seg[Modality.c0], c0_path, 'assn','pred')
#                 # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.t2],  warp_seg[Modality.t2], t2_path, 'assn','pred')
#         ds={"c0":[],"t2":[],"de":[]}
#         hds={"c0":[],"t2":[],"de":[]}
#         for dir in range(26,46):
#             for modality in ['c0','t2','de']:
#                 try:
#                     gds=sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
#                     preds=sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")
#                     gds_list=[]
#                     preds_list=[]
#                     assert len(gds)==len(preds)
#                     if len(gds)==0:
#                         continue
#
#                     for gd,pred in zip(gds,preds):
#                         # print(f"{gd}-{pred}")
#                         gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)),axis=0))
#                         preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)),axis=0))
#                     gds_arr=np.concatenate(gds_list,axis=0)
#                     preds_arr=np.concatenate(preds_list,axis=0)
#                     gds_arr=np.squeeze(reindex_label_array_by_dict(gds_arr,{1:[1220,2221,200,500]}))
#                     preds_arr=np.squeeze(preds_arr)
#                     ds[modality].append(dc(gds_arr,preds_arr))
#                     # gd3d=sort_glob(f"{self.args.dataset_dir}/valid5_croped/*{subdir}_gd*nii.gz")
#                     # para=sitk.ReadImage(gd3d[0])
#                     # hds[modality].append(hd95(gds_arr,preds_arr,para.GetSpacing()))
#                     if len(gds_arr.shape)==2:
#                         gds_arr=np.expand_dims(gds_arr,axis=-1)
#                         preds_arr = np.expand_dims(preds_arr, axis=-1)
#
#                     hds[modality].append(hd95(gds_arr,preds_arr,(1,1,1)))
#                 except Exception as e:
#                     logging.error(e)
#
#         for k in ds.keys():
#             if (len(ds[k]))>0:
#                 # print(ds[k])
#                 write_array(self.args.res_excel, f'myops_asn_{k}_ds', ds[k])
#                 logging.info(f'subject level evaluation:  DS {k}: {np.mean(ds[k])}')
#                 logging.info(f'subject level evaluation:  DS {k}: {np.std(ds[k])}')
#                 # print(hds[k])
#                 write_array(self.args.res_excel, f'myops_asn_{k}_hd95', hds[k])
#                 logging.info(f'subject level evaluation:  HD {k}: {np.mean(hds[k])}')
#                 logging.info(f'subject level evaluation:  HD {k}: {np.std(hds[k])}')
#
#
#     def test(self):
#         print("staget test ASSN output")
#         mk_or_cleardir(self.args.gen_dir)
#         size=[self.args.image_size,self.args.image_size]
#         with torch.no_grad():
#             for img_c0, img_t2, img_de,c0_path,t2_path,de_path in self.test_loader:
#                 img= self.create_test_torch_tensor(img_c0,img_t2, img_de)
#                 warp_img = {}
#                 warp_seg={}
#                 para={}
#                 ori_array={}
#                 c0_pred,t2_pred,de_pred,theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
#                 para['c0'] = sitk.ReadImage(c0_path)
#                 ori_array['c0'] = sitk.GetArrayFromImage(para['c0'])
#
#                 para['t2'] = sitk.ReadImage(t2_path)
#                 ori_array['t2'] = sitk.GetArrayFromImage(para['t2'])
#
#                 para['de'] = sitk.ReadImage(de_path)
#                 ori_array['de'] = sitk.GetArrayFromImage(para['de'])
#
#                 warp_seg[Modality.c0] = self.model.warp(c0_pred, theta_c0)
#                 warp_seg[Modality.t2] = self.model.warp(t2_pred, theta_t2)
#
#
#                 de_pred = torch.argmax(de_pred, dim=1, keepdim=True)
#                 warp_seg[Modality.c0] = torch.argmax(warp_seg[Modality.c0], dim=1, keepdim=True)
#                 warp_seg[Modality.t2] = torch.argmax(warp_seg[Modality.t2], dim=1, keepdim=True)
#
#                 warp_img[Modality.c0] = self.model.warp(
#                     self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(),
#                     theta_c0)
#
#                 warp_img[Modality.t2] = self.model.warp(
#                     self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(),
#                     theta_t2)
#
#                 subdir = os.path.basename(os.path.dirname(c0_path[0]))
#                 output = os.path.join(self.args.gen_dir, subdir)
#                 mkdir_if_not_exist(output)
#                 self.save_diff_img(ori_array['c0'], ori_array['de'], output + "_diff",
#                                    self._rename(c0_path, 'diff_img') + ".png")
#                 self.save_diff_img(ori_array['t2'], ori_array['de'], output + "_diff",
#                                    self._rename(t2_path, 'diff_img') + ".png")
#                 self.save_diff_img(warp_img[Modality.c0].cpu().numpy(), ori_array['de'], output + "_diff",
#                                    self._rename(c0_path, 'diff_warp_img') + ".png")
#                 self.save_diff_img(warp_img[Modality.t2].cpu().numpy(), ori_array['de'], output + "_diff",
#                                    self._rename(t2_path, 'diff_warp_img') + ".png")
#
#                 self.save_img_with_tps(ori_array['c0'],theta_c0,output+"_tps", self._rename(c0_path, 'tps_img')+".png")
#                 self.save_img_with_tps(ori_array['t2'],theta_t2,output+"_tps", self._rename(t2_path, 'tps_img')+".png")
#                 self.save_img(ori_array['de'],output+"_tps",self._rename(de_path, 'tps_img')+".png")
#
#                 self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'], output, self._rename(c0_path, 'assn_img'))
#                 self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'], output, self._rename(c0_path, 'assn_lab'), is_label=True)
#
#                 self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'], output, self._rename(t2_path, 'assn_img'))
#                 self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'], output, self._rename(t2_path, 'assn_lab'),is_label=True)
#
#                 self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'], output, self._rename(de_path, 'assn_img'))
#                 self.save_tensor_with_parameter(de_pred, para['de'], output, self._rename(de_path, 'assn_lab'), is_label=True)
#
#     def create_torch_tensor(self, img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de):
#         img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
#         img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
#         img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
#         lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
#         lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
#         lab_de = lab_de.to(device=self.device, dtype=torch.float32)
#         #bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask,myo_scar_ede_mask,ori_lab
#         c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
#         t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
#         de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
#
#         c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=4, length=1)
#         t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=4, length=1)
#         de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=4, length=1)
#
#         lab_c0 = lab_c0.narrow(dim=1, start=-1, length=1)
#         lab_t2 = lab_t2.narrow(dim=1, start=-1, length=1)
#         lab_de = lab_de.narrow(dim=1, start=-1, length=1)
#
#         img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
#         lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
#         roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
#         roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
#         # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
#         return  img,lab,roi_lab1,roi_lab2
#
#     def create_test_torch_tensor(self, img_c0, img_t2, img_de):
#         img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
#         img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
#         img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
#         img = {Modality.c0: img_c0, Modality.t2: img_t2, Modality.de: img_de}
#         # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
#         return img
#
#     def save_img_with_tps(self,img,control_points,dir,name):
#         mkdir_if_not_exist(dir)
#         source_array = np.squeeze(img)
#         control_points=(control_points.data[0])
#         tmp=sitk.GetImageFromArray(source_array)
#         tmp=clipseScaleSitkImage(tmp,0,100)
#         source_array=sitk.GetArrayFromImage(tmp).astype('uint8')
#
#         save_image_with_tps_points(control_points, source_array, dir, name, self.args.grid_size, 500, 0)
#
#     def save_img(self,img,dir,name):
#         mkdir_if_not_exist(dir)
#         source_array = np.squeeze(img)
#         tmp = sitk.GetImageFromArray(source_array)
#         tmp = clipseScaleSitkImage(tmp, 0, 100)
#         source_array = sitk.GetArrayFromImage(tmp).astype('uint8')
#         save_img(source_array,dir,name,img_size=500,border=0)
#
#
#     def _rename(self,name,tag):
#         term = os.path.basename((name[0])).split("_")
#         name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{tag}_{term[4]}'
#         return name
#
#     def save_diff_img(self,array1,array2,dir,name):
#         mkdir_if_not_exist(dir)
#         array1=np.squeeze(array1).astype(np.float32)
#         array2=np.squeeze(array2).astype(np.float32)
#         array1=cv2.resize(array1,(500,500))
#         array2=cv2.resize(array2,(500,500))
#         diff = compare_images(rescale_intensity(array1),
#                               rescale_intensity(array2),
#                               method='checkerboard',n_tiles=(4,4))
#         diff=(diff * 255).astype(np.uint8)
#         cv2.imwrite(f"{dir}/{name}",diff )
#
#     def save_tensor_with_parameter(self, tensor, parameter, outputdir, name, is_label=False):
#         array=tensor.cpu().numpy()
#         array=np.squeeze(array)
#         target_size=parameter.GetSize()
#         if is_label==True:
#             array=self.op.resize(array,(target_size[1],target_size[0]))
#         else:
#             array=self.op.resize(array,(target_size[1],target_size[0]),0)
#
#         array=np.expand_dims(array,axis=0)
#         if is_label==True:
#             array=np.round(array).astype(np.int16)
#
#         img = sitk.GetImageFromArray(array)
#         img.CopyInformation(parameter)
#         sitk.WriteImage(img, os.path.join(outputdir, name+'.nii.gz'))