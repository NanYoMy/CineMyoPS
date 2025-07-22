from dataloader.util import SkimageOP_MSCMR
import logging
from os import path as osp
import cv2
import numpy as np
import torch
from medpy.metric import dc,hd95
from torch import optim
from torch.utils.data import DataLoader
import SimpleITK as sitk
from baseclass.medicalimage import Modality
from experiment.baseexperiment import BaseMSCMRExperiment
import os
from tools.dir import sort_glob, mk_or_cleardir,sort_time_glob,mkdir_if_not_exist
from tools.set_random_seed import worker_init_fn

from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftConsistentDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
from jrs_networks.jrs_losses import BinaryGaussianDice


class Experiment_ASN_MSCMR(BaseMSCMRExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.args=args
        from dataloader.mscmrdataset import MSCMRDataSet as DataSetLoader

        # if args.net == "tps":
        from jrs_networks.jrs_3m_tps_seg_net import JRS3MROITpsSegNet as RSNet
        # else:
        #     logging.error("unimplmented type")
        #     exit(-200)

        self.model = RSNet(args)

        if args.load:
            model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[args.ckpt]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.model = self.model.load(model_path, self.device)
            logging.info(f'Model loaded from {args.load} {model_path}')
        self.model.to(device= self.device)

        self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology',ret_path=False),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       drop_last=True,# 如果最后一个batch是为1，就出错。
                                       worker_init_fn=worker_init_fn)
        # self.trainall_loader = DataLoader(DataSetLoader(args, type="trainall",augo=True, task='pathology',ret_path=False),
        #                                batch_size=args.batch_size,
        #                                shuffle=True,
        #                                num_workers=4,
        #                                pin_memory=True,
        #                                worker_init_fn=worker_init_fn)

        self.val_loader = DataLoader(DataSetLoader(args, type="valid",augo=False,  task='pathology',ret_path=True),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)

        self.test_loader = DataLoader(DataSetLoader(args, type="test",augo=False,  task='pathology',ret_path=True),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)
        self.op = SkimageOP_MSCMR()
        logging.info(f'''Starting training:
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Optimizer:       {self.args.optimizer}
            Checkpoints:     {self.args.save_cp}
            Device:          {self.device.type}
            load:           {self.args.load}
            weight:          {self.args.weight}
            log:            {self.args.log_dir}
        ''')

    def validate_net(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        try:
            self.gen_valid()
        except Exception as e :
            print(e)

    # def train_eval_net(self):
    #     global_step = 0
    #     if self.args.optimizer == 'Adam':
    #         optimizer = optim.Adam(self.model.parameters(),
    #                                lr=self.args.lr)
    #     elif self.args.optimizer == 'RMSprop':
    #         optimizer = optim.RMSprop(self.model.parameters(),
    #                                   lr=self.args.lr,
    #                                   weight_decay=self.args.weight_decay)
    #     else:
    #         optimizer = optim.SGD(self.model.parameters(),
    #                               lr=self.args.lr,
    #                               momentum=self.args.momentum,
    #                               weight_decay=self.args.weight_decay,
    #                               nesterov=self.args.nesterov)
    #
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                milestones=self.args.lr_decay_milestones,
    #                                                gamma = self.args.lr_decay_gamma)
    #
    #
    #     # crit_reg=BinaryDiceLoss()
    #
    #     regcrit=BinaryGaussianDice()
    #     consiscrit=SoftConsistentDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
    #     # segcrit=DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    #     segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
    #     self.model.train() # dropout, BN在train和test的行为不一样
    #     mk_or_cleardir(self.args.log_dir)
    #     # loss=0
    #
    #     for epoch in range(self.args.init_epochs,self.args.epochs+1):
    #         sts_total_loss = 0
    #         np.random.seed(17 + epoch)
    #         #前1000个epoch只进行配准
    #
    #         print("train.................")
    #         for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
    #             #load data
    #             img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0,  img_t2, img_de,lab_c0, lab_t2, lab_de)
    #
    #             # train
    #             c0_pred,t2_pred,de_pred,theta_c0,theta_t2= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
    #             # c0_pred,de_pred,t2_pred,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.de], img[Modality.t2])
    #             #loss
    #             # warp_c0_pred = self.model.warp(c0_pred, theta_c0)
    #             # warp_t2_pred = self.model.warp(t2_pred, theta_t2)
    #             # loss_consis_c0 = regcrit(warp_c0_pred, de_pred)
    #             # loss_consis_t2 = regcrit(warp_t2_pred, de_pred)
    #
    #
    #             warp_c0_lab= self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
    #             warp_t2_lab= self.model.warp(roi_lab_myo[Modality.t2], theta_t2)
    #             loss_reg_c0 = regcrit(warp_c0_lab, roi_lab_myo[Modality.de])
    #             loss_reg_t2 = regcrit(warp_t2_lab, roi_lab_myo[Modality.de])
    #
    #
    #             loss_seg_c0 = segcrit(c0_pred, roi_lab_myo[Modality.c0])
    #             loss_seg_t2 = segcrit(t2_pred, roi_lab_myo[Modality.t2])
    #             loss_seg_de = segcrit(de_pred, roi_lab_myo[Modality.de])
    #
    #             # loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
    #             loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_c0+loss_reg_t2)
    #             # loss_all=loss_seg_t2+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
    #             #statistic
    #             sts_loss={}
    #             # sts_loss["train/reg_pre_c0"]=(loss_consis_c0.item())
    #             # sts_loss["train/reg_pre_t2"]=(loss_consis_t2.item())
    #             sts_loss["train/reg_c0"]=(loss_reg_c0.item())
    #             sts_loss["train/reg_t2"]=(loss_reg_t2.item())
    #
    #             sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
    #             sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
    #             sts_loss["train/loss_seg_de"]=(loss_seg_de.item())
    #
    #             sts_loss["train/loss_total"]=(loss_all.item())
    #
    #
    #             optimizer.zero_grad()
    #             loss_all.backward()
    #             torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
    #             optimizer.step()
    #             global_step += 1
    #
    #             self.write_dict_to_tb(sts_loss,global_step)
    #
    #         scheduler.step()
    #         if (epoch+1)%self.args.save_freq==0 or epoch==0:
    #             try:
    #                 print(sts_loss)
    #                 self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
    #                 self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
    #                 self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
    #
    #                 self.eval_writer.add_images("train/lv_labs/de", (roi_lab_lv[Modality.de]), global_step)
    #                 self.eval_writer.add_images("train/lv_labs/t2", (roi_lab_lv[Modality.t2]), global_step)
    #                 self.eval_writer.add_images("train/lv_labs/c0", (roi_lab_lv[Modality.c0]), global_step)
    #
    #                 self.eval_writer.add_images("train/myo_labs/de", (roi_lab_myo[Modality.de]), global_step)
    #                 self.eval_writer.add_images("train/myo_labs/t2", (roi_lab_myo[Modality.t2]), global_step)
    #                 self.eval_writer.add_images("train/myo_labs/c0", (roi_lab_myo[Modality.c0]), global_step)
    #
    #                 # evaluation befor save checkpoints
    #                 self.model.eval()
    #                 self.validate_net()
    #                 self.model.train()
    #                 ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
    #                 self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
    #                 logging.info(f'Checkpoint {epoch + 1} saved !')
    #             except OSError:
    #                 logging.info(f'Checkpoint {epoch + 1} save failed !')
    #                 exit(-999)
    #
    #
    #     self.eval_writer.close()
    #
    #

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

        regcrit=BinaryGaussianDice()
        consiscrit=SoftConsistentDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
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
                img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0,  img_t2, img_de,lab_c0, lab_t2, lab_de)

                # train
                c0_pred,t2_pred,de_pred,theta_c0,theta_t2= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                # c0_pred,de_pred,t2_pred,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.de], img[Modality.t2])
                #loss
                # warp_c0_pred = self.model.warp(c0_pred, theta_c0)
                # warp_t2_pred = self.model.warp(t2_pred, theta_t2)
                # loss_consis_c0 = regcrit(warp_c0_pred, de_pred)
                # loss_consis_t2 = regcrit(warp_t2_pred, de_pred)


                warp_c0_lab= self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
                warp_t2_lab= self.model.warp(roi_lab_myo[Modality.t2], theta_t2)
                loss_reg_c0 = regcrit(warp_c0_lab, roi_lab_myo[Modality.de])
                loss_reg_t2 = regcrit(warp_t2_lab, roi_lab_myo[Modality.de])


                loss_seg_c0 = segcrit(c0_pred, roi_lab_lv[Modality.c0])
                loss_seg_t2 = segcrit(t2_pred, roi_lab_lv[Modality.t2])
                loss_seg_de = segcrit(de_pred, roi_lab_lv[Modality.de])

                # loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
                loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_c0+loss_reg_t2)
                # loss_all=loss_seg_t2+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
                #statistic
                sts_loss={}
                # sts_loss["train/reg_pre_c0"]=(loss_consis_c0.item())
                # sts_loss["train/reg_pre_t2"]=(loss_consis_t2.item())
                sts_loss["train/reg_c0"]=(loss_reg_c0.item())
                sts_loss["train/reg_t2"]=(loss_reg_t2.item())

                sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
                sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
                sts_loss["train/loss_seg_de"]=(loss_seg_de.item())

                sts_loss["train/loss_total"]=(loss_all.item())


                optimizer.zero_grad()
                loss_all.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
                optimizer.step()
                global_step += 1

                self.write_dict_to_tb(sts_loss,global_step)

            scheduler.step()
            if (epoch+1)%self.args.save_freq==0 or epoch==0:
                try:
                    print(sts_loss)
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)

                    self.eval_writer.add_images("train/lv_labs/de", (roi_lab_lv[Modality.de]), global_step)
                    self.eval_writer.add_images("train/lv_labs/t2", (roi_lab_lv[Modality.t2]), global_step)
                    self.eval_writer.add_images("train/lv_labs/c0", (roi_lab_lv[Modality.c0]), global_step)

                    self.eval_writer.add_images("train/myo_labs/de", (roi_lab_myo[Modality.de]), global_step)
                    self.eval_writer.add_images("train/myo_labs/t2", (roi_lab_myo[Modality.t2]), global_step)
                    self.eval_writer.add_images("train/myo_labs/c0", (roi_lab_myo[Modality.c0]), global_step)

                    # evaluation befor save checkpoints
                    self.model.eval()
                    self.validate_net()
                    self.model.train()
                    ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                    self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                except OSError:
                    logging.info(f'Checkpoint {epoch + 1} save failed !')
                    exit(-999)


        self.eval_writer.close()

    def gen_valid(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        mk_or_cleardir(self.args.gen_dir)
        print("staget ASSN output")
        size=[self.args.image_size,self.args.image_size]
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
                img, lab, roi_lab_myo, roi_lab_lv = self.create_torch_tensor(img_c0,img_t2, img_de,  lab_c0, lab_t2, lab_de)
                warp_img = {}
                warp_seg={}
                para={}
                ori_array={}
                c0_pred,t2_pred,de_pred,theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                # c0_pred, de_pred, t2_pred, theta_c0, theta_t2 = self.model(img[Modality.c0], img[Modality.de], img[Modality.t2])
                para['c0']=sitk.ReadImage(c0_path)
                ori_array['c0']=sitk.GetArrayFromImage(para['c0'])

                para['t2'] = sitk.ReadImage(t2_path)
                ori_array['t2'] = sitk.GetArrayFromImage(para['t2'])

                para['de'] = sitk.ReadImage(de_path)
                ori_array['de'] = sitk.GetArrayFromImage(para['de'])



                warp_seg[Modality.c0]=self.model.warp(c0_pred, theta_c0)
                warp_seg[Modality.t2]=self.model.warp(t2_pred, theta_t2)

                # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
                # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
                de_pred=torch.argmax(de_pred,dim=1,keepdim=True)
                c0_pred=torch.argmax(c0_pred,dim=1,keepdim=True)
                t2_pred=torch.argmax(t2_pred,dim=1,keepdim=True)

                warp_seg[Modality.c0]=torch.argmax(warp_seg[Modality.c0],dim=1,keepdim=True)
                warp_seg[Modality.t2]=torch.argmax(warp_seg[Modality.t2],dim=1,keepdim=True)

                warp_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(), theta_c0)
                warp_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(), theta_t2)


                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                self.save_diff_img(ori_array['c0'], ori_array['de'], output +"_diff", self.renamepath(c0_path, 'diff_img') + ".png")
                self.save_diff_img(ori_array['t2'], ori_array['de'], output +"_diff", self.renamepath(t2_path, 'diff_img') + ".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(), ori_array['de'], output +"_diff", self.renamepath(c0_path, 'diff_warp_img') + ".png")
                self.save_diff_img(warp_img[Modality.t2].cpu().numpy(), ori_array['de'], output +"_diff", self.renamepath(t2_path, 'diff_warp_img') + ".png")


                self.save_img_with_tps(ori_array['c0'], theta_c0, output +"_tps", self.renamepath(c0_path, 'tps_img') + ".png")
                self.save_img_with_tps(ori_array['t2'], theta_t2, output +"_tps", self.renamepath(t2_path, 'tps_img') + ".png")
                self.save_img(ori_array['de'], output +"_tps", self.renamepath(de_path, 'tps_img') + ".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'], output, self.renamepath(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'], output, self.renamepath(de_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(c0_pred, para['c0'], output, self.renamepath(c0_path, 'branch_lab'), is_label=True)
                self.save_tensor_with_parameter(t2_pred, para['t2'], output, self.renamepath(t2_path, 'branch_lab'), is_label=True)
                self.save_tensor_with_parameter(de_pred, para['de'], output, self.renamepath(de_path, 'branch_lab'), is_label=True)

                # self.save_torch_img_lab(self.args.gen_dir, self.op.convert_img_2_torch(ori_array['de']),de_seg, de_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.c0],  warp_seg[Modality.c0], c0_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.t2],  warp_seg[Modality.t2], t2_path, 'assn','pred')
        seg_ds={"c0":[],"t2":[],"de":[]}
        seg_hds={"c0":[],"t2":[],"de":[]}
        reg_ds={"c0":[],"t2":[],"de":[]}
        reg_hds={"c0":[],"t2":[],"de":[]}

        for dir in range(26,46):
            for modality in ['c0','t2','de']:
                seg_gds=sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
                seg_preds=sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_branch_lab*nii.gz")

                reg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*de_*nii.gz")
                reg_preds=sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")

                if len(seg_gds)==0:
                    continue

                ds_res, hd_res = self.cal_ds_hd(seg_gds, seg_preds)
                seg_ds[modality].append(ds_res)
                seg_hds[modality].append(hd_res)

                ds_res, hd_res = self.cal_ds_hd(reg_gds, reg_preds)
                reg_ds[modality].append(ds_res)
                reg_hds[modality].append(hd_res)

        print("=========segmentation=================")
        self.print_res(seg_ds, seg_hds)
        print("=========registration=================")
        self.print_res(reg_ds, reg_hds)


    def test(self):
        print("staget test ASSN output")
        mk_or_cleardir(self.args.gen_dir)
        size=[self.args.image_size,self.args.image_size]
        with torch.no_grad():
            for img_c0, img_t2, img_de,c0_path,t2_path,de_path in self.test_loader:
                img= self.create_test_torch_tensor(img_c0,img_t2, img_de)
                warp_img = {}
                warp_seg={}
                para={}
                ori_array={}
                c0_pred,t2_pred,de_pred,theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                para['c0'] = sitk.ReadImage(c0_path)
                ori_array['c0'] = sitk.GetArrayFromImage(para['c0'])

                para['t2'] = sitk.ReadImage(t2_path)
                ori_array['t2'] = sitk.GetArrayFromImage(para['t2'])

                para['de'] = sitk.ReadImage(de_path)
                ori_array['de'] = sitk.GetArrayFromImage(para['de'])

                warp_seg[Modality.c0] = self.model.warp(c0_pred, theta_c0)
                warp_seg[Modality.t2] = self.model.warp(t2_pred, theta_t2)


                de_pred = torch.argmax(de_pred, dim=1, keepdim=True)
                warp_seg[Modality.c0] = torch.argmax(warp_seg[Modality.c0], dim=1, keepdim=True)
                warp_seg[Modality.t2] = torch.argmax(warp_seg[Modality.t2], dim=1, keepdim=True)

                warp_img[Modality.c0] = self.model.warp(
                    self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(),
                    theta_c0)

                warp_img[Modality.t2] = self.model.warp(
                    self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(),
                    theta_t2)

                subdir = os.path.basename(os.path.dirname(c0_path[0]))
                output = os.path.join(self.args.gen_dir, subdir)
                mkdir_if_not_exist(output)
                self.save_diff_img(ori_array['c0'], ori_array['de'], output + "_diff",
                                   self.renamepath(c0_path, 'diff_img') + ".png")
                self.save_diff_img(ori_array['t2'], ori_array['de'], output + "_diff",
                                   self.renamepath(t2_path, 'diff_img') + ".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(), ori_array['de'], output + "_diff",
                                   self.renamepath(c0_path, 'diff_warp_img') + ".png")
                self.save_diff_img(warp_img[Modality.t2].cpu().numpy(), ori_array['de'], output + "_diff",
                                   self.renamepath(t2_path, 'diff_warp_img') + ".png")

                self.save_img_with_tps(ori_array['c0'], theta_c0, output +"_tps", self.renamepath(c0_path, 'tps_img') + ".png")
                self.save_img_with_tps(ori_array['t2'], theta_t2, output +"_tps", self.renamepath(t2_path, 'tps_img') + ".png")
                self.save_img(ori_array['de'], output +"_tps", self.renamepath(de_path, 'tps_img') + ".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'], output, self.renamepath(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'], output, self.renamepath(de_path, 'assn_lab'), is_label=True)


class Experiment_ASN_MSCMR_Myo(Experiment_ASN_MSCMR):
    def __init__(self, args):
        super().__init__(args)


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

        regcrit=BinaryGaussianDice()
        consiscrit=SoftConsistentDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
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
                img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0,  img_t2, img_de,lab_c0, lab_t2, lab_de)

                # train
                c0_pred,t2_pred,de_pred,theta_c0,theta_t2= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                # c0_pred,de_pred,t2_pred,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.de], img[Modality.t2])
                #loss
                # warp_c0_pred = self.model.warp(c0_pred, theta_c0)
                # warp_t2_pred = self.model.warp(t2_pred, theta_t2)
                # loss_consis_c0 = regcrit(warp_c0_pred, de_pred)
                # loss_consis_t2 = regcrit(warp_t2_pred, de_pred)
                warp_img={}
                warp_roi_lab_myo={}
                warp_roi_lab_lv={}

                warp_roi_lab_myo[Modality.c0]= self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
                warp_roi_lab_myo[Modality.t2]= self.model.warp(roi_lab_myo[Modality.t2], theta_t2)

                warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta_c0)
                warp_roi_lab_lv[Modality.t2] = self.model.warp(roi_lab_lv[Modality.t2], theta_t2)

                loss_reg_c0 = regcrit(warp_roi_lab_myo[Modality.c0], roi_lab_myo[Modality.de])#+regcrit(warp_roi_lab_lv[Modality.c0], roi_lab_lv[Modality.de])
                loss_reg_t2 = regcrit(warp_roi_lab_myo[Modality.t2], roi_lab_myo[Modality.de])#+regcrit(warp_roi_lab_lv[Modality.t2], roi_lab_lv[Modality.de])


                loss_seg_c0 = segcrit(c0_pred, roi_lab_myo[Modality.c0])
                loss_seg_t2 = segcrit(t2_pred, roi_lab_myo[Modality.t2])
                loss_seg_de = segcrit(de_pred, roi_lab_myo[Modality.de])

                # loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
                loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_c0+loss_reg_t2)
                # loss_all=(loss_seg_de+loss_seg_c0)+self.args.weight*(loss_reg_c0)
                # loss_all=loss_seg_t2+self.args.weight*(loss_consis_c0+loss_consis_t2+loss_reg_c0+loss_reg_t2)
                #statistic
                sts_loss={}
                # sts_loss["train/reg_pre_c0"]=(loss_consis_c0.item())
                # sts_loss["train/reg_pre_t2"]=(loss_consis_t2.item())
                sts_loss["train/reg_c0"]=(loss_reg_c0.item())
                sts_loss["train/reg_t2"]=(loss_reg_t2.item())

                sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
                sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
                sts_loss["train/loss_seg_de"]=(loss_seg_de.item())

                sts_loss["train/loss_total"]=(loss_all.item())


                optimizer.zero_grad()
                loss_all.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
                optimizer.step()
                global_step += 1

                self.write_dict_to_tb(sts_loss,global_step)

            scheduler.step()
            if (epoch+1)%self.args.save_freq==0 or epoch==0:
                try:
                    print(sts_loss)
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)

                    # self.eval_writer.add_images("train/lv_labs/de", (roi_lab_lv[Modality.de]), global_step)
                    # self.eval_writer.add_images("train/lv_labs/t2", (roi_lab_lv[Modality.t2]), global_step)
                    # self.eval_writer.add_images("train/lv_labs/c0", (roi_lab_lv[Modality.c0]), global_step)

                    self.eval_writer.add_images("train/myo_labs/de", (roi_lab_myo[Modality.de]), global_step)
                    self.eval_writer.add_images("train/myo_labs/t2", (roi_lab_myo[Modality.t2]), global_step)
                    self.eval_writer.add_images("train/myo_labs/c0", (roi_lab_myo[Modality.c0]), global_step)

                    # evaluation befor save checkpoints
                    self.model.eval()
                    self.validate_net()
                    self.model.train()
                    ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                    self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                except OSError:
                    logging.info(f'Checkpoint {epoch + 1} save failed !')
                    exit(-999)


        self.eval_writer.close()

    def gen_valid(self):
        """
        Evaluation without the densecrf with the dice coefficient
        """
        mk_or_cleardir(self.args.gen_dir)
        print("staget ASSN output")
        size=[self.args.image_size,self.args.image_size]
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
                img, lab, roi_lab_myo, roi_lab_lv = self.create_torch_tensor(img_c0,img_t2, img_de,  lab_c0, lab_t2, lab_de)
                warp_img = {}
                warp_seg={}
                para={}
                ori_array={}
                c0_pred,t2_pred,de_pred,theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                # c0_pred, de_pred, t2_pred, theta_c0, theta_t2 = self.model(img[Modality.c0], img[Modality.de], img[Modality.t2])
                para['c0']=sitk.ReadImage(c0_path)
                ori_array['c0']=sitk.GetArrayFromImage(para['c0'])

                para['t2'] = sitk.ReadImage(t2_path)
                ori_array['t2'] = sitk.GetArrayFromImage(para['t2'])

                para['de'] = sitk.ReadImage(de_path)
                ori_array['de'] = sitk.GetArrayFromImage(para['de'])



                warp_seg[Modality.c0]=self.model.warp(c0_pred, theta_c0)
                warp_seg[Modality.t2]=self.model.warp(t2_pred, theta_t2)

                # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
                # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
                de_pred=torch.argmax(de_pred,dim=1,keepdim=True)
                c0_pred=torch.argmax(c0_pred,dim=1,keepdim=True)
                t2_pred=torch.argmax(t2_pred,dim=1,keepdim=True)

                warp_seg[Modality.c0]=torch.argmax(warp_seg[Modality.c0],dim=1,keepdim=True)
                warp_seg[Modality.t2]=torch.argmax(warp_seg[Modality.t2],dim=1,keepdim=True)

                warp_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(), theta_c0)
                warp_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(), theta_t2)


                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                self.save_diff_img(ori_array['c0'], ori_array['de'], output +"_diff", self.renamepath(c0_path, 'diff_img') + ".png")
                self.save_diff_img(ori_array['t2'], ori_array['de'], output +"_diff", self.renamepath(t2_path, 'diff_img') + ".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(), ori_array['de'], output +"_diff", self.renamepath(c0_path, 'diff_warp_img') + ".png")
                self.save_diff_img(warp_img[Modality.t2].cpu().numpy(), ori_array['de'], output +"_diff", self.renamepath(t2_path, 'diff_warp_img') + ".png")


                self.save_img_with_tps(ori_array['c0'], theta_c0, output +"_tps", self.renamepath(c0_path, 'tps_img') + ".png")
                self.save_img_with_tps(ori_array['t2'], theta_t2, output +"_tps", self.renamepath(t2_path, 'tps_img') + ".png")
                self.save_img(ori_array['de'], output +"_tps", self.renamepath(de_path, 'tps_img') + ".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'], output, self.renamepath(c0_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'], output, self.renamepath(t2_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'], output, self.renamepath(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'], output, self.renamepath(de_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(c0_pred, para['c0'], output, self.renamepath(c0_path, 'branch_lab'), is_label=True)
                self.save_tensor_with_parameter(t2_pred, para['t2'], output, self.renamepath(t2_path, 'branch_lab'), is_label=True)
                self.save_tensor_with_parameter(de_pred, para['de'], output, self.renamepath(de_path, 'branch_lab'), is_label=True)

                # self.save_torch_img_lab(self.args.gen_dir, self.op.convert_img_2_torch(ori_array['de']),de_seg, de_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.c0],  warp_seg[Modality.c0], c0_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.t2],  warp_seg[Modality.t2], t2_path, 'assn','pred')
        seg_ds={"c0":[],"t2":[],"de":[]}
        seg_hds={"c0":[],"t2":[],"de":[]}
        reg_ds={"c0":[],"t2":[],"de":[]}
        reg_hds={"c0":[],"t2":[],"de":[]}

        for dir in range(26,46):
            for modality in ['c0','t2','de']:
                # print(f"=>>>>>{dir}")
                seg_gds=sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
                seg_preds=sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_branch_lab*nii.gz")

                reg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*de_*nii.gz")
                reg_preds=sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")

                if len(seg_gds)==0 or len(seg_preds)==0 or len(reg_gds)==0 or len(reg_preds)==0:
                    continue

                try:
                    ds_res, hd_res = self.cal_ds_hd(seg_gds, seg_preds,{1:[1220,2221,200]})
                    seg_ds[modality].append(ds_res)
                    seg_hds[modality].append(hd_res)

                    ds_res, hd_res = self.cal_ds_hd(reg_gds, reg_preds,{1:[1220,2221,200]})
                    reg_ds[modality].append(ds_res)
                    reg_hds[modality].append(hd_res)
                except Exception as e:
                    logging.error(e)

        print("=========segmentation=================")
        self.print_res(seg_ds, seg_hds,'seg')
        print("=========registration=================")
        self.print_res(reg_ds, reg_hds,'reg')

