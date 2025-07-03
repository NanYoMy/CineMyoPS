import logging
from os import path as osp

import SimpleITK as sitk
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from baseclass.medicalimage import Modality
# from dataloader.jrsdataset import MyoPSDataSet_unliagnedV2 as DataSetLoader
from dataloader.jrsdataset import MyoPSDataSet_unliagnedV3 as DataSetLoader
from dataloader.util import SkimageOP_MSCMR
from experiment.baseexperiment import BaseMSCMRExperiment
# from jrs_networks.jrs_tps_seg_net import JRS3TpsSegNet as RSNet
from jrs_networks.jrs_3m_tps_seg_net import JRS3MROITpsSegNet as RSNet
from tools.dir import sort_time_glob, sort_glob, mk_or_cleardir
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.set_random_seed import worker_init_fn
import itertools
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss,SoftDiceLoss,SoftConsistentDiceLoss
'''

'''
# from tools.model_static import static_model
# class Experiment_RJASN(BaseMSCMRExperiment):
#     def __init__(self,args):
#         super().__init__(args)
#         self.args=args
#
#         self.model = RSNet(args)
#         self.op = SkimageOP_MSCMR()
#         if args.load:
#             model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[-1]
#             # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
#             logging.info(f'Model loaded from : {args.load} {model_path}')
#             self.model = self.model.load(model_path, self.device)
#         self.model.to(device= self.device)
#         static_model(self.model)
#
#         self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology'),
#                                        batch_size=args.batch_size,
#                                        shuffle=True,
#                                        num_workers=4,
#                                        pin_memory=True,
#                                        worker_init_fn=worker_init_fn)
#         self.val_loader = DataLoader(DataSetLoader(args, type="test",augo=False,  task='pathology'),
#                                      batch_size=1,
#                                      shuffle=False,
#                                      num_workers=4,
#                                      pin_memory=True,
#                                      worker_init_fn=worker_init_fn)
#
#         self.val_loader = DataLoader(DataSetLoader(args, type="all",augo=False,  task='pathology'),
#                                      batch_size=1,
#                                      shuffle=False,
#                                      num_workers=4,
#                                      pin_memory=True,
#                                      worker_init_fn=worker_init_fn)
#
#         r1 = self.args.span_range_height
#         r2 = self.args.span_range_width
#         control_points = np.array(list(itertools.product(
#             np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.args.grid_height - 1)),
#             np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.args.grid_width - 1)),
#         )))
#         self.base_control_points=np.zeros_like(control_points)
#         self.base_control_points[:,0]=control_points[:,1]
#         self.base_control_points[:,1]=control_points[:,0]
#
#         logging.info(f'''Starting training:
#             Epochs:          {self.args.epochs}
#             Batch size:      {self.args.batch_size}
#             Learning rate:   {self.args.lr}
#             Optimizer:       {self.args.optimizer}
#             Checkpoints:     {self.args.save_cp}
#             Device:          {self.device.type}
#             load:           {self.args.load}
#         ''')
#
#
#     def validate_net(self):
#         """
#         Evaluation without the densecrf with the dice coefficient
#         """
#         self.val_loader = DataLoader(DataSetLoader(self.args, type="all",augo=False,  task='pathology'),
#                                      batch_size=1,
#                                      shuffle=False,
#                                      num_workers=4,
#                                      pin_memory=True,
#                                      worker_init_fn=worker_init_fn)
#
#
#         size=(256,256)
#         import os
#         from tools.dir import mkdir_if_not_exist
#         with torch.no_grad():
#             for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
#                 #
#                 # if  c0_path[0].find('33')<0:
#                 #     continue
#
#
#                 c0_lab_path=c0_path[0].replace("img_c0","ana_c0")
#                 t2_lab_path=t2_path[0].replace('img_t2','ana_patho_t2_edema')
#                 de_lab_path=de_path[0].replace('img_de','ana_patho_de_scar')
#
#                 img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv = self.create_torch_tensor(img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de)
#                 warp_pred = {}
#                 pred={}
#                 para={}
#                 ori_img={}
#                 ori_lab={}
#                 ums_img={}
#                 warp_ori_img = {}
#                 warp_ori_lab={}
#                 # train
#                 c0_seg,t2_seg,lge_seg,theta_c0,theta_t2 = self.model(img[Modality.c0],img[Modality.t2], img[Modality.de])
#
#                 para['c0'] = sitk.ReadImage(c0_path)
#                 ori_img['c0'] = sitk.GetArrayFromImage(para['c0'])
#                 # ori_img['c0'] = self.op.usm(ori_img['c0'])
#                 ori_lab['c0']=sitk.GetArrayFromImage(sitk.ReadImage(c0_lab_path))
#
#                 para['t2'] = sitk.ReadImage(t2_path)
#                 ori_img['t2'] = sitk.GetArrayFromImage(para['t2'])
#                 # ori_img['t2'] = self.op.usm(ori_img['t2'])
#                 ori_lab['t2']=sitk.GetArrayFromImage(sitk.ReadImage(t2_lab_path))
#
#                 para['de'] = sitk.ReadImage(de_path)
#                 ori_img['de'] = sitk.GetArrayFromImage(para['de'])
#                 # ori_img['de'] = self.op.usm(ori_img['de'])
#                 ori_lab['de']=sitk.GetArrayFromImage(sitk.ReadImage(de_lab_path))
#
#
#                 warp_pred[Modality.c0]=self.model.warp(c0_seg, theta_c0)
#                 warp_pred[Modality.t2]=self.model.warp(t2_seg, theta_t2)
#                 # print(de_path)
#                 # print(theta_c0.cpu().numpy()[0,:,:]-self.base_control_points)
#                 # print(theta_t2.cpu().numpy()[0,:,:]-self.base_control_points)
#
#                 # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
#                 # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
#                 pred['de']=torch.argmax(lge_seg,dim=1,keepdim=True)
#                 pred['c0']=torch.argmax(c0_seg,dim=1,keepdim=True)
#                 pred['t2']=torch.argmax(t2_seg,dim=1,keepdim=True)
#
#                 warp_pred[Modality.c0]=torch.argmax(warp_pred[Modality.c0],dim=1,keepdim=True)
#                 warp_pred[Modality.t2]=torch.argmax(warp_pred[Modality.t2],dim=1,keepdim=True)
#
#
#                 warp_ori_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_img['c0'], size), axis=0)).cuda(), theta_c0)
#                 warp_ori_lab[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_lab['c0'], size,order=0), axis=0)).cuda(), theta_c0,mode='nearest')
#
#                 warp_ori_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_img['t2'], size), axis=0)).cuda(), theta_t2)
#                 warp_ori_lab[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_lab['t2'], size,order=0), axis=0)).cuda(), theta_t2,mode='nearest')
#
#
#                 subdir=os.path.basename(os.path.dirname(c0_path[0]))
#                 output=os.path.join(self.args.gen_dir,subdir)
#                 mkdir_if_not_exist(output)
#
#                 if self.args.save_imgs==True:
#                     self.save_diff_img(ori_img['c0'],ori_img['de'],output+"_diff",self.renamepath(c0_path, 'diff_img')+".png")
#                     self.save_diff_img(ori_img['t2'],ori_img['de'],output+"_diff",self.renamepath(t2_path, 'diff_img')+".png")
#                     self.save_diff_img(warp_ori_img[Modality.c0].cpu().numpy(),ori_img['de'],output+"_diff",self.renamepath(c0_path, 'diff_warp_img')+".png")
#                     self.save_diff_img(warp_ori_img[Modality.t2].cpu().numpy(),ori_img['de'],output+"_diff",self.renamepath(t2_path, 'diff_warp_img')+".png")
#
#                     self.save_img_with_mv_fix_contorusV2(ori_img['c0'],
#                                                        reindex_label_array_by_dict(ori_lab['c0'],{2:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(ori_lab['de'], {1: [1, 200, 1220,
#                                                                                                        2221, 500]}),
#                                                        output +"_contours", self.renamepath(c0_path, 'gt_con_img') +".png")
#                     self.save_img_with_mv_fix_contorusV2(ori_img['t2'],
#                                                 reindex_label_array_by_dict(ori_lab['t2'],{2:[1,200,1220,2221,500]}),
#                                                 reindex_label_array_by_dict(ori_lab['de'],{1: [1, 200, 1220, 2221, 500]}),
#                                                 output+"_contours",self.renamepath(t2_path, 'gt_con_img')+".png")
#                     self.save_img_with_contorusV2(ori_img['de'],
#                                                 reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
#                                                 output + "_contours", self.renamepath(de_path, 'gt_con_img') + ".png")
#
#                     self.save_img_with_mv_fix_contorusV2(warp_ori_img[Modality.c0].cpu().numpy(),
#                                                        reindex_label_array_by_dict(warp_ori_lab[Modality.c0].cpu().numpy(),{2:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
#                                                        output+"_reg_contours",
#                                                        self.renamepath(c0_path, 'warp_img_fix_mv')+".png")
#
#                     self.save_img_with_mv_fix_contorusV2(warp_ori_img[Modality.t2].cpu().numpy(),
#                                                        reindex_label_array_by_dict(warp_ori_lab[Modality.t2].cpu().numpy(),{2:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
#                                                        output+"_reg_contours",
#                                                        self.renamepath(t2_path, 'warp_img_fix_mv')+".png")
#
#                     self.save_image_with_pred_gt_contousV2(warp_ori_img[Modality.c0].cpu().numpy(),
#                                                        reindex_label_array_by_dict(warp_pred[Modality.c0].cpu().numpy(),{3:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(warp_ori_lab[Modality.c0].cpu().numpy(),{2:[1,200,1220,2221,500]}),
#                                                        output+"_seg_contours",
#                                                        self.renamepath(c0_path, 'warp_img_pre')+".png")
#
#                     self.save_image_with_pred_gt_contousV2(warp_ori_img[Modality.t2].cpu().numpy(),
#                                                        reindex_label_array_by_dict(warp_pred[Modality.t2].cpu().numpy(),{3:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(warp_ori_lab[Modality.t2].cpu().numpy(),{2:[1,200,1220,2221,500]}),
#                                                        output+"_seg_contours",
#                                                        self.renamepath(t2_path, 'warp_img_pre')+".png")
#
#                     self.save_image_with_pred_gt_contousV2(ori_img['de'],
#                                                        reindex_label_array_by_dict(pred['de'].cpu().numpy(),{3:[1,200,1220,2221,500]}),
#                                                        reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221,500]}),
#                                                        output+"_seg_contours",
#                                                        self.renamepath(de_path, 'warp_img_pre')+".png")
#
#                     self.save_img_with_tps(ori_img['c0'],theta_c0,output+"_tps", self.renamepath(c0_path, 'tps_img')+".png")
#                     self.save_img_with_tps(ori_img['t2'],theta_t2,output+"_tps", self.renamepath(t2_path, 'tps_img')+".png")
#                     self.save_img(ori_img['de'],output+"_tps",self.renamepath(de_path, 'tps_img')+".png")
#
#                 self.save_tensor_with_parameter(warp_ori_img[Modality.c0], para['c0'],output , self.renamepath(c0_path, 'assn_img'))
#                 self.save_tensor_with_parameter(warp_pred[Modality.c0], para['c0'],output, self.renamepath(c0_path, 'assn_lab'),is_label=True)
#                 self.save_tensor_with_parameter(warp_ori_lab[Modality.c0], para['c0'],output, self.renamepath(c0_path, 'assn_gt_lab'),is_label=True)
#
#                 self.save_tensor_with_parameter(warp_ori_img[Modality.t2], para['t2'],output , self.renamepath(t2_path, 'assn_img'))
#                 self.save_tensor_with_parameter(warp_pred[Modality.t2], para['t2'],output, self.renamepath(t2_path, 'assn_lab'),is_label=True)
#                 self.save_tensor_with_parameter(warp_ori_lab[Modality.t2], para['t2'],output, self.renamepath(t2_path, 'assn_gt_lab'),is_label=True)
#
#
#                 self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_img['de']), para['de'],output, self.renamepath(de_path, 'assn_img'))
#                 self.save_tensor_with_parameter(pred['de'], para['de'],output, self.renamepath(de_path, 'assn_lab'),is_label=True)
#                 self.save_tensor_with_parameter(ori_lab['de'], para['de'],output, self.renamepath(de_path, 'assn_gt_lab'),is_label=True)
#
#                 self.save_tensor_with_parameter(pred['c0'], para['c0'],output, self.renamepath(c0_path, 'branch_lab'),is_label=True)
#                 self.save_tensor_with_parameter(pred['t2'], para['t2'],output, self.renamepath(t2_path, 'branch_lab'),is_label=True)
#                 self.save_tensor_with_parameter(pred['de'], para['de'],output, self.renamepath(de_path, 'branch_lab'),is_label=True)
#
#
#         # logging.info(f'slice level evaluation reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
#         # logging.info(f'slice level evaluation seg mv:{np.mean(seg_error["mv"])} | fix: {np.mean(seg_error["fix"])}')
#
#         seg_ds = {"c0": [], "t2": [], "de": []}
#         seg_hds = {"c0": [], "t2": [], "de": []}
#         reg_ds = {"c0": [], "t2": [], "de": []}
#         reg_hds = {"c0": [], "t2": [], "de": []}
#
#         for dir in range(26, 46):
#             for modality in ['c0','t2', 'de']:
#                 seg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
#                 seg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_branch_lab*nii.gz")
#
#                 reg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*de_*nii.gz")
#                 reg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")
#
#                 print(f"================>{dir}")
#                 if len(seg_gds) == 0:
#                     continue
#                 try:
#                     ds_res, hd_res = self.cal_ds_hd(seg_gds, seg_preds, {1: [1220, 2221, 200,500]})
#                     seg_ds[modality].append(ds_res)
#                     seg_hds[modality].append(hd_res)
#
#                     ds_res, hd_res = self.cal_ds_hd(reg_gds, reg_preds, {1: [1220, 2221, 200,500]})
#                     reg_ds[modality].append(ds_res)
#                     reg_hds[modality].append(hd_res)
#                 except Exception as e:
#                     print(e)
#                     logging.error(e)
#
#         print("=========segmentation=================")
#         self.print_res(seg_ds, seg_hds, 'seg')
#         print("=========registration=================")
#         self.print_res(reg_ds, reg_hds, 'reg')
#
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
#         from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
#         from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
#         from nnunet.utilities.nd_softmax import softmax_helper
#         # crit_reg=BinaryGaussianDice()
#         #regcrit=SoftConsistentDiceLoss
#         regcrit=BinaryGaussianDice(sigma=1)
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
#                 img, lab, roi_lab_myo, roi_lab_lv,roi_lab_rv= self.create_torch_tensor(img_c0, img_t2,img_de, lab_c0, lab_t2, lab_de)
#                 warp_img={}
#                 warp_roi_lab_myo={}
#                 sts_loss = {}
#                 warp_roi_lab_lv={}
#                 warp_roi_lab_rv={}
#                 # train
#                 c0_seg,t2_seg,lge_seg,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.t2], img[Modality.de])
#
#                 #loss
#                 warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
#                 warp_roi_lab_myo[Modality.t2] = self.model.warp(roi_lab_myo[Modality.t2], theta_t2)
#
#                 warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta_c0)
#                 warp_roi_lab_lv[Modality.t2] = self.model.warp(roi_lab_lv[Modality.t2], theta_t2)
#
#                 warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta_c0)
#                 warp_roi_lab_rv[Modality.t2] = self.model.warp(roi_lab_rv[Modality.t2], theta_t2)
#
#
#                 warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta_c0)
#                 warp_img[Modality.t2] = self.model.warp(img[Modality.t2], theta_t2)
#
#                 loss_reg_myo = regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.c0])+regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.t2])
#                 loss_reg_lv = regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.c0])+regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.t2])
#                 loss_reg_rv = regcrit(roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.c0])+regcrit(roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.t2])
#                 #分割目标是myo
#                 # loss_seg_lge = segcrit(lge_seg, roi_lab_myo[Modality.de])
#                 # loss_seg_c0 = segcrit(c0_seg, roi_lab_myo[Modality.c0])
#                 # loss_seg_t2 = segcrit(t2_seg, roi_lab_myo[Modality.t2])
#
#                 loss_seg_lge = segcrit(lge_seg, roi_lab_lv[Modality.de])
#                 loss_seg_c0 = segcrit(c0_seg, roi_lab_lv[Modality.c0])
#                 loss_seg_t2 = segcrit(t2_seg, roi_lab_lv[Modality.t2])
#
#                 # loss_all=(loss_seg_lge+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_myo+loss_reg_lv+loss_reg_rv)
#                 loss_all=(2*loss_seg_lge+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_myo+loss_reg_lv)
#                 #statistic
#
#                 sts_loss["train/reg_myo"]=(loss_reg_myo.item())
#                 sts_loss["train/reg_lv"]=(loss_reg_lv.item())
#                 sts_loss["train/reg_rv"] = (loss_reg_rv.item())
#                 sts_loss["train/loss_seg_lge"]=(loss_seg_lge.item())
#                 sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
#                 sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
#                 sts_loss["train/reg_total"]=(loss_all.item())
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
#             if (epoch)%self.args.save_freq==0:
#                 try:
#                     self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
#                     self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
#                     self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
#                     # self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
#                     # self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
#                     # self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)
#                     self.eval_writer.add_images("train/labs/lv_de", roi_lab_lv[Modality.de], global_step)
#                     self.eval_writer.add_images("train/labs/lv_t2", roi_lab_lv[Modality.t2], global_step)
#                     self.eval_writer.add_images("train/labs/lv_c0", roi_lab_lv[Modality.c0], global_step)
#                     self.eval_writer.add_images("train/labs/myo_de", roi_lab_myo[Modality.de], global_step)
#                     self.eval_writer.add_images("train/labs/myo_t2", roi_lab_myo[Modality.t2], global_step)
#                     self.eval_writer.add_images("train/labs/myo_c0", roi_lab_myo[Modality.c0], global_step)
#
#                     # evaluation befor save checkpoints
#                     self.model.eval()
#                     self.validate_net()
#                     self.model.train()
#                     ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
#                     self.model.save(osp.join(self.args.checkpoint_dir, ckpt_name))
#                     logging.info(f'Checkpoint {epoch + 1} saved !')
#                 except Exception as e:
#                     print(e)
#                     logging.error(f'Checkpoint {epoch + 1} {e} ')
#
#
#         self.eval_writer.close()
#
#     def gen(self):
#         print("generating................")
#         #
#         # # reg_error = {"init": [], "reg": []}
#         # # seg_error = {"mv": [], "fix": []}
#         # # with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
#         # i = 0
#         # with torch.no_grad():
#         #     for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path, t2_path, de_path in self.val_loader:
#         #         img, lab, roi_lab_myo, roi_lab_rv = self.create_torch_tensor(img_c0,  img_t2,img_de, lab_c0,lab_t2, lab_de)
#         #         warp_img = {}
#         #         warp_lab = {}
#         #         warp_roi_lab_myo = {}
#         #         warp_roi_lab_rv = {}
#         #         # train
#         #         c0_seg, t2_seg,lge_seg, theta_c0,theta_t2 = self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
#         #         c0_seg = torch.argmax(c0_seg, dim=1, keepdim=True)
#         #         t2_seg = torch.argmax(t2_seg, dim=1, keepdim=True)
#         #         lge_seg = torch.argmax(lge_seg, dim=1, keepdim=True)
#         #
#         #         warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta_c0)
#         #         warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], theta_c0, mode='nearest')
#         #
#         #         warp_img[Modality.t2] = self.model.warp(img[Modality.t2], theta_t2)
#         #         warp_lab[Modality.t2] = self.model.warp(lab[Modality.t2], theta_c0, mode='nearest')
#         #
#         #
#         #         warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta_c0, mode='nearest')
#         #         warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta_c0, mode='nearest')
#         #
#         #         # bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask
#         #         #warped c0
#         #         self.save_torch_to_nii(self.args.gen_dir,warp_img[Modality.c0],c0_path, 'c0_ori',is_lab=False)
#         #         self.save_torch_to_nii(self.args.gen_dir,warp_roi_lab_myo[Modality.c0],c0_path, 'c0_ori',is_lab=True)
#         #         self.save_torch_to_nii(self.args.gen_dir, c0_seg, c0_path, "c0_pred",is_lab=True)
#         #         #de
#         #         self.save_torch_to_nii(self.args.gen_dir,img[Modality.de],de_path, 'de_ori',is_lab=False)
#         #
#         #         myo_scar_mask = lab_de.narrow(dim=1, start=6, length=1)
#         #         self.save_torch_to_nii(self.args.gen_dir,myo_scar_mask,de_path, 'de_ori',is_lab=True)
#         #         self.save_torch_to_nii(self.args.gen_dir, lge_seg, c0_path, "de_pred",is_lab=True)
#         #