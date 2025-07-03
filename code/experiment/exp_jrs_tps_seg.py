import logging
from os import path as osp

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import SimpleITK as sitk
from baseclass.medicalimage import Modality
from experiment.baseexperiment import BaseMSCMRExperiment
from tools.dir import sort_glob, mk_or_cleardir
from tools.set_random_seed import worker_init_fn
from jrs_networks.jrs_tps_seg_net import JRS2TpsSegNet as RS2Net
from dataloader.util import SkimageOP_MSCMR

from dataloader.jrsdataset import DataSetRJ as DataSetLoader
class Experiment(BaseMSCMRExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.args=args

        self.model = RS2Net(args)
        self.op = SkimageOP_MSCMR()
        if args.load:
            model_path = sort_glob(args.checkpoint_dir + f"/*.pth")[-1]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.model = self.model.load(model_path, self.device)
            logging.info(f'Model loaded from : {args.load} {model_path}')
        self.model.to(device= self.device)

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
        self.val_loader = DataLoader(DataSetLoader(self.args, type="test",augo=False,  task='pathology'),
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
                img, lab, roi_lab_myo, roi_lab_rv = self.create_torch_tensor(img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de)
                warp_img = {}
                warp_seg = {}
                para={}
                ori_array={}
                # train
                mv_seg,fix_seg,theta = self.model(img[Modality.c0], img[Modality.de])

                para['c0'] = sitk.ReadImage(c0_path)
                ori_array['c0'] = sitk.GetArrayFromImage(para['c0'])

                para['de'] = sitk.ReadImage(de_path)
                ori_array['de'] = sitk.GetArrayFromImage(para['de'])


                warp_seg[Modality.c0]=self.model.warp(mv_seg, theta)

                # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
                # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
                de_pred=torch.argmax(fix_seg,dim=1,keepdim=True)
                c0_pred=torch.argmax(mv_seg,dim=1,keepdim=True)


                warp_seg[Modality.c0]=torch.argmax(warp_seg[Modality.c0],dim=1,keepdim=True)


                warp_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(), theta)


                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                self.save_diff_img(ori_array['c0'],ori_array['de'],output+"_diff",self.renamepath(c0_path, 'diff_img')+".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(),ori_array['de'],output+"_diff",self.renamepath(c0_path, 'diff_warp_img')+".png")


                self.save_img_with_tps(ori_array['c0'],theta,output+"_tps", self.renamepath(c0_path, 'tps_img')+".png")
                self.save_img(ori_array['de'],output+"_tps",self.renamepath(de_path, 'tps_img')+".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'],output , self.renamepath(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'],output, self.renamepath(c0_path, 'assn_lab'),is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'],output, self.renamepath(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'],output, self.renamepath(de_path, 'assn_lab'),is_label=True)

                self.save_tensor_with_parameter(c0_pred, para['c0'],output, self.renamepath(c0_path, 'branch_lab'),is_label=True)
                self.save_tensor_with_parameter(de_pred, para['de'],output, self.renamepath(de_path, 'branch_lab'),is_label=True)

        # logging.info(f'slice level evaluation reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
        # logging.info(f'slice level evaluation seg mv:{np.mean(seg_error["mv"])} | fix: {np.mean(seg_error["fix"])}')

        seg_ds = {"c0": [], "t2": [], "de": []}
        seg_hds = {"c0": [], "t2": [], "de": []}
        seg_asds = {"c0": [], "t2": [], "de": []}
        reg_ds = {"c0": [], "t2": [], "de": []}
        reg_hds = {"c0": [], "t2": [], "de": []}

        for dir in range(26, 46):
            for modality in ['c0', 'de']:
                seg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*{modality}_*nii.gz")
                seg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_branch_lab*nii.gz")

                reg_gds = sort_glob(f"{self.args.dataset_dir}/subject_{dir}/*ana_*de_*nii.gz")
                reg_preds = sort_glob(f"{self.args.gen_dir}/subject_{dir}/*{modality}_assn_lab*nii.gz")

                print(f"================>{dir}")
                if len(seg_gds) == 0:
                    continue
                try:
                    ds_res, hd_res,asd_res = self.cal_ds_hd(seg_gds, seg_preds, {1: [1220, 2221, 200]})
                    seg_ds[modality].append(ds_res)
                    seg_hds[modality].append(hd_res)
                    seg_asds[modality].append(asd_res)

                    ds_res, hd_res ,asd_res= self.cal_ds_hd(reg_gds, reg_preds, {1: [1220, 2221, 200]})
                    reg_ds[modality].append(ds_res)
                    reg_hds[modality].append(hd_res)
                except Exception as e:
                    logging.error(e)

        print("=========segmentation=================")
        self.print_res(seg_ds, seg_hds, 'seg')
        print("=========registration=================")
        self.print_res(reg_ds, reg_hds, 'reg')


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
        from jrs_networks.jrs_losses import BinaryGaussianDice
        from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
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
                img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0, img_t2,img_de, lab_c0, lab_t2, lab_de)
                warp_img={}
                warp_roi_lab_myo={}
                sts_loss = {}
                warp_roi_lab_lv={}
                # train
                mv_seg,fix_seg,theta= self.model(img[Modality.c0], img[Modality.de])
                #loss
                warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta)
                warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta)
                warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta)
                loss_reg_myo = regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.c0])
                loss_reg_lv = regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.c0])
                loss_seg_fix = segcrit(fix_seg, roi_lab_myo[Modality.de])
                loss_seg_mv = segcrit(mv_seg, roi_lab_myo[Modality.c0])
                loss_all=(loss_seg_fix+loss_seg_mv)+self.args.weight*(loss_reg_myo+loss_reg_lv)
                #statistic

                sts_loss["train/reg_myo"]=(loss_reg_myo.item())
                sts_loss["train/reg_lv"]=(loss_reg_lv.item())
                sts_loss["train/loss_seg_fix"]=(loss_seg_fix.item())
                sts_loss["train/loss_seg_mv"]=(loss_seg_mv.item())
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
        print("evaluation................")

        # reg_error = {"init": [], "reg": []}
        # seg_error = {"mv": [], "fix": []}
        # with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:

        i = 0
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path, t2_path, de_path in self.all_loader:
                img, lab, roi_lab_myo, roi_lab_rv = self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de,lab_t2)
                warp_img = {}
                warp_lab = {}
                warp_roi_lab_myo = {}
                warp_roi_lab_rv = {}
                # train
                mv_seg, fix_seg, theta = self.model(img[Modality.c0], img[Modality.de])
                mv_seg = torch.argmax(mv_seg, dim=1, keepdim=True)
                fix_seg = torch.argmax(fix_seg, dim=1, keepdim=True)

                warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta)

                warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], theta, mode='nearest')

                warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta, mode='nearest')

                warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta, mode='nearest')

                # bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask


                #warped c0
                self.save_torch_to_nii(self.args.gen_dir,warp_img[Modality.c0],c0_path, 'c0_ori',is_lab=False)
                self.save_torch_to_nii(self.args.gen_dir,warp_roi_lab_myo[Modality.c0],c0_path, 'c0_ori',is_lab=True)
                self.save_torch_to_nii(self.args.gen_dir, mv_seg, c0_path, "c0_pred",is_lab=True)
                #de
                self.save_torch_to_nii(self.args.gen_dir,img[Modality.de],de_path, 'de_ori',is_lab=False)

                myo_scar_mask = lab_de.narrow(dim=1, start=6, length=1)
                self.save_torch_to_nii(self.args.gen_dir,myo_scar_mask,de_path, 'de_ori',is_lab=True)
                self.save_torch_to_nii(self.args.gen_dir, fix_seg, c0_path, "de_pred",is_lab=True)

'''
de branch的skip connection 来源于encoder_t2 encoder_
'''

# from tools.np_metric import dice_multi_class
# class Experiment_patho(BaseMyoPSExperiment):
#     def __init__(self,args):
#         super().__init__(args)
#         self.args=args
#         if args.data_source=="unaligned":
#             from dataloader.myopsdataset import MyoPSDataSet_unliagnedV2 as DataSetLoader
#         if args.data_source=="aff_aligned":
#             from dataloader.myopsdataset import MyoPSDataSet_aff_aligendV2 as DataSetLoader
#
#         if args.net == "tps":
#             from jrs_networks.jrs_tps_seg_net import JRS2TpsPathoSegNet as RpSNet
#         else:
#             logging.error("unimplmented type")
#             exit(404)
#
#         self.model = RpSNet(args)
#
#         if args.load:
#             model_path = sort_glob(args.checkpoint_dir + f"/*{args.epochs}.pth")[-1]
#             # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
#             self.model = self.model.load(model_path, self.device)
#             logging.info(f'Model loaded from {args.load}')
#         self.model.to(device= self.device)
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
#         print("evaluation................")
#
#         reg_error={"init":[],"reg":[]}
#         seg_error={"mv":[],"fix":[]}
#         # with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
#         i=0
#         with torch.no_grad():
#             for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
#                 img, lab, roi_lab_myo, roi_lab_rv,roi_lab_scar_myo = self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
#                 warp_img = {}
#                 warp_lab = {}
#                 warp_roi_lab_myo = {}
#                 warp_roi_lab_rv = {}
#                 # train
#                 mv_seg,fix_seg,theta = self.model(img[Modality.c0], img[Modality.de])
#                 mv_seg=torch.argmax(mv_seg,dim=1,keepdim=True)
#                 fix_seg=torch.argmax(fix_seg,dim=1,keepdim=True)
#
#                 warp_img[Modality.c0]=self.model.warp(img[Modality.c0], theta)
#
#                 warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], theta, mode='nearest')
#
#                 warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta, mode='nearest')
#
#                 warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta, mode='nearest')
#
#                 if self.args.save_imgs:
#                     self.save_torch_img_lab(self.args.output_dir, img[Modality.de], roi_lab_scar_myo[Modality.de], de_path, 'de_ori')
#                     self.save_torch_img_lab(self.args.output_dir, img[Modality.c0], roi_lab_scar_myo[Modality.c0], c0_path, 'c0_ori')
#                     self.save_torch_img_lab(self.args.output_dir, warp_img[Modality.c0], warp_roi_lab_myo[Modality.c0], c0_path, 'c0_warp')
#                     self.save_torch_to_nii(self.args.output_dir, mv_seg, c0_path, "c0_pred")
#                     self.save_torch_to_nii(self.args.output_dir, fix_seg, c0_path, "de_pred")
#
#
#                 for fix, warp_mv,ori_mv in zip(roi_lab_myo[Modality.de],warp_roi_lab_myo[Modality.c0],roi_lab_myo[Modality.c0]):
#                     warp_mv = (warp_mv>self.args.out_threshold ).cpu().numpy().astype(np.int16)
#                     ori_mv = (ori_mv > self.args.out_threshold).cpu().numpy().astype(np.int16)
#                     fix=fix.cpu().numpy().astype(np.int16)
#                     reg_error['reg'].append(dc(np.squeeze(warp_mv),np.squeeze(fix)))
#                     reg_error['init'].append(dc(np.squeeze(ori_mv),np.squeeze(fix)))
#
#                 for predict,gd in zip(mv_seg,roi_lab_myo[Modality.c0]):
#                     predict=predict.cpu().numpy().astype(np.int16)
#                     gd=gd.cpu().numpy().astype(np.int16)
#                     seg_error['mv'].append(dc(np.squeeze(gd),np.squeeze(predict)))
#
#                 for predict,gd in zip(fix_seg,roi_lab_myo[Modality.de]):
#                     predict=predict.cpu().numpy().astype(np.int16)
#                     gd=gd.cpu().numpy().astype(np.int16)
#                     seg_error['fix'].append(dc(np.squeeze(gd),np.squeeze(predict)))
#
#         logging.info(f'eval reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
#         logging.info(f'eval seg mv: {np.mean(seg_error["mv"])}| fix:{np.mean(seg_error["fix"])}')
#
#
#         subjets=sort_glob(f"{self.args.output_dir}/*")
#         ds={"c0":[],"t2":[],"de":[]}
#         for dir in subjets:
#             for modality in ['c0','t2','de']:
#                 gds=sort_glob(f"{dir}/*lab_{modality}_ori*nii.gz")
#                 preds=sort_glob(f"{dir}/*lab_{modality}_pred*nii.gz")
#                 gds_list=[]
#                 preds_list=[]
#                 assert len(gds)==len(preds)
#                 if len(gds)==0:
#                     continue
#                 for gd,pred in zip(gds,preds):
#                     print(f"{gd}-{pred}")
#                     gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)),axis=0))
#                     preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)),axis=0))
#                 gds_arr=np.concatenate(gds_list,axis=0)
#                 preds_arr=np.concatenate(preds_list,axis=0)
#                 res=dice_multi_class(gds_arr,preds_arr)
#                 ds[modality].append(res)
#
#         for k in ds.keys():
#             if (len(ds[k]))>0:
#                 logging.info(f'subject level evaluation:  seg {k}: {np.mean(ds[k],axis=0)}')
#
#
#
#
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
#         # crit_reg=BinaryDiceLoss()
#         from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
#         from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
#         from nnunet.utilities.nd_softmax import softmax_helper
#         # crit_reg=BinaryGaussianDice()
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
#                 img, lab, roi_lab_myo, roi_lab_lv,roi_lab_scar_myo= self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
#                 warp_img={}
#                 warp_roi_lab_myo={}
#                 warp_roi_lab_lv={}
#                 # train
#                 mv_seg,fix_seg,theta= self.model(img[Modality.c0], img[Modality.de])
#                 #loss
#                 warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta)
#                 warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta)
#                 warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta)
#                 loss_reg_myo = regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.c0])
#                 loss_reg_lv = regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.c0])
#                 loss_seg_mv = segcrit(mv_seg, roi_lab_myo[Modality.c0])
#
#                 loss_seg_fix = segcrit(fix_seg, roi_lab_scar_myo[Modality.de])
#
#                 loss_all=(loss_seg_fix+loss_seg_mv)+self.args.weight*(loss_reg_myo+loss_reg_lv)
#                 #statistic
#                 sts_loss={}
#                 sts_loss["train/reg_myo"]=(loss_reg_myo.item())
#                 sts_loss["train/reg_lv"]=(loss_reg_lv.item())
#                 sts_loss["train/loss_seg_fix"]=(loss_seg_fix.item())
#                 sts_loss["train/loss_seg_mv"]=(loss_seg_mv.item())
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
#             if (epoch+1)%self.args.save_freq==0:
#                 try:
#                     self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
#                     self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
#                     self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
#                     self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
#                     self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
#                     self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)
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
#
#     def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
#         img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
#         img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
#         img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
#         lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
#         lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
#         lab_de = lab_de.to(device=self.device, dtype=torch.float32)
#         #bg_mask, myo_mask, edema_scar_mask, scar_mask, lv_mask, rv_mask,ori_label
#         c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
#         t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
#         de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
#         c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=4, length=1)
#         t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=4, length=1)
#         de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=4, length=1)
#         c0_roi_reg_mask_3 = lab_c0.narrow(dim=1, start=3, length=1)#rv
#         t2_roi_reg_mask_3 = lab_t2.narrow(dim=1, start=3, length=1)#rv
#         de_roi_reg_mask_3 = lab_de.narrow(dim=1, start=3, length=1)#rv
#         img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
#         lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
#         roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
#         roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
#         roi_lab3={Modality.c0:c0_roi_reg_mask_3+c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_3+t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_3+de_roi_reg_mask_1}
#
#         return  img,lab,roi_lab1,roi_lab2,roi_lab3
#
