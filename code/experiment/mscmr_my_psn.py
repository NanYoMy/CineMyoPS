import logging
from os import path as osp

import SimpleITK as sitk
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from baseclass.medicalimage import Modality
from dataloader.jrsdataset import DataSetRJ_PSN as DataSetLoader
from dataloader.util import SkimageOP_RJ_PSN
from experiment.baseexperiment import BaseMSCMRExperiment
from jrs_networks.jrs_patho_seg import Mask2B_AttU_Unet
# from jrs_networks.jrs_3m_tps_seg_net import JRS3MROITpsSegNet as RSNet
from tools.dir import sort_time_glob, sort_glob, mk_or_cleardir,natsort_glob
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.set_random_seed import worker_init_fn
import itertools
'''
renji的数据的实验
'''
from tools.dir import mkdir_if_not_exist
from tools.metric import print_mean_and_std
import os
from tools.itkdatawriter import sitk_write_image
from medpy.metric import dc, specificity as spec, sensitivity as sens, precision as prec
from tools.np_sitk_tools import reindex_label_array_by_dict
class ExperimentRJ_PSN(BaseMSCMRExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.args=args

        self.model = Mask2B_AttU_Unet(args)
        self.op = SkimageOP_RJ_PSN()
        if args.load:
            if self.args.ckpt==-1:
                model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[-1]
            else:
                model_path = sort_time_glob(args.checkpoint_dir + f"/*{self.args.ckpt}*.pth")[-1]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            logging.info(f'Model loaded from : {args.load} {model_path}')
            self.model = self.model.load(model_path, self.device)
        self.model.to(device= self.device)


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
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Optimizer:       {self.args.optimizer}
            Checkpoints:     {self.args.save_cp}
            Device:          {self.device.type}
            load:           {self.args.load}
        ''')
    def save_torch_to_nii(self, output_dir, array, source_img_path, modality, is_lab=True):
        array = np.squeeze(array[0].detach().cpu().numpy())
        param=sitk.ReadImage(source_img_path)
        param_array=np.squeeze(sitk.GetArrayFromImage(param))
        array=self.op.resize(array,param_array.shape,order=0)
        output_dir=f"{output_dir}/{os.path.basename(os.path.dirname(source_img_path[0]))}"
        mkdir_if_not_exist(output_dir)
        term=os.path.basename((source_img_path[0])).split("_")

        if is_lab==True:
            lab_name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{modality}_{term[6]}'
            sitk_write_image(np.round(array).astype(np.int16), None, output_dir, lab_name)
        else:

            img_name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{modality}_{term[6]}'
            sitk_write_image(array, None, output_dir, img_name)

    def validate_net(self):

        print(f"evaluating.....................................")
        self.val_loader = DataLoader(DataSetLoader(self.args, type="test",augo=False,  task='pathology'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)
        mk_or_cleardir(self.args.output_dir)
        with torch.no_grad():
            for c0_data, t2_data, de_data in self.val_loader:
                img, prior, lab = self.to_cuda(c0_data, t2_data, de_data)
                # train

                pred_scar, pred_edema = self.model(img[Modality.t2], img[Modality.de], None)
                pred_scar=torch.argmax(pred_scar,dim=1,keepdim=True)
                pred_edema=torch.argmax(pred_edema,dim=1,keepdim=True)



                self.save_torch_to_nii(self.args.output_dir, pred_scar, de_data["path"], "de_pred_scar")
                self.save_torch_to_nii(self.args.output_dir, pred_edema, t2_data["path"], "t2_pred_edema")

        pred_subjets=natsort_glob(f"{self.args.output_dir}/*")
        gt_subjets=natsort_glob(f"{self.args.dataset_dir}/*[0-9]")[-25:]

        dice = {"de": [], 't2': []}
        precision = {"de": [], 't2': []}
        sensitivity = {"de": [], 't2': []}
        specifitivity = {"de": [], 't2': []}
        res = {'dice': dice, 'prec': precision, "sens": sensitivity, 'spec': specifitivity}
        for gt_dir,dir in zip(gt_subjets,pred_subjets):
            for modality in ['t2','de']:
                gds=sort_glob(f"{gt_dir}/*{modality}_*gt_lab*nii.gz")
                preds=sort_glob(f"{dir}/*{modality}_pred*nii.gz")
                gds_list=[]
                preds_list=[]
                assert len(gds)==len(preds)
                if len(gds)==0:
                    continue
                for gd,pred in zip(gds,preds):
                    # print(f"{gd}-{pred}")
                    gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)),axis=0))
                    preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)),axis=0))
                gds_arr=np.squeeze(np.concatenate(gds_list,axis=0))
                preds_arr=np.squeeze(np.concatenate(preds_list,axis=0))
                gds_arr=reindex_label_array_by_dict(gds_arr,{1:[1220,2221]})
                res['dice'][modality].append(dc(preds_arr, gds_arr))
                res['spec'][modality].append(spec(preds_arr, gds_arr))
                res['sens'][modality].append(sens(preds_arr, gds_arr))
                res['prec'][modality].append(prec(preds_arr, gds_arr))
        # print(res)
        # for m in ['dice','spec','sens','prec']:
        for m in ['dice']:
            for t in ['de','t2']:
                print(f"============={t}-{m}=============")
                print_mean_and_std(res[m][t],detail=False)


    def train_eval_net(self):
        global_step = 0
        self.train_loader = DataLoader(DataSetLoader(self.args, type="train",augo=True, task='pathology'),
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=1,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_fn)
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

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        #                                            milestones=self.args.lr_decay_milestones,
        #                                            gamma = self.args.lr_decay_gamma)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=self.args.lr_decay_gamma)
        # scheduler =optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma=self.args.lr_decay_gamma,verbose=True)

        MAX_STEP=int(1e10)
        scheduler =optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-5)


        # crit_reg=BinaryDiceLoss()
        from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
        from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
        from nnunet.utilities.nd_softmax import softmax_helper
        # crit_reg=BinaryGaussianDice()
        regcrit=BinaryGaussianDice(sigma=1)
        segcrit=DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {},weight_ce=0)
        # segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
        self.model.train() # dropout, BN在train和test的行为不一样
        mk_or_cleardir(self.args.log_dir)
        # loss=0

        for epoch in range(self.args.init_epochs,self.args.epochs+1):
            sts_total_loss = 0
            np.random.seed(17 + epoch)
            #前1000个epoch只进行配准

            print("train.................")
            for c0_data,t2_data,de_data in self.train_loader:
                #load data
                img, prior,lab= self.to_cuda(c0_data,t2_data,de_data)
                sts_loss = {}

                # scar,edema= self.model(img[Modality.t2], img[Modality.de],prior)
                pred_scar,pred_edema= self.model(img[Modality.t2], img[Modality.de],prior)
                # loss_all=segcrit(pred_scar,lab[Modality.de])+segcrit(pred_edema,lab[Modality.t2])
                loss_all=segcrit(pred_scar,lab[Modality.de])


                sts_loss["train/reg_total"]=(loss_all.item())

                optimizer.zero_grad()
                loss_all.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(),0.1)
                optimizer.step()
                global_step += 1

                self.write_dict_to_tb(sts_loss,global_step)

            scheduler.step()
            print(scheduler.get_lr())
            if (epoch)%self.args.save_freq==0:
                try:
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                    # self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)
                    self.eval_writer.add_images("train/labs/de", lab[Modality.de], global_step)
                    self.eval_writer.add_images("train/labs/t2", lab[Modality.t2], global_step)
                    # self.eval_writer.add_images("train/labs/lv_c0", lab[Modality.c0], global_step)
                    self.eval_writer.add_images("train/labs/prior", prior, global_step)

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

    def to_cuda(self, c0_data,t2_data,de_data):
        img_c0 = c0_data["img"].to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = t2_data["img"].to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = de_data["img"].to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd

        prior=c0_data["prior"]+t2_data["prior"]+de_data["prior"]
        prior=torch.where(prior>0,1,0)

        prior=prior.to(device=self.device, dtype=torch.float32)

        lab_c0 = c0_data["gt_lab"].to(device=self.device, dtype=torch.float32)
        lab_t2 = t2_data["gt_lab"].to(device=self.device, dtype=torch.float32)
        lab_de = de_data["gt_lab"].to(device=self.device, dtype=torch.float32)


        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}

        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}

        return  img,prior,lab

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