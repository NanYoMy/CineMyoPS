import logging
from os import path as osp

import numpy as np
import torch
from medpy.metric import dc
from torch import optim
from torch.utils.data import DataLoader
import SimpleITK as sitk
from baseclass.medicalimage import Modality
from experiment.baseexperiment import BaseMyoPSExperiment

from tools.dir import sort_glob, mk_or_cleardir
from tools.set_random_seed import worker_init_fn


class Experiment(BaseMyoPSExperiment):
    def __init__(self,args):
        super().__init__(args)
        self.args=args

        from dataloader.myops20dataset import MyoPS20DataSet as DataSetLoader

        if args.net == "tps":
            from jrs_networks.jrs_tps_seg_net import JRS2TpsSegNet as RSNet
        else:
            logging.error("unimplmented type")
            exit(-200)

        self.model = RSNet(args)

        if args.load:
            model_path = sort_glob(args.checkpoint_dir + f"/*.pth")[args.ckpt]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.model = self.model.load(model_path, self.device)
            logging.info(f'Model loaded from {args.load}')
        self.model.to(device= self.device)

        self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology',ret_path=False),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_fn)
        self.trainall_loader = DataLoader(DataSetLoader(args, type="trainall",augo=True, task='pathology',ret_path=False),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_fn)

        self.val_loader = DataLoader(DataSetLoader(args, type="vali",augo=False,  task='pathology',ret_path=True),
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
        print("validet................")
        reg_error={"init":[],"reg":[]}
        seg_error={"mv":[],"fix":[]}
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
                img, lab, roi_lab_myo, roi_lab_lv = self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
                warp_img = {}
                warp_lab = {}
                warp_roi_lab_myo = {}
                warp_roi_lab_lv = {}
                # train
                mv_seg,fix_seg,theta = self.model(img[Modality.c0], img[Modality.de])
                mv_seg=torch.argmax(mv_seg,dim=1,keepdim=True)
                fix_seg=torch.argmax(fix_seg,dim=1,keepdim=True)

                warp_img[Modality.c0]=self.model.warp(img[Modality.c0], theta)

                warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], theta, mode='nearest')

                warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta, mode='nearest')

                warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta, mode='nearest')

                self.save_torch_img_lab(self.args.output_dir, img[Modality.de], roi_lab_myo[Modality.de], de_path, 'de_ori')
                self.save_torch_img_lab(self.args.output_dir,img[Modality.c0],roi_lab_myo[Modality.c0],c0_path,'c0_ori')
                self.save_torch_img_lab(self.args.output_dir, warp_img[Modality.c0], warp_roi_lab_myo[Modality.c0], c0_path, 'c0_warp')
                self.save_torch_to_nii(self.args.output_dir, mv_seg, c0_path, "c0_pred")
                self.save_torch_to_nii(self.args.output_dir, fix_seg, c0_path, "de_pred")

        # logging.info(f'slice level evaluation reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
        # logging.info(f'slice level evaluation seg mv:{np.mean(seg_error["mv"])} | fix: {np.mean(seg_error["fix"])}')

        subjets=sort_glob(f"{self.args.output_dir}/*")
        ds={"c0":[],"t2":[],"de":[]}
        for dir in subjets:
            for modality in ['c0','t2','de']:
                gds=sort_glob(f"{dir}/*lab_{modality}_ori*nii.gz")
                preds=sort_glob(f"{dir}/*lab_{modality}_pred*nii.gz")
                gds_list=[]
                preds_list=[]
                assert len(gds)==len(preds)
                if len(gds)==0:
                    continue
                for gd,pred in zip(gds,preds):
                    # print(f"{gd}-{pred}")
                    gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)),axis=0))
                    preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)),axis=0))
                gds_arr=np.concatenate(gds_list,axis=0)
                preds_arr=np.concatenate(preds_list,axis=0)
                ds[modality].append(dc(gds_arr,preds_arr))

        for k in ds.keys():
            if (len(ds[k]))>0:
                logging.info(f'subject level evaluation:  seg {k}: {np.mean(ds[k])}')


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
                img, lab, roi_lab_myo, roi_lab_lv= self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
                warp_img={}
                warp_roi_lab_myo={}
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
                sts_loss={}
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
            if (epoch+1)%self.args.save_freq==0:
                try:
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)

                    self.eval_writer.add_images("train/labs/de", (lab[Modality.de]), global_step)
                    self.eval_writer.add_images("train/labs/t2", (lab[Modality.t2]), global_step)
                    self.eval_writer.add_images("train/labs/c0", (lab[Modality.c0]), global_step)

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

    def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask,myo_scar_ede_mask,ori_lab
        c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)

        c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=4, length=1)
        t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=4, length=1)
        de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=4, length=1)

        lab_c0 = lab_c0.narrow(dim=1, start=-1, length=1)
        lab_t2 = lab_t2.narrow(dim=1, start=-1, length=1)
        lab_de = lab_de.narrow(dim=1, start=-1, length=1)

        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
        roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab1,roi_lab2
