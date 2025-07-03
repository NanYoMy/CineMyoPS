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


from tools.metric import dice_multi_class
class Experiment_2M_Pathology(BaseMyoPSExperiment):

    def save_torch_img_mask_lab(self, output_dir, img,mask, lab, name, modality):
        self.save_torch_to_nii(output_dir,img,name,modality,is_lab=False)
        self.save_torch_to_nii(output_dir,mask,name,modality)
        self.save_torch_to_nii(output_dir,lab,name,modality)


    def __init__(self,args):
        super().__init__(args)
        self.args=args
        from dataloader.jrsdataset import jsr2MPathologyDataset as DataSetLoader
        from jrs_networks.jrs_patho_seg import Patho2Seg as PSNet
        # from jrs_networks.jrs_patho_seg import Patho2Seg_Masked as PSNet

        self.model = PSNet(args)
        if args.load:
            model_path = sort_glob(args.checkpoint_dir + f"/*{args.epochs}.pth")[-1]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.model = self.model.load(model_path, self.device)
            logging.info(f'Model loaded from {args.load}')
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

        with torch.no_grad():
            for img_c0,img_de,mask_c0,mask_de,lab_c0,lab_de, c0_path,de_path in self.val_loader:
                img, mask,lab= self.create_torch_tensor(img_c0,img_de,mask_c0,mask_de,lab_c0,lab_de)
                # train
                mv_seg,fix_seg = self.model(img[Modality.c0],mask[Modality.c0], img[Modality.de],mask[Modality.de])
                mv_seg=torch.argmax(mv_seg,dim=1,keepdim=True)
                fix_seg=torch.argmax(fix_seg,dim=1,keepdim=True)

                self.save_torch_img_mask_lab(self.args.output_dir, img[Modality.c0],mask[Modality.c0], lab[Modality.c0], de_path, 'c0_ori')
                self.save_torch_img_mask_lab(self.args.output_dir, img[Modality.de],mask[Modality.de], lab[Modality.de], de_path, 'de_ori')
                # self.save_img_lab(self.args.output_dir,img[Modality.c0],roi_lab_myo[Modality.c0],c0_path,'c0_ori')
                self.save_torch_to_nii(self.args.output_dir, mv_seg, c0_path, "c0_pred")
                self.save_torch_to_nii(self.args.output_dir, fix_seg, c0_path, "de_pred")

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
                if modality=='c0':
                    ds[modality].append(dice_multi_class(gds_arr,preds_arr,2))
                elif modality == 'de':
                    ds[modality].append(dice_multi_class(gds_arr, preds_arr,3))
        for k in ds.keys():
            if (len(ds[k]))>0:
                print(ds[k])
                logging.info(f'subject level evaluation:  seg {k}: {np.mean(ds[k],axis=0)}')


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

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
        #                                            milestones=self.args.lr_decay_milestones,
        #                                            gamma = self.args.lr_decay_gamma)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.args.lr_decay_gamma)

        from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
        from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, SoftDiceLoss
        from nnunet.utilities.nd_softmax import softmax_helper

        segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
        # segcrit=BinaryGaussianDice()
        segcrit_ds_ce=DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        self.model.train() # dropout, BN在train和test的行为不一样
        mk_or_cleardir(self.args.log_dir)

        for epoch in range(self.args.init_epochs,self.args.epochs+1):
            sts_total_loss = 0
            np.random.seed(17 + epoch)
            #前1000个epoch只进行配准

            print("train.................")
            for img_c0,img_de,mask_c0,mask_de,lab_c0,lab_de in self.train_loader:
                #load data
                img,mask,lab= self.create_torch_tensor(img_c0,img_de,mask_c0,mask_de,lab_c0,lab_de)

                # train
                mv_seg,fix_seg= self.model(img[Modality.c0],mask[Modality.c0], img[Modality.de],mask[Modality.de])
                #loss

                loss_seg_mv = segcrit(mv_seg, lab[Modality.c0])
                loss_seg_fix = segcrit_ds_ce(fix_seg, lab[Modality.de])
                loss_all=loss_seg_fix+self.args.weight*loss_seg_mv
                #statistic
                sts_loss={}

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
                    self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                    self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                    self.eval_writer.add_images("train/mask/c0", mask[Modality.c0], global_step)
                    self.eval_writer.add_images("train/mask/de", mask[Modality.de], global_step)
                    self.eval_writer.add_images("train/labs/c0", lab[Modality.c0], global_step)
                    self.eval_writer.add_images("train/labs/de", lab[Modality.de], global_step)
                    # self.save_torch_to_nii("../outputs/tmp/", lab[Modality.de], ['subject_00_lab_de_ori_0.nii.gz'], 'de', is_lab=True)
                    # self.save_torch_to_nii("../outputs/tmp/", img[Modality.de], ['subject_00_lab_de_ori_0.nii.gz'],  'de', is_lab=False)
                    self.eval_writer.add_images("train/pred/c0", torch.argmax(mv_seg, dim=1, keepdim=True), global_step)
                    self.eval_writer.add_images("train/pred/de", torch.argmax(fix_seg, dim=1, keepdim=True),global_step)
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

    def create_torch_tensor(self, img_c0,img_de,mask_c0,mask_de,lab_c0,lab_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        mask_c0 = mask_c0.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        mask_de = mask_de.to(device=self.device, dtype=torch.float32)
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_scar_mask, scar_mask, lv_mask, rv_mask,ori_label
        img={Modality.c0:img_c0, Modality.de:img_de}
        mask={Modality.c0:mask_c0, Modality.de:mask_de}
        lab={Modality.c0:lab_c0, Modality.de:lab_de}

        return  img,mask,lab