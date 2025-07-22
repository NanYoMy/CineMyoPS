import logging
import os
from os import path as osp

import SimpleITK as sitk
import numpy as np
import torch
from medpy.metric import dc
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baseclass.medicalimage import Modality
from dataloader.jrsdataset import DataSetWarpedC0RJ as DataSetLoader
from dataloader.util import SkimageOP_MSCMR
from experiment.baseexperiment import BaseMSCMRExperiment
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from tools.dir import sort_time_glob, mkdir_if_not_exist, sort_glob, mk_or_cleardir
from tools.itkdatawriter import sitk_write_image
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.set_random_seed import worker_init_fn
from unet import UNet


class Experiment_warped_c0_RJ(BaseMSCMRExperiment):

    def __init__(self,args):

        self.args=args

        self.net = UNet(args)
        self.op = SkimageOP_MSCMR()
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.load:
            if self.args.ckpt == -1:
                model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[-1]
            else:
                model_path = sort_time_glob(args.checkpoint_dir + f"/*{self.args.ckpt}*.pth")[-1]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.net = self.net.load(model_path, self.device)
            logging.info(f'Model loaded from {model_path}')
        self.net.to(device= self.device)

        self.train_loader = DataLoader(DataSetLoader(args, type="train", task='pathology'),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(DataSetLoader(args, type="test", augo=False, task='pathology'),
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     worker_init_fn=worker_init_fn)
        if args.print_tb:
            self.eval_writer = SummaryWriter(log_dir=f"{args.log_dir}")
        else:
            self.eval_writer = None
        logging.info(f'''Starting training:
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Learning rate:   {self.args.lr}
            Optimizer:       {self.args.optimizer}
            Checkpoints:     {self.args.save_cp}
            Device:          {self.device.type}
            load:           {self.args.load}
        ''')


    def save_img_lab(self,output_dir,img,lab,name,modality):
        img = np.squeeze(img[0].detach().cpu().numpy())
        lab = np.squeeze(lab[0].detach().cpu().numpy())

        output_dir=f"{output_dir}/{os.path.basename(os.path.dirname(name[0]))}"
        mkdir_if_not_exist(output_dir)

        term=os.path.basename((name[0])).split("_")
        img_name=f'{term[0]}_{term[1]}_img_{modality}_{term[-1]}'
        lab_name=f'{term[0]}_{term[1]}_lab_{modality}_{term[-1]}'

        sitk_write_image(img,None,output_dir,img_name)
        sitk_write_image(np.round(lab),None,output_dir,lab_name)

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

                c0_lab_path=c0_path[0].replace("_C0_","_C0_manual_")
                t2_lab_path=t2_path[0].replace('_T2_','_T2_manual_')
                de_lab_path=de_path[0].replace('_DE_','_DE_manual_')

                img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv = self.create_torch_tensor(img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de)
                # warp_pred = {}
                pred={}
                para={}
                ori_img={}
                ori_lab={}
                tmp = self.net(img[self.args.modality])
                c0_seg=tmp
                t2_seg=tmp
                lge_seg=tmp

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


                pred['de']=torch.argmax(lge_seg,dim=1,keepdim=True)
                pred['c0']=torch.argmax(c0_seg,dim=1,keepdim=True)
                pred['t2']=torch.argmax(t2_seg,dim=1,keepdim=True)
                # pred['de']=(lge_seg>0.5).short()
                # pred['c0']=(c0_seg >0.5).short()
                # pred['t2']=(t2_seg>0.5).short()

                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                from visulize.color import colorC0,colorDe,colorT2,colorGD
                if self.args.save_imgs==True:

                    # self.save_img_with_mask(ori_img['de'],
                    #                             reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221]}),
                    #                             output + "_contours", self.renamepath(de_path, 'gt_con_img') + ".png")

                    if self.args.modality==Modality.c0:

                        self.save_image_with_pred_gt_contousV2(ori_img['t2'],
                                                             reindex_label_array_by_dict(pred['t2'].cpu().numpy(),
                                                                                         {3: [1, 200, 1220, 2221]}),
                                                             reindex_label_array_by_dict(ori_lab['t2'],
                                                                                         {1: [1, 200, 1220, 2221]}),
                                                           output+"_seg_contours",
                                                           self.renamepath(t2_path, 'warp_img_pre')+".png",colorT2,colorGD)

                        self.save_image_with_pred_gt_contousV2(ori_img['de'],
                                                           reindex_label_array_by_dict(pred['de'].cpu().numpy(),{3:[1,200,1220,2221]}),
                                                           reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221]}),
                                                           output+"_seg_contours",
                                                           self.renamepath(de_path, 'warp_img_pre')+".png",colorDe,colorGD)
                    elif self.args.modality==Modality.t2:

                        self.save_image_with_pred_gt_contousV2(ori_img['t2'],
                                                             reindex_label_array_by_dict(pred['t2'].cpu().numpy(),
                                                                                         {3: [1, 200, 1220, 2221]}),
                                                             reindex_label_array_by_dict(ori_lab['t2'],
                                                                                         {1: [1, 200, 1220, 2221]}),
                                                           output+"_seg_contours",
                                                           self.renamepath(t2_path, 'warp_img_pre')+".png",colorT2,colorGD)


                    elif self.args.modality==Modality.de:
                        self.save_image_with_pred_gt_contousV2(ori_img['de'],
                                                           reindex_label_array_by_dict(pred['de'].cpu().numpy(),{3:[1,200,1220,2221]}),
                                                           reindex_label_array_by_dict(ori_lab['de'],{1:[1,200,1220,2221]}),
                                                           output+"_seg_contours",
                                                           self.renamepath(de_path, 'warp_img_pre')+".png",colorDe,colorGD)







                self.save_tensor_with_parameter(pred['c0'], para['c0'],output, self.renamepath(c0_path, 'branch_lab'),is_label=True)
                self.save_tensor_with_parameter(pred['t2'], para['t2'],output, self.renamepath(t2_path, 'branch_lab'),is_label=True)
                self.save_tensor_with_parameter(pred['de'], para['de'],output, self.renamepath(de_path, 'branch_lab'),is_label=True)


        # logging.info(f'slice level evaluation reg mv->fix: {np.mean(reg_error["init"])}| warp_mv->fix:{np.mean(reg_error["reg"])}')
        # logging.info(f'slice level evaluation seg mv:{np.mean(seg_error["mv"])} | fix: {np.mean(seg_error["fix"])}')

        seg_ds = {"c0": [], "t2": [], "de": []}
        seg_hds = {"c0": [], "t2": [], "de": []}
        seg_asds = {"c0": [], "t2": [], "de": []}
        # reg_ds = {"c0": [], "t2": [], "de": []}
        # reg_hds = {"c0": [], "t2": [], "de": []}

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
                    ds_res, hd_res,asd_res = self.cal_ds_hd(seg_gds, seg_preds, {1: [1220, 2221, 200]})
                    seg_ds[modality].append(ds_res)
                    seg_hds[modality].append(hd_res)
                    seg_asds[modality].append(asd_res)

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

    def renamepath(self, name, tag):
        term = os.path.basename((name[0])).split("_")
        name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}'
        return name

    def train_eval_net(self):
        global_step = 0
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=self.args.lr)
        elif self.args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.net.parameters(),
                                      lr=self.args.lr,
                                      weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.SGD(self.net.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay,
                                  nesterov=self.args.nesterov)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_decay_milestones,
                                                   gamma = self.args.lr_decay_gamma)


        # crit_seg=BinaryDiceLoss()
        crit_seg = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

        self.net.train() # dropout, BN在train和test的行为不一样
        mk_or_cleardir(self.args.log_dir)
        # loss=0

        for epoch in range(self.args.init_epochs,self.args.epochs):
            epoch_loss = 0
            np.random.seed(17 + epoch)
            #前1000个epoch只进行配准

            # if epoch <2000:
            #     loss_weight = [1.0, 1.0]
            # else:
            #     loss_weight = [0.1, 1.0]
            loss_weight = [0.0, 1.0]

            # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            print("train.................")
            for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
                #load data
                img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv= self.create_torch_tensor(img_c0, img_t2,img_de, lab_c0, lab_t2, lab_de)

                # train
                y_c0_lab = self.net(img[self.args.modality])

                loss_seg=crit_seg(y_c0_lab, roi_lab_myo[self.args.modality])
                loss_all=loss_weight[1]*loss_seg
                epoch_loss += loss_all.item()
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                global_step += 1

            if self.args.print_tb:
                # self.eval_writer.add_scalar("train/reg", loss_all.item(), global_step)
                self.eval_writer.add_scalar("train/seg", loss_seg.item(), global_step)


            scheduler.step()

            if (epoch+1)%self.args.save_freq==0:
                try:
                    # evaluation befor save checkpoints
                    self.eval_writer.add_images("train/imgs/img", img[self.args.modality], global_step)
                    # self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                    # self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                    # self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/lv_de", roi_lab_lv[Modality.de], global_step)
                    # self.eval_writer.add_images("train/labs/lv_t2", roi_lab_lv[Modality.t2], global_step)
                    self.eval_writer.add_images("train/labs/gt", roi_lab_myo[self.args.modality], global_step)
                    self.eval_writer.add_images("train/labs/preditct", torch.argmax(y_c0_lab,dim=1,keepdim=True), global_step)
                    # self.eval_writer.add_images("train/labs/myo_de", roi_lab_myo[Modality.de], global_step)
                    # self.eval_writer.add_images("train/labs/myo_t2", roi_lab_myo[Modality.t2], global_step)
                    # self.eval_writer.add_images("train/labs/myo_c0", roi_lab_myo[Modality.c0], global_step)
                    self.net.eval()
                    self.validate_net()
                    self.net.train()
                    ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                    self.net.save(osp.join(self.args.checkpoint_dir, ckpt_name))
                    logging.info(f'Checkpoint {epoch + 1} saved !')
                except OSError:
                    logging.info(f'Checkpoint {epoch + 1} save failed !')
                    exit(-999)
                # if self.args.print_tb == True:
                #     self.eval_writer.add_scalar('evluation/myo_dice', val_score[0], global_step)

        if self.args.print_tb==True:
            self.eval_writer.close()

    # def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
    #     img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
    #     img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
    #     img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
    #     lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
    #     lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
    #     lab_de = lab_de.to(device=self.device, dtype=torch.float32)
    #     c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
    #     t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
    #     de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
    #     c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=5, length=1)
    #     t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=5, length=1)
    #     de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=5, length=1)
    #     img={Modality.c0.name:img_c0, Modality.t2.name:img_t2, Modality.de.name:img_de}
    #     lab={Modality.c0.name:lab_c0, Modality.t2.name:lab_t2, Modality.de.name:lab_de}
    #     roi_lab1={Modality.c0.name:c0_roi_reg_mask_1, Modality.t2.name:t2_roi_reg_mask_1, Modality.de.name:de_roi_reg_mask_1}
    #     roi_lab2={Modality.c0.name:c0_roi_reg_mask_2, Modality.t2.name:t2_roi_reg_mask_2, Modality.de.name:de_roi_reg_mask_2}
    #     return  img,lab,roi_lab1,roi_lab2