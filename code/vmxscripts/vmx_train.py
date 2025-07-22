# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:44
#单独分割某个类别  myo,scar,scar_edema, 通过class_index控制
"""
输入为多模态 未配准的图像，输出为单独一个前景+背景
"""

import logging
import os
import sys
import torch
from torch.utils.data import DataLoader
# from jrs_networks.jrs_tps_seg_net import JRSTpsSegNet

from medpy.metric import dc
import numpy as np
from tools.dir import mk_or_cleardir,mkdir_if_not_exist
from tools.set_random_seed import setup_seed, worker_init_fn

from tools.itkdatawriter import sitk_write_image
from torch.utils.tensorboard  import SummaryWriter
import voxelmorph as vxm  # nopep8
import time

MOLD_ID='myops'
from baseclass.medicalimage import Modality

class Experiment():
    def __init__(self,args):
        self.args=args
        # load and prepare training data
        if args.data_source=="unaligned":
            from dataloader.jrsdataset import DataSetRJ as DataSetLoader
        if args.data_source=="aff_aligned":
            from dataloader.jrsdataset import MyoPSDataSet_aff_aligendV2 as DataSetLoader

        self.train_loader = DataLoader(DataSetLoader(args, type="train", augo=True, task='pathology'),
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

        inshape = (args.image_size,args.image_size,)
        # prepare model folder
        model_dir = args.model_dir
        os.makedirs(model_dir, exist_ok=True)

        # device handling
        gpus = args.gpu.split(',')
        nb_gpus = len(gpus)
        self.device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        assert np.mod(args.batch_size, nb_gpus) == 0, \
            'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

        # enabling cudnn determinism appears to speed up training by a lot
        torch.backends.cudnn.deterministic = not args.cudnn_nondet

        # unet architecture
        enc_nf = args.enc if args.enc else [16, 32, 32, 32]
        dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

        if args.load_model:
            # load initial model (if specified)
            self.model = vxm.networks.VxmDense.load(args.load_model, self.device)
        else:
            # otherwise configure new model
            self.model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[enc_nf, dec_nf],
                bidir=args.bidir,
                int_steps=args.int_steps,
                int_downsize=args.int_downsize
            )

        # if nb_gpus > 1:
        #     # use multiple GPUs via DataParallel
        #     self.model = torch.nn.DataParallel(self.model)
        #     self.model.save = self.model.module.save

        # prepare the model for training and send to device
        self.model.to(self.device)
        self.initilize()
    def initilize(self):
        mk_or_cleardir(self.args.log_dir)
        self.eval_writer = SummaryWriter(log_dir=f"{self.args.log_dir}")
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

    def save_img_labV2(self,output_dir,img,lab,name,modality):
        img = np.squeeze(img[0].detach().cpu().numpy())
        lab = np.squeeze(lab[0].detach().cpu().numpy())

        mkdir_if_not_exist(output_dir)

        sitk_write_image(img,None,output_dir,f"img_{name}")
        sitk_write_image(np.round(lab),None,output_dir,f"lab_{name}")


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

        epoch_loss = []
        dc_total_loss = []
        dc_init_total_loss=[]
        epoch_step_time = []

        step_start_time = time.time()
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de,c0_path,t2_path,de_path in self.val_loader:
                # load data
                img, lab, roi_lab_myo, roi_lab_rv = self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
                warp_img = {}
                warp_lab = {}
                warp_roi_lab_myo = {}
                warp_roi_lab_rv = {}
                flow = self.model(img[Modality.de], img[Modality.t2])
                # y_pred = self.model(inputs)
                warp_img[Modality.t2] = self.model.warp(img[Modality.t2], flow)
                warp_lab[Modality.t2] = self.model.warp(lab[Modality.t2], flow)
                warp_roi_lab_myo[Modality.t2] = self.model.warp(roi_lab_myo[Modality.t2], flow, mode='nearest')
                warp_roi_lab_rv[Modality.t2] = self.model.warp(roi_lab_rv[Modality.t2], flow, mode='nearest')
                warp_lab[Modality.c0] = self.model.warp(lab[Modality.c0], flow, mode='nearest')
                warp_lab[Modality.t2] = self.model.warp(lab[Modality.t2], flow, mode='nearest')
                # calculate total loss
                loss_list = []
                loss_reg = vxm.losses.Dice().loss(warp_roi_lab_myo[Modality.t2], roi_lab_myo[Modality.de])
                loss_reg_init = vxm.losses.Dice().loss(roi_lab_myo[Modality.t2], roi_lab_myo[Modality.de])
                loss_bend =vxm.losses.BendEng('l2').loss(flow)
                # loss = loss_reg + self.args.weight * loss_bend
                loss_list.append(loss_reg.item())
                loss_list.append(self.args.weight * loss_bend.item())
                epoch_loss.append(loss_list)
                dc_total_loss.append(loss_reg.item())
                dc_init_total_loss.append(loss_reg_init.item())
                # get compute time
                epoch_step_time.append(time.time() - step_start_time)


                if self.args.save_imgs:
                    self.save_img_lab(self.args.output_dir, warp_img[Modality.t2], warp_roi_lab_myo[Modality.t2], t2_path, 't2_warp')
                    self.save_img_lab(self.args.output_dir, img[Modality.de], roi_lab_myo[Modality.de], de_path, 'de_ori')
                    self.save_img_lab(self.args.output_dir, img[Modality.t2], roi_lab_myo[Modality.t2], t2_path, 't2_ori')


            # print epoch info
        time_info = 'evaluation current : %.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'dc_loss: %.4e, -> %.4e  (%s)' % (np.mean(dc_init_total_loss)  , np.mean(dc_total_loss), losses_info)
        print(dc_init_total_loss)
        print(dc_total_loss)


        print(' - '.join((time_info, loss_info)), flush=True)


    def train_eval_net(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # prepare image loss
        # if args.image_loss == 'ncc':
        #     image_loss_func = vxm.losses.NCC().loss
        # elif args.image_loss == 'mse':
        #     image_loss_func = vxm.losses.MSE().loss
        # elif args.image_loss == 'dc':
        #     image_loss_func=vxm.losses.Dice().loss
        # else:
        #     raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
        #
        # # need two image loss functions if bidirectional
        # if args.bidir:
        #     losses = [image_loss_func, image_loss_func]
        #     weights = [0.5, 0.5]
        # else:
        #     losses = [image_loss_func]
        #     weights = [1]
        #
        # # prepare deformation loss
        # losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
        # weights += [args.weight]
        from jrs_networks.jrs_losses import GaussianNGF,BinaryGaussianDice
        global_step=0
        gauDice=BinaryGaussianDice(sigma=1)
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []
            step_start_time = time.time()
            for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
                #load data
                img, lab, roi_lab_myo, roi_lab_rv= self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
                warp_img={}
                warp_lab={}
                warp_roi_lab_myo={}
                warp_roi_lab_rv={}
                # train
                # save model checkpoint
                flow = self.model(img[Modality.de], img[Modality.t2])
                # y_pred = self.model(inputs)
                warp_img[Modality.t2]=self.model.warp(img[Modality.t2],flow)
                warp_lab[Modality.t2]=self.model.warp(lab[Modality.t2],flow)
                warp_roi_lab_myo[Modality.t2]=self.model.warp(roi_lab_myo[Modality.t2],flow)
                warp_roi_lab_rv[Modality.t2]=self.model.warp(roi_lab_rv[Modality.t2],flow)

                # calculate total loss
                loss_list = []

                # loss_reg=vxm.losses.Dice().loss(warp_roi_lab_myo[Modality.t2],roi_lab_myo[Modality.de])
                # loss_reg=gauNGF(warp_img[Modality.t2],img[Modality.de])
                loss_reg_myo=gauDice(warp_roi_lab_myo[Modality.t2],roi_lab_myo[Modality.de])
                loss_reg_rv=gauDice(warp_roi_lab_rv[Modality.t2],roi_lab_rv[Modality.de])*0
                # loss_bend=vxm.losses.Grad2D('l2', loss_mult=self.args.int_downsize).loss(None,flow)
                loss_bend=vxm.losses.BendEng('l2').loss(flow)

                total_loss=loss_reg_myo+loss_reg_rv+self.args.weight*loss_bend

                epoch_total_loss.append(total_loss.item())

                loss_list.append(loss_reg_myo.item())
                loss_list.append(loss_reg_rv.item())
                loss_list.append(self.args.weight*loss_bend.item())
                epoch_loss.append(loss_list)
                # backpropagate and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                self.eval_writer.add_scalar("loss/reg",loss_reg_myo.item(),global_step)
                self.eval_writer.add_scalar("loss/bend",loss_bend.item(),global_step)
                self.eval_writer.add_scalar("loss/total",total_loss.item(),global_step)
                global_step+=1
                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

            # print epoch info
            epoch_info = 'Epoch %d/%d' % (epoch + 1, self.args.epochs)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
            loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
            print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

            if epoch % self.args.save_freq == 0:
                self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de], dim=1, keepdim=True), global_step)
                self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2], dim=1, keepdim=True), global_step)
                self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0], dim=1, keepdim=True), global_step)

                self.model.save(os.path.join(self.args.model_dir, '%04d.pt' % epoch))
                self.model.eval()
                self.validate_net()
                self.model.train()
                self.save_img_labV2("../outputs/tmp",warp_img[Modality.t2],warp_roi_lab_myo[Modality.t2],'warp_mv','t2')
                self.save_img_labV2("../outputs/tmp",img[Modality.t2],roi_lab_myo[Modality.t2],'mv','t2')
                self.save_img_labV2("../outputs/tmp",img[Modality.de],roi_lab_myo[Modality.de],'fix','de')

        # final model save
        self.model.save(os.path.join(self.args.model_dir, '%04d.pt' % self.args.epochs))

    def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
        c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=5, length=1)
        t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=5, length=1)
        de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=5, length=1)
        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
        roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
        return  img,lab,roi_lab1,roi_lab2

from jrs_networks.jrs_losses import GaussianNGF
from voxelmorph.torch.losses import NCC
class Experiment_NFG(Experiment):
    def train_eval_net(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)


        # crit_reg=GaussianNGF(sigma=2)
        crit_reg=NCC(win=(6,6)).loss
        global_step=0
        self.model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []
            step_start_time = time.time()
            for img_c0,img_t2,img_de,lab_c0,lab_t2,lab_de in self.train_loader:
                #load data
                img, lab, roi_lab_myo, roi_lab_rv= self.create_torch_tensor(img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2)
                warp_img={}
                warp_lab={}
                warp_roi_lab_myo={}
                warp_roi_lab_rv={}
                # train
                # save model checkpoint
                flow = self.model(img[Modality.de], img[Modality.t2])
                # y_pred = self.model(inputs)
                warp_img[Modality.t2]=self.model.warp(img[Modality.t2],flow)
                warp_lab[Modality.t2]=self.model.warp(lab[Modality.t2],flow)
                warp_roi_lab_myo[Modality.t2]=self.model.warp(roi_lab_myo[Modality.t2],flow)
                warp_roi_lab_rv[Modality.t2]=self.model.warp(roi_lab_rv[Modality.t2],flow)

                # calculate total loss
                loss_list = []

                # loss_reg=vxm.losses.Dice().loss(warp_roi_lab_myo[Modality.t2],roi_lab_myo[Modality.de])
                loss_reg=crit_reg(warp_img[Modality.t2],img[Modality.de])
                loss_bend=self.args.weight*vxm.losses.BendEng('l2').loss(flow)
                loss=loss_reg+loss_bend

                loss_list.append(loss_reg)
                loss_list.append(loss_bend)
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())


                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step+=1
                # get compute time
                epoch_step_time.append(time.time() - step_start_time)
                self.eval_writer.add_scalar("loss/reg",loss_reg.item(),global_step)
                self.eval_writer.add_scalar("loss/bend",loss_bend.item(),global_step)
                self.eval_writer.add_scalar("loss/total",loss.item(),global_step)
                self.eval_writer.add_images("train/imgs/de", img[Modality.de], global_step)
                self.eval_writer.add_images("train/imgs/t2", img[Modality.t2], global_step)
                self.eval_writer.add_images("train/imgs/c0", img[Modality.c0], global_step)
                self.eval_writer.add_images("train/labs/de", torch.sum(lab[Modality.de],dim=1,keepdim=True), 0)
                self.eval_writer.add_images("train/labs/t2", torch.sum(lab[Modality.t2],dim=1,keepdim=True), 0)
                self.eval_writer.add_images("train/labs/c0", torch.sum(lab[Modality.c0],dim=1,keepdim=True), 0)


            # print epoch info
            epoch_info = 'Epoch %d/%d' % (epoch + 1, self.args.epochs)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
            loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
            print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

            if epoch % self.args.save_freq == 0:
                self.model.save(os.path.join(self.args.model_dir, '%04d.pt' % epoch))
                self.model.eval()
                self.validate_net()
                self.model.train()
                self.save_img_labV2("../outputs/tmp",warp_img[Modality.t2],warp_roi_lab_myo[Modality.t2],'warp_mv','t2')
                self.save_img_labV2("../outputs/tmp",img[Modality.t2],roi_lab_myo[Modality.t2],'mv','t2')
                self.save_img_labV2("../outputs/tmp",img[Modality.de],roi_lab_myo[Modality.de],'fix','de')

        # final model save
        self.model.save(os.path.join(self.args.model_dir, '%04d.pt' % self.args.epochs))




