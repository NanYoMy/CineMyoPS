import logging
from os import path as osp
import cv2
import numpy
import numpy as np
import torch
from medpy.metric import dc,hd95
from torch import optim
from torch.utils.data import DataLoader
import SimpleITK as sitk
from baseclass.medicalimage import Modality
from experiment.baseexperiment import BaseMyoPSExperiment
import os
from tools.dir import sort_glob, mk_or_cleardir,sort_time_glob,mkdir_if_not_exist
from tools.set_random_seed import worker_init_fn
from dataloader.util import SkimageOP_MyoPS20
from tools.np_sitk_tools import reindex_label_array_by_dict
from tools.tps_painter import save_image_with_tps_points
from tools.nii_lab_to_png import save_img
from tools.np_sitk_tools import clipseScaleSitkImage
from skimage.util.compare import compare_images
from skimage.exposure import rescale_intensity
import  itertools
import seaborn
from tools.excel import write_array
from dataloader.myops20dataset import MyoPS20DataSet as DataSetLoader
from jrs_networks.jrs_tps_seg_net import JRS3TpsSegNet as RSNet

class Experiment_ASN_MyoPS(BaseMyoPSExperiment):
    '''
    The experiment for myops segmentation, consistent loss.
    '''
    def __init__(self,args):
        super().__init__(args)
        self.args=args



        if args.net == "tps":
            pass
            # from jrs_networks.jrs_3m_tps_seg_net import JRS3MROITpsSegNet as RSNet
        else:
            logging.error("unimplmented type")
            exit(-200)

        self.model = RSNet(args)

        r1 = self.args.span_range_height
        r2 = self.args.span_range_width
        control_points = np.array(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.args.grid_width - 1)),
        )))
        self.base_control_points=np.zeros_like(control_points)
        self.base_control_points[:,0]=control_points[:,1]
        self.base_control_points[:,1]=control_points[:,0]

        if args.load:
            model_path = sort_time_glob(args.checkpoint_dir + f"/*.pth")[args.ckpt]
            # model_path = sort_glob(args.checkpoint_dir + "/*.pth")[-1]
            self.model = self.model.load(model_path, self.device)
            logging.info(f'Model loaded from {args.load} : {model_path}')
        self.model.to(device= self.device)

        self.train_loader = DataLoader(DataSetLoader(args, type="train",augo=True, task='pathology',ret_path=False),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
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
        self.op = SkimageOP_MyoPS20()
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
        self.gen_valid()
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
        from nnunet.training.loss_functions.dice_loss import SoftDiceLoss,SoftConsistentDiceLoss
        from jrs_networks.jrs_losses import BinaryGaussianDice
        from nnunet.utilities.nd_softmax import softmax_helper
        # crit_reg=BinaryGaussianDice()
        # regcrit=SoftConsistentDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)
        # segcrit=SoftDiceLoss(apply_nonlin=softmax_helper,batch_dice= True, smooth= 1e-5, do_bg= False)

        regcrit=BinaryGaussianDice(sigma=1)
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
                img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv= self.create_torch_tensor(img_c0,  img_t2, img_de,lab_c0, lab_t2, lab_de)

                # train
                # c0_pred,t2_pred,de_pred,theta_c0,theta_t2= self.model(img[Modality.c0], img[Modality.t2],img[Modality.de])
                # c0_pred,de_pred,t2_pred,theta_c0,theta_t2= self.model(img[Modality.c0],img[Modality.de], img[Modality.t2])
                #loss

                warp_img = {}
                warp_roi_lab_myo = {}
                sts_loss = {}
                warp_roi_lab_lv = {}
                warp_roi_lab_rv={}
                # train
                c0_seg, t2_seg, lge_seg, theta_c0, theta_t2 = self.model(img[Modality.c0], img[Modality.t2],
                                                                         img[Modality.de])


                # loss
                warp_roi_lab_myo[Modality.c0] = self.model.warp(roi_lab_myo[Modality.c0], theta_c0)
                warp_roi_lab_myo[Modality.t2] = self.model.warp(roi_lab_myo[Modality.t2], theta_t2)
                warp_roi_lab_lv[Modality.c0] = self.model.warp(roi_lab_lv[Modality.c0], theta_c0)
                warp_roi_lab_lv[Modality.t2] = self.model.warp(roi_lab_lv[Modality.t2], theta_t2)

                warp_roi_lab_rv[Modality.c0] = self.model.warp(roi_lab_rv[Modality.c0], theta_c0)
                warp_roi_lab_rv[Modality.t2] = self.model.warp(roi_lab_rv[Modality.t2], theta_t2)

                warp_img[Modality.c0] = self.model.warp(img[Modality.c0], theta_c0)
                warp_img[Modality.t2] = self.model.warp(img[Modality.t2], theta_t2)

                loss_reg_myo = regcrit(roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.c0]) + regcrit(
                    roi_lab_myo[Modality.de], warp_roi_lab_myo[Modality.t2])
                loss_reg_lv = regcrit(roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.c0]) + regcrit(
                    roi_lab_lv[Modality.de], warp_roi_lab_lv[Modality.t2])
                loss_reg_rv = regcrit(roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.c0]) + regcrit(
                    roi_lab_rv[Modality.de], warp_roi_lab_rv[Modality.t2])
                # 分割目标是myo
                # loss_seg_lge = segcrit(lge_seg, roi_lab_myo[Modality.de])
                # loss_seg_c0 = segcrit(c0_seg, roi_lab_myo[Modality.c0])
                # loss_seg_t2 = segcrit(t2_seg, roi_lab_myo[Modality.t2])

                loss_seg_lge = segcrit(lge_seg, roi_lab_lv[Modality.de])
                loss_seg_c0 = segcrit(c0_seg, roi_lab_lv[Modality.c0])
                loss_seg_t2 = segcrit(t2_seg, roi_lab_lv[Modality.t2])

                loss_all = (2 *loss_seg_lge + loss_seg_c0 + loss_seg_t2) + self.args.weight * ( loss_reg_myo + loss_reg_lv+loss_reg_rv)
                # loss_all = (2 *loss_seg_lge + loss_seg_c0 + loss_seg_t2) + self.args.weight * ( loss_reg_myo + loss_reg_lv)

                # warp_c0_pred = self.model.warp(c0_pred, theta_c0)
                # warp_t2_pred = self.model.warp(t2_pred, theta_t2)
                #
                # loss_reg_c0 = regcrit(warp_c0_pred, de_pred)
                # loss_reg_t2 = regcrit(warp_t2_pred, de_pred)
                #
                #
                # loss_seg_c0 = segcrit(c0_pred, roi_lab_lv[Modality.c0])
                # loss_seg_t2 = segcrit(t2_pred, roi_lab_lv[Modality.t2])
                # loss_seg_de = segcrit(de_pred, roi_lab_lv[Modality.de])
                #
                # loss_all=(loss_seg_de+loss_seg_c0+loss_seg_t2)+self.args.weight*(loss_reg_c0+loss_reg_t2)
                #statistic
                sts_loss={}
                sts_loss["train/loss_reg_myo"]=(loss_reg_myo.item())
                sts_loss["train/loss_reg_lv"]=(loss_reg_lv.item())
                sts_loss["train/loss_reg_rv"]=(loss_reg_rv.item())
                sts_loss["train/loss_seg_c0"]=(loss_seg_c0.item())
                sts_loss["train/loss_seg_t2"]=(loss_seg_t2.item())
                sts_loss["train/loss_seg_lge"]=(loss_seg_lge.item())
                sts_loss["train/loss_total"]=(loss_all.item())

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

                    self.eval_writer.add_images("train/labs/de", (roi_lab_lv[Modality.de]), global_step)
                    self.eval_writer.add_images("train/labs/t2", (roi_lab_lv[Modality.t2]), global_step)
                    self.eval_writer.add_images("train/labs/c0", (roi_lab_lv[Modality.c0]), global_step)

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
        theta_c0_s=[]
        theta_t2_s=[]
        with torch.no_grad():
            for img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de, c0_path,t2_path,de_path in self.val_loader:
                img, lab, roi_lab_myo, roi_lab_lv, roi_lab_rv = self.create_torch_tensor(img_c0,img_t2, img_de,  lab_c0, lab_t2, lab_de)
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
                print(de_path)
                print(theta_c0.cpu().numpy()[0,:,:]-self.base_control_points)
                print(theta_t2.cpu().numpy()[0,:,:]-self.base_control_points)


                # c0_seg=torch.argmax(c0_seg,dim=1,keepdim=True)
                # t2_seg=torch.argmax(t2_seg,dim=1,keepdim=True)
                de_pred=torch.argmax(de_pred,dim=1,keepdim=True)
                warp_seg[Modality.c0]=torch.argmax(warp_seg[Modality.c0],dim=1,keepdim=True)
                warp_seg[Modality.t2]=torch.argmax(warp_seg[Modality.t2],dim=1,keepdim=True)

                warp_img[Modality.c0]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['c0'], size), axis=0)).cuda(), theta_c0)
                warp_img[Modality.t2]=self.model.warp(self.op.convert_array_2_torch(np.expand_dims(self.op.resize(ori_array['t2'], size), axis=0)).cuda(), theta_t2)


                subdir=os.path.basename(os.path.dirname(c0_path[0]))
                output=os.path.join(self.args.gen_dir,subdir)
                mkdir_if_not_exist(output)

                self.save_diff_img(ori_array['c0'],ori_array['de'],output+"_diff",self._rename(c0_path, 'diff_img')+".png")
                self.save_diff_img(ori_array['t2'],ori_array['de'],output+"_diff",self._rename(t2_path, 'diff_img')+".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(),ori_array['de'],output+"_diff",self._rename(c0_path, 'diff_warp_img')+".png")
                self.save_diff_img(warp_img[Modality.t2].cpu().numpy(),ori_array['de'],output+"_diff",self._rename(t2_path, 'diff_warp_img')+".png")




                self.save_img_with_tps(ori_array['c0'],theta_c0,output+"_tps", self._rename(c0_path, 'tps_img')+".png")
                self.save_img_with_tps(ori_array['t2'],theta_t2,output+"_tps", self._rename(t2_path, 'tps_img')+".png")
                self.save_img(ori_array['de'],output+"_tps",self._rename(de_path, 'tps_img')+".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'],output , self._rename(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'],output, self._rename(c0_path, 'assn_lab'),is_label=True)

                self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'],output, self._rename(t2_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'],output, self._rename(t2_path, 'assn_lab'),is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'],output, self._rename(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'],output, self._rename(de_path, 'assn_lab'),is_label=True)

                # self.save_torch_img_lab(self.args.gen_dir, self.op.convert_img_2_torch(ori_array['de']),de_seg, de_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.c0],  warp_seg[Modality.c0], c0_path, 'assn','pred')
                # self.save_torch_img_lab(self.args.gen_dir, warp_img[Modality.t2],  warp_seg[Modality.t2], t2_path, 'assn','pred')
        subjets=sort_glob(f"{self.args.gen_dir}/*")
        ds={"C0":[],"T2":[],"DE":[]}
        hds={"C0":[],"T2":[],"DE":[]}
        for dir in subjets:
            subdir=os.path.basename(dir)
            for modality in ['C0','T2','DE']:

                gds=sort_glob(f"{self.args.dataset_dir}/valid5/{subdir}/*{modality}_gd*nii.gz")
                preds=sort_glob(f"{dir}/*{modality}_assn_lab*nii.gz")
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
                gds_arr=np.squeeze(reindex_label_array_by_dict(gds_arr,{1:[1,2,3,5]}))
                preds_arr=np.squeeze(preds_arr)
                ds[modality].append(dc(gds_arr,preds_arr))
                gd3d=sort_glob(f"{self.args.dataset_dir}/valid5_croped/*{subdir}_gd*nii.gz")
                para=sitk.ReadImage(gd3d[0])
                hds[modality].append(hd95(gds_arr,preds_arr,(para.GetSpacing()[-1],para.GetSpacing()[1],para.GetSpacing()[0])))

        for k in ds.keys():
            if (len(ds[k]))>0:
                # print(ds[k])
                write_array(self.args.res_excel, f'myops_asn_{k}_ds', ds[k])
                logging.info(f'subject level evaluation:  DS {k}: {np.mean(ds[k])}')
                logging.info(f'subject level evaluation:  DS {k}: {np.std(ds[k])}')
                # print(hds[k])
                write_array(self.args.res_excel, f'myops_asn_{k}_hd95', hds[k])
                logging.info(f'subject level evaluation:  HD {k}: {np.mean(hds[k])}')
                logging.info(f'subject level evaluation:  HD {k}: {np.std(hds[k])}')


    def test(self):
        print("staget test ASSN output")
        mk_or_cleardir(self.args.gen_dir)
        size=[self.args.image_size,self.args.image_size]
        theta_c0_s=[]
        theta_t2_s=[]

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
                print(c0_path)
                print(t2_path)
                print(de_path)

                print(theta_c0.cpu().numpy()[0,:,:]-self.base_control_points)
                print(theta_t2.cpu().numpy()[0,:,:]-self.base_control_points)
                theta_c0_s.append(theta_c0.cpu().numpy()[0,:,:]-self.base_control_points)
                theta_t2_s.append(theta_t2.cpu().numpy()[0,:,:]-self.base_control_points)



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
                                   self._rename(c0_path, 'diff_img') + ".png")
                self.save_diff_img(ori_array['t2'], ori_array['de'], output + "_diff",
                                   self._rename(t2_path, 'diff_img') + ".png")
                self.save_diff_img(warp_img[Modality.c0].cpu().numpy(), ori_array['de'], output + "_diff",
                                   self._rename(c0_path, 'diff_warp_img') + ".png")
                self.save_diff_img(warp_img[Modality.t2].cpu().numpy(), ori_array['de'], output + "_diff",
                                   self._rename(t2_path, 'diff_warp_img') + ".png")

                self.save_img(ori_array['de'], output + "_ori", self._rename(de_path, 'ori_img') + ".png")

                self.save_img(ori_array['c0'],output + "_ori",self._rename(c0_path, 'ori_img') + ".png")
                self.save_img(warp_img[Modality.c0].cpu().numpy(),output + "_warp",self._rename(c0_path, 'warp_img') + ".png")

                self.save_img(ori_array['t2'],output + "_ori",self._rename(t2_path, 'ori_img') + ".png")
                self.save_img(warp_img[Modality.t2].cpu().numpy(),output + "_warp",self._rename(t2_path, 'warp_img') + ".png")

                self.save_img_with_tps(ori_array['c0'],theta_c0,output+"_tps", self._rename(c0_path, 'tps_img')+".png")
                self.save_img_with_tps(ori_array['t2'],theta_t2,output+"_tps", self._rename(t2_path, 'tps_img')+".png")
                self.save_img(ori_array['de'],output+"_tps",self._rename(de_path, 'tps_img')+".png")

                self.save_tensor_with_parameter(warp_img[Modality.c0], para['c0'], output, self._rename(c0_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.c0], para['c0'], output, self._rename(c0_path, 'assn_lab'), is_label=True)

                self.save_tensor_with_parameter(warp_img[Modality.t2], para['t2'], output, self._rename(t2_path, 'assn_img'))
                self.save_tensor_with_parameter(warp_seg[Modality.t2], para['t2'], output, self._rename(t2_path, 'assn_lab'),is_label=True)

                self.save_tensor_with_parameter(self.op.convert_array_2_torch(ori_array['de']), para['de'], output, self._rename(de_path, 'assn_img'))
                self.save_tensor_with_parameter(de_pred, para['de'], output, self._rename(de_path, 'assn_lab'), is_label=True)

        self.plotbox(np.array(theta_t2_s)[:, :, 0],self.args.gen_dir,'t2_0.png')
        self.plotbox(np.array(theta_t2_s)[:, :, 1],self.args.gen_dir,'t2_1.png')
        self.plotbox(np.array(theta_c0_s)[:, :, 0],self.args.gen_dir,'c0_0.png')
        self.plotbox(np.array(theta_c0_s)[:, :, 1],self.args.gen_dir,'c0_1.png')

        self.plotbox(np.sqrt(np.array(theta_t2_s)[:, :, 0]**2+np.array(theta_t2_s)[:, :, 1]**2),self.args.gen_dir,'t2.png')
        self.plotbox(np.sqrt(np.array(theta_c0_s)[:, :, 0]**2+np.array(theta_c0_s)[:, :, 1]**2),self.args.gen_dir,'c0.png')


    def create_torch_tensor(self, img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de):
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

        c0_roi_reg_mask_3 = lab_c0.narrow(dim=1, start=5, length=1)
        t2_roi_reg_mask_3 = lab_t2.narrow(dim=1, start=5, length=1)
        de_roi_reg_mask_3 = lab_de.narrow(dim=1, start=5, length=1)

        lab_c0 = lab_c0.narrow(dim=1, start=-1, length=1)
        lab_t2 = lab_t2.narrow(dim=1, start=-1, length=1)
        lab_de = lab_de.narrow(dim=1, start=-1, length=1)

        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
        roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
        roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab1,roi_lab2,roi_lab3

    def create_test_torch_tensor(self, img_c0, img_t2, img_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        img = {Modality.c0: img_c0, Modality.t2: img_t2, Modality.de: img_de}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return img


    def _rename(self,name,tag):
        term = os.path.basename((name[0])).split("_")
        name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{tag}_{term[4]}'
        return name
