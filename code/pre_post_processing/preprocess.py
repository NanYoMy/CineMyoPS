import shutil

from baseclass.medicalimage import MyoPS20MultiModalityImage,DataInfo,MyoPSLabelIndex
import SimpleITK as sitk
from tools.np_sitk_tools import get_bounding_box_by_idsV2,crop_by_bbox,get_bounding_box_by_ids
from tools.dir import sort_glob
from simpleitkop.itkdatawriter import sitk_write_image
import numpy as np
import os
from dataloader.util import SkimageOP_Base
class MyoPSPreProcess():
    def __init__(self, args, img_dir, lab_dir, aug=False):
        self.c0_imgs=sort_glob(f'{img_dir}/*C0*.nii.gz')
        self.t2_imgs=sort_glob(f'{img_dir}/*T2*.nii.gz')
        self.lge_imgs=sort_glob(f'{img_dir}/*DE*.nii.gz')
        self.labs=sort_glob(f"{lab_dir}/*.nii.gz")
        n_val = int(len(self.c0_imgs) * args.val_percent)
        n_train = len(self.c0_imgs) - n_val
        self.op=SkimageOP_Base()
        self.aug=aug
        self.args=args
        if aug == False:
            self.c0_imgs = self.c0_imgs[:n_train]
            self.t2_imgs = self.t2_imgs[:n_train]
            self.lge_imgs = self.lge_imgs[:n_train]
            self.labs = self.labs[:n_train]
        else:
            self.c0_imgs = self.c0_imgs[n_train:]
            self.t2_imgs = self.t2_imgs[n_train:]
            self.lge_imgs = self.lge_imgs[n_train:]
            self.labs = self.labs[n_train:]

    def reindex_for_myo_scar_edema_ZHANGZHEN(self,img,to_nn=True):
        arr=sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        if to_nn:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo.value, MyoPSLabelIndex.myo_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema.value,MyoPSLabelIndex.edema_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar.value,MyoPSLabelIndex.scar_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p.value, MyoPSLabelIndex.lv_p_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.rv.value, MyoPSLabelIndex.rv_nn.value, 0)
        else:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo_nn.value, MyoPSLabelIndex.myo.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema_nn.value, MyoPSLabelIndex.edema.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar_nn.value, MyoPSLabelIndex.scar.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p_nn.value,  MyoPSLabelIndex.lv_p.value, 0)
            new_array = new_array + np.where(arr ==  MyoPSLabelIndex.rv_nn.value,  MyoPSLabelIndex.rv.value, 0)

        new_img=sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    def random_affine(self,img,mask):
        img_array=sitk.GetArrayFromImage(img).astype("float")
        mask_array=sitk.GetArrayFromImage(mask).astype("float")
        img_array,mask_array=self.op.random_affine(img_array,mask_array)
        new_img=sitk.GetImageFromArray(img_array)
        new_mask=sitk.GetImageFromArray(mask_array)
        new_img.CopyInformation(img)
        new_mask.CopyInformation(mask)
        return new_img,new_mask

    def crop_by_lab_3D(self, ids, out_put_dir):

        for img_c0,img_t2,img_lge,lab in zip(self.c0_imgs,self.t2_imgs,self.lge_imgs,self.labs):

            medical=MyoPS20MultiModalityImage(img_c0,img_t2,img_lge,lab)
            lab=medical.get_data(DataInfo.lab)
            lab_array=sitk.GetArrayViewFromImage(lab)
            bbox=get_bounding_box_by_ids(lab_array,15,ids)
            crop_lab = crop_by_bbox(lab, bbox)
            crop_lab=self.reindex_for_myo_scar_edema_ZHANGZHEN(crop_lab)
            crop_c0=crop_by_bbox(medical.get_data(DataInfo.C0_img),bbox)
            crop_t2=crop_by_bbox(medical.get_data(DataInfo.T2_img),bbox)
            crop_lge=crop_by_bbox(medical.get_data(DataInfo.LGE_img),bbox)
            sub_out_out_dir=f"{out_put_dir}/{(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0]).replace('_gd','')}"
            for i in range(crop_lab.GetSize()[-1]):

                C0=sitk.GetArrayFromImage(crop_c0[:,:,i])
                lab=sitk.GetArrayFromImage(crop_lab[:,:,i])
                T2=sitk.GetArrayFromImage(crop_t2[:, :, i])
                LGE=sitk.GetArrayFromImage(crop_lge[:, :, i])
                C0,T2,LGE,lab=self.op.normalize_multiseq(C0, T2, LGE, lab,(self.args.image_size,self.args.image_size))


                if self.aug==True:
                    C0,  C0_lab, LGE,  LGE_lab, T2, T2_lab=self.op.aug_multiseq(C0, LGE, T2, lab)
                    # C0_lab=lab
                    # T2_lab=lab
                    # LGE_lab=lab
                else:
                    C0_lab=lab
                    T2_lab=lab
                    LGE_lab=lab

                sitk_write_image(C0, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.C0_path)).split('.')[0],i))
                sitk_write_image(np.round(C0_lab).astype(np.int16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','C0_gd'),i))

                sitk_write_image(T2, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.T2_path)).split('.')[0],i))
                sitk_write_image(np.round(T2_lab).astype(np.int16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','T2_gd'),i))

                sitk_write_image(LGE, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.LGE_path)).split('.')[0],i))
                sitk_write_image(np.round(LGE_lab).astype(np.int16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','DE_gd'),i))

from tools.np_sitk_tools import  sitkRespacing,reindex_label_by_dict

class Coordinate():
    def __init__(self,start,stop):
        self.start=start
        self.stop=stop

class RJPreProcess_un_aligned():
    def __init__(self, args, datadir,aug=False):


        self.datadir=datadir

        self.op=SkimageOP_Base()
        self.aug=aug
        self.args=args

    def reindex_for_myo_scar_edema_ZHANGZHEN(self,img,to_nn=True):
        arr=sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        if to_nn:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo.value, MyoPSLabelIndex.myo_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema.value,MyoPSLabelIndex.edema_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar.value,MyoPSLabelIndex.scar_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p.value, MyoPSLabelIndex.lv_p_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.rv.value, MyoPSLabelIndex.rv_nn.value, 0)
        else:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo_nn.value, MyoPSLabelIndex.myo.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema_nn.value, MyoPSLabelIndex.edema.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar_nn.value, MyoPSLabelIndex.scar.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p_nn.value,  MyoPSLabelIndex.lv_p.value, 0)
            new_array = new_array + np.where(arr ==  MyoPSLabelIndex.rv_nn.value,  MyoPSLabelIndex.rv.value, 0)

        new_img=sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    def random_affine(self,img,mask):
        img_array=sitk.GetArrayFromImage(img).astype("float")
        mask_array=sitk.GetArrayFromImage(mask).astype("float")
        img_array,mask_array=self.op.random_affine(img_array,mask_array)
        new_img=sitk.GetImageFromArray(img_array)
        new_mask=sitk.GetImageFromArray(mask_array)
        new_img.CopyInformation(img)
        new_mask.CopyInformation(mask)
        return new_img,new_mask


    def crop_by_lab_3D(self, p_imgs, p_labs, ids, out_put_dir,normalize=True):

        for p_img,p_lab in zip(p_imgs , p_labs):
            print(f"{p_img} {p_lab}")
            lab=sitk.ReadImage(p_lab)
            img=sitk.ReadImage(p_img)
            lab_array=sitk.GetArrayViewFromImage(lab)
            bbox=get_bounding_box_by_ids(lab_array,15,ids)
            crop_lab = crop_by_bbox(lab, bbox)
            crop_img = crop_by_bbox(img,bbox)
            # crop_lab=self.reindex_for_myo_scar_edema_ZHANGZHEN(crop_lab)

            sub_out_dir=f"{out_put_dir}/{(os.path.basename(p_img).split('_')[0])}_{(os.path.basename(p_img).split('_')[1])}"
            # crop
            for i in range(crop_lab.GetSize()[-1]):
                slice_img=sitk.GetArrayFromImage(crop_img[:,:,i])
                slice_lab=sitk.GetArrayFromImage(crop_lab[:,:,i])

                if normalize==True:
                    mysize = (self.args.image_size, self.args.image_size,)
                    slice_img, slice_lab = self.op.normalize_image(slice_img.astype("float"), slice_lab.astype("float"), size=mysize,clip=True)  # 对于结构的配准与分割可以考虑clip
                sitk_write_image(slice_img, dir=sub_out_dir, name="%s_%d"%(os.path.basename(p_img).split('.')[0],i))
                sitk_write_image(np.round(slice_lab).astype(np.int16),dir=sub_out_dir,name="%s_%d"%(os.path.basename(p_lab).split('.')[0],i))

    def mergebox(self,box1,box2,box3):
        res=[]
        for i in range(3):
            start=np.min(np.array([box1[i].start,box2[i].start,box3[i].start]))
            stop=np.max(np.array([box1[i].stop,box2[i].stop,box3[i].stop]))
            res.append(Coordinate(start,stop))
        return res

    def reduce_z(self, box, c0, t2, de, condition=[200, 500, 600]):
        slices=[]
        for i in range(box[0].start,box[0].stop+1):
            set_c0=np.unique(c0[i,:,:])
            set_t2=np.unique(t2[i,:,:])
            set_de=np.unique(de[i, :, :])
            res=True
            for s in [set_c0, set_t2, set_de]:
                for c in condition:
                    if not c in s:
                        res=False

            if not 1220 in set_t2:
                res=False
            if not 2221 in set_de:
                res=False

            if res==True:
                slices.append(i)

        box[0].start=slices[0]
        box[0].stop=slices[-1]
        return box





    def checkoutput(self,array,path):
        labs=np.unique(array)
        ret=True
        if not (200 in labs and 500 in labs and 600 in labs):
            ret=False

        if path.find("_c0_"):
            pass
        elif path.find("_t2_"):
            if not 1220 in labs:
                ret=False
        elif path.find("_de_"):
            if not 2221 in labs:
                ret=False
        return ret

    def generate_one_modality(self, img, lab, bbox, out_put_dir, crop_dir, p_img, p_lab, normalize):
        crop_lab = crop_by_bbox(lab, bbox)
        crop_img = crop_by_bbox(img, bbox)

        # crop_lab=self.reindex_for_myo_scar_edema_ZHANGZHEN(crop_lab)

        sitk_write_image(crop_img, parameter_img=None, dir=crop_dir, name=get_name_wo_suffix(p_img))
        sitk_write_image(crop_lab, parameter_img=None, dir=crop_dir, name=get_name_wo_suffix(p_lab))

        sub_out_dir = f"{out_put_dir}/{(os.path.basename(p_img).split('_')[0])}_{(os.path.basename(p_img).split('_')[1]).zfill(2)}"
        # crop




        j=0
        for i in range(crop_lab.GetSize()[-1]):
            slice_img = sitk.GetArrayFromImage(crop_img[:, :, i])
            slice_lab = sitk.GetArrayFromImage(crop_lab[:, :, i])

            if normalize == True:
                mysize = (self.args.image_size, self.args.image_size,)
                slice_img, slice_lab = self.op.normalize_image(slice_img.astype("float"), slice_lab.astype("float"),
                                                               size=mysize, clip=True)  # 对于结构的配准与分割可以考虑clip

            if not self.checkoutput(crop_lab, p_lab):
                print(f"error slice : {p_img} {i}")
                continue

            sitk_write_image(slice_img, dir=sub_out_dir, name="%s_%d" % (os.path.basename(p_img).split('.')[0], i))
            sitk_write_image(np.round(slice_lab).astype(np.int16), dir=sub_out_dir, name="%s_%d" % (os.path.basename(p_lab).split('.')[0], i))
            j=j+1

    def xy_resample(self, img, lab):
        print(img.GetSpacing())
        n_lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])

        n_img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
        return n_img,n_lab


    def _merge(self,labA,labB):
        labA_arr=sitk.GetArrayFromImage(labA)
        labB_arr=sitk.GetArrayFromImage(labB)
        mask=np.where(labB_arr>0,0,1)
        new_arr=labA_arr*mask+labB_arr
        new_img=sitk.GetImageFromArray(new_arr.astype(np.uint16))

        new_img.CopyInformation(labA)
        return  new_img

    def crop_ms_by_lab_3D(self, ids, out_put_dir,crop_dir,normalize=True):

        self.c0_imgs=sort_glob(f'{self.datadir}/align_subject*_C0.nii.gz')
        self.c0_labs=sort_glob(f"{self.datadir}/align_subject*C0_manual.nii.gz")

        self.t2_imgs=sort_glob(f'{self.datadir}/align_subject*_T2.nii.gz')
        self.t2_labs=sort_glob(f"{self.datadir}/align_subject*T2_manual.nii.gz")
        self.t2_edema_labs=sort_glob(f"{self.datadir}/align_subject*T2_manual_edema.nii.gz")

        self.de_imgs=sort_glob(f'{self.datadir}/align_subject*_DE.nii.gz')
        self.de_labs=sort_glob(f"{self.datadir}/align_subject*DE_manual.nii.gz")
        self.de_scar_labs=sort_glob(f"{self.datadir}/align_subject*DE_manual_scar.nii.gz")

        for p_c0_img,p_c0_lab,p_t2_img,p_t2_lab,p_de_img,p_de_lab,p_edema_lab,p_scar_lab in zip(self.c0_imgs, self.c0_labs, self.t2_imgs, self.t2_labs, self.de_imgs, self.de_labs,self.t2_edema_labs,self.de_scar_labs):
            # if p_c0_img.find("1006")>0:
            print(f"processing {p_c0_img}")

            c0_lab=sitk.ReadImage(p_c0_lab)
            t2_lab=sitk.ReadImage(p_t2_lab)
            de_lab=sitk.ReadImage(p_de_lab)
            scar_lab=sitk.ReadImage(p_scar_lab)
            scar_lab=reindex_label_by_dict(scar_lab,{2221:[200]})
            de_lab=self._merge(de_lab,scar_lab)
            edema_lab=sitk.ReadImage(p_edema_lab)
            edema_lab=reindex_label_by_dict(edema_lab,{1220:[200]})
            t2_lab=self._merge(t2_lab,edema_lab)

            c0_img=sitk.ReadImage(p_c0_img)
            t2_img=sitk.ReadImage(p_t2_img)
            de_img=sitk.ReadImage(p_de_img)

            #cause, the spacing of multi-modality images are different.
            c0_img,c0_lab=self.xy_resample(c0_img, c0_lab)
            t2_img,t2_lab=self.xy_resample(t2_img, t2_lab)
            de_img,de_lab=self.xy_resample(de_img, de_lab)

            c0_lab_array=sitk.GetArrayViewFromImage(c0_lab)
            t2_lab_array=sitk.GetArrayViewFromImage(t2_lab)
            de_lab_array=sitk.GetArrayViewFromImage(de_lab)


            c0_bbox1=get_bounding_box_by_idsV2(c0_lab_array,[0,15,15],ids=ids)
            t2_bbox2=get_bounding_box_by_idsV2(t2_lab_array,[0,15,15],ids=ids)
            de_bbox3=get_bounding_box_by_idsV2(de_lab_array,[0,15,15],ids=ids)

            # 在common space中进行裁剪
            bbox=self.mergebox(c0_bbox1,t2_bbox2,de_bbox3)
            bbox=self.reduce_z(bbox,c0_lab_array,t2_lab_array,de_lab_array,[200,500,600])
            self.generate_one_modality(c0_img,c0_lab,bbox,out_put_dir,crop_dir,p_c0_img,p_c0_lab,normalize)
            self.generate_one_modality(t2_img,t2_lab,bbox,out_put_dir,crop_dir,p_t2_img,p_t2_lab,normalize)
            self.generate_one_modality(de_img,de_lab,bbox,out_put_dir,crop_dir,p_de_img,p_de_lab,normalize)

    # mscmr
    def split_train_and_test_like_myops(self):
        ori_dir=self.datadir
        self.datadir=(f"{self.datadir}_rerank")
        mk_or_cleardir(self.datadir)
        names=list(range(1,46))
        test_ids=[45,20,18,14,10,8,5,43,37,42,1,31,23,40,39,21,30,26,36,12]

        [names.remove(t) for t in test_ids]
        train_ids =names

        rerank_train_idx = [12, 1, 9, 2, 17, 24, 23, 7, 20, 11, 5, 16, 25, 21, 13, 4, 10, 22, 19, 14, 8, 18, 3, 6, 15]
        #one more rerank
        rerank_train_id=[]
        for i in range(len(rerank_train_idx)):
            rerank_train_id.append(train_ids[rerank_train_idx[i]-1])


        total=[]
        total.extend(rerank_train_id)
        total.extend(test_ids)
        for i,idx in enumerate(total):
            files=sort_glob(f"{ori_dir}/subject_{idx}_*")
            for f in files:
                f_name=os.path.basename(f)
                f_terms=f_name.split('_')
                f_terms[1]=str(i+1)
                f_name="_".join(f_terms)
                shutil.copy(f,f"{self.datadir}/{f_name}")



    def process(self, roi_ids, out_put_dir,crop_dir):
        # self.split_train_and_test_like_myops()
        self.datadir=(f"{self.datadir}")
        self.crop_ms_by_lab_3D(roi_ids,out_put_dir,crop_dir,normalize=False)
        # self.crop_by_lab_3D(self.c0_imgs,self.c0_labs,)
        # self.crop_by_lab_3D(self.t2_imgs,self.t2_labs,roi_ids,out_put_dir)
        # self.crop_by_lab_3D(self.de_imgs,self.de_labs,roi_ids,out_put_dir)
    def process_old(self, roi_ids, out_put_dir,crop_dir):
        self.split_train_and_test_like_myops()
        self.crop_ms_by_lab_3D(roi_ids,out_put_dir,crop_dir,normalize=False)
        # self.crop_by_lab_3D(self.c0_imgs,self.c0_labs,)
        # self.crop_by_lab_3D(self.t2_imgs,self.t2_labs,roi_ids,out_put_dir)
        # self.crop_by_lab_3D(self.de_imgs,self.de_labs,roi_ids,out_put_dir)


class RJPreProcess_aligned():
    def __init__(self, args, datadir,aug=False):


        self.datadir=datadir

        self.op=SkimageOP_Base()
        self.aug=aug
        self.args=args

    def reindex_for_myo_scar_edema_ZHANGZHEN(self,img,to_nn=True):
        arr=sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        if to_nn:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo.value, MyoPSLabelIndex.myo_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema.value,MyoPSLabelIndex.edema_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar.value,MyoPSLabelIndex.scar_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p.value, MyoPSLabelIndex.lv_p_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.rv.value, MyoPSLabelIndex.rv_nn.value, 0)
        else:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo_nn.value, MyoPSLabelIndex.myo.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema_nn.value, MyoPSLabelIndex.edema.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar_nn.value, MyoPSLabelIndex.scar.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p_nn.value,  MyoPSLabelIndex.lv_p.value, 0)
            new_array = new_array + np.where(arr ==  MyoPSLabelIndex.rv_nn.value,  MyoPSLabelIndex.rv.value, 0)

        new_img=sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    def random_affine(self,img,mask):
        img_array=sitk.GetArrayFromImage(img).astype("float")
        mask_array=sitk.GetArrayFromImage(mask).astype("float")
        img_array,mask_array=self.op.random_affine(img_array,mask_array)
        new_img=sitk.GetImageFromArray(img_array)
        new_mask=sitk.GetImageFromArray(mask_array)
        new_img.CopyInformation(img)
        new_mask.CopyInformation(mask)
        return new_img,new_mask


    def crop_by_lab_3D(self, p_imgs, p_labs, ids, out_put_dir,normalize=True):

        for p_img,p_lab in zip(p_imgs , p_labs):
            print(f"{p_img} {p_lab}")
            lab=sitk.ReadImage(p_lab)
            img=sitk.ReadImage(p_img)
            lab_array=sitk.GetArrayViewFromImage(lab)
            bbox=get_bounding_box_by_ids(lab_array,15,ids)
            crop_lab = crop_by_bbox(lab, bbox)
            crop_img = crop_by_bbox(img,bbox)
            # crop_lab=self.reindex_for_myo_scar_edema_ZHANGZHEN(crop_lab)

            sub_out_dir=f"{out_put_dir}/{(os.path.basename(p_img).split('_')[0])}_{(os.path.basename(p_img).split('_')[1])}"
            # crop
            for i in range(crop_lab.GetSize()[-1]):
                slice_img=sitk.GetArrayFromImage(crop_img[:,:,i])
                slice_lab=sitk.GetArrayFromImage(crop_lab[:,:,i])

                if normalize==True:
                    mysize = (self.args.image_size, self.args.image_size,)
                    slice_img, slice_lab = self.op.normalize_image(slice_img.astype("float"), slice_lab.astype("float"), size=mysize,clip=True)  # 对于结构的配准与分割可以考虑clip
                sitk_write_image(slice_img, dir=sub_out_dir, name="%s_%d"%(os.path.basename(p_img).split('.')[0],i))
                sitk_write_image(np.round(slice_lab).astype(np.int16),dir=sub_out_dir,name="%s_%d"%(os.path.basename(p_lab).split('.')[0],i))

    def mergebox(self,box1,box2,box3):
        res=[]
        for i in range(3):
            start=np.min(np.array([box1[i].start,box2[i].start,box3[i].start]))
            stop=np.max(np.array([box1[i].stop,box2[i].stop,box3[i].stop]))
            res.append(Coordinate(start,stop))
        return res

    def reduce_z(self, box, c0, t2, de, condition=[200, 500, 600]):
        slices=[]
        for i in range(box[0].start,box[0].stop+1):
            set_c0=np.unique(c0[i,:,:])
            set_t2=np.unique(t2[i,:,:])
            set_de=np.unique(de[i, :, :])
            res=True
            for s in [set_c0, set_t2, set_de]:
                for c in condition:
                    if not c in s:
                        res=False

            if not 1220 in set_t2:
                res=False
            if not 2221 in set_de:
                res=False

            if res==True:
                slices.append(i)

        box[0].start=slices[0]
        box[0].stop=slices[-1]
        return box





    def checkoutput(self,array,path):
        labs=np.unique(array)
        ret=True
        if not (200 in labs and 500 in labs and 600 in labs):
            ret=False

        if path.find("_c0_"):
            pass
        elif path.find("_t2_"):
            if not 1220 in labs:
                ret=False
        elif path.find("_de_"):
            if not 2221 in labs:
                ret=False
        return ret

    def generate_one_modality(self, img, lab, bbox, out_put_dir, crop_dir, p_img, p_lab, normalize):
        crop_lab = crop_by_bbox(lab, bbox)
        crop_img = crop_by_bbox(img, bbox)

        # crop_lab=self.reindex_for_myo_scar_edema_ZHANGZHEN(crop_lab)

        sitk_write_image(crop_img, parameter_img=None, dir=crop_dir, name=get_name_wo_suffix(p_img))
        sitk_write_image(crop_lab, parameter_img=None, dir=crop_dir, name=get_name_wo_suffix(p_lab))

        sub_out_dir = f"{out_put_dir}/{(os.path.basename(p_img).split('_')[0])}_{(os.path.basename(p_img).split('_')[1]).zfill(2)}"
        # crop




        j=0
        for i in range(crop_lab.GetSize()[-1]):
            slice_img = sitk.GetArrayFromImage(crop_img[:, :, i])
            slice_lab = sitk.GetArrayFromImage(crop_lab[:, :, i])

            if normalize == True:
                mysize = (self.args.image_size, self.args.image_size,)
                slice_img, slice_lab = self.op.normalize_image(slice_img.astype("float"), slice_lab.astype("float"),
                                                               size=mysize, clip=True)  # 对于结构的配准与分割可以考虑clip

            if not self.checkoutput(crop_lab, p_lab):
                print(f"error slice : {p_img} {i}")
                continue

            sitk_write_image(slice_img, dir=sub_out_dir, name="%s_%d" % (os.path.basename(p_img).split('.')[0], i))
            sitk_write_image(np.round(slice_lab).astype(np.int16), dir=sub_out_dir, name="%s_%d" % (os.path.basename(p_lab).split('.')[0], i))
            j=j+1

    def xy_resample(self, img, lab):
        print(img.GetSpacing())
        n_lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])

        n_img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
        return n_img,n_lab


    def crop_ms_by_lab_3D(self, ids, out_put_dir,crop_dir,normalize=True):

        self.c0_imgs=sort_glob(f'{self.datadir}/*img_c0*.nii*')
        self.t2_imgs=sort_glob(f'{self.datadir}/*img_t2*.nii*')
        self.de_imgs=sort_glob(f'{self.datadir}/*img_de*.nii*')
        self.c0_labs=sort_glob(f"{self.datadir}/*ana*c0*.nii*")
        self.t2_labs=sort_glob(f"{self.datadir}/*ana*t2_*.nii*")
        self.de_labs=sort_glob(f"{self.datadir}/*ana*de_*.nii*")

        for p_c0_img,p_c0_lab,p_t2_img,p_t2_lab,p_de_img,p_de_lab in zip(self.c0_imgs,self.c0_labs,self.t2_imgs,self.t2_labs,self.de_imgs,self.de_labs):
            # if p_c0_img.find("1006")>0:
            print(f"processing {p_c0_img}")

            c0_lab=sitk.ReadImage(p_c0_lab)
            t2_lab=sitk.ReadImage(p_t2_lab)
            de_lab=sitk.ReadImage(p_de_lab)
            c0_img=sitk.ReadImage(p_c0_img)
            t2_img=sitk.ReadImage(p_t2_img)
            de_img=sitk.ReadImage(p_de_img)

            #cause, the spacing of multi-modality images are different.
            c0_img,c0_lab=self.xy_resample(c0_img, c0_lab)
            t2_img,t2_lab=self.xy_resample(t2_img, t2_lab)
            de_img,de_lab=self.xy_resample(de_img, de_lab)

            c0_lab_array=sitk.GetArrayViewFromImage(c0_lab)
            t2_lab_array=sitk.GetArrayViewFromImage(t2_lab)
            de_lab_array=sitk.GetArrayViewFromImage(de_lab)


            c0_bbox1=get_bounding_box_by_idsV2(c0_lab_array,[0,15,15],ids=ids)
            t2_bbox2=get_bounding_box_by_idsV2(t2_lab_array,[0,15,15],ids=ids)
            de_bbox3=get_bounding_box_by_idsV2(de_lab_array,[0,15,15],ids=ids)

            # 在common space中进行裁剪
            bbox=self.mergebox(c0_bbox1,t2_bbox2,de_bbox3)
            bbox=self.reduce_z(bbox,c0_lab_array,t2_lab_array,de_lab_array,[200,500,600])
            self.generate_one_modality(c0_img,c0_lab,bbox,out_put_dir,crop_dir,p_c0_img,p_c0_lab,normalize)
            self.generate_one_modality(t2_img,t2_lab,bbox,out_put_dir,crop_dir,p_t2_img,p_t2_lab,normalize)
            self.generate_one_modality(de_img,de_lab,bbox,out_put_dir,crop_dir,p_de_img,p_de_lab,normalize)

    # mscmr
    def split_train_and_test_like_myops(self):
        ori_dir=self.datadir
        self.datadir=(f"{self.datadir}_rerank")
        mk_or_cleardir(self.datadir)
        names=list(range(1,46))
        test_ids=[45,20,18,14,10,8,5,43,37,42,1,31,23,40,39,21,30,26,36,12]

        [names.remove(t) for t in test_ids]
        train_ids =names

        rerank_train_idx = [12, 1, 9, 2, 17, 24, 23, 7, 20, 11, 5, 16, 25, 21, 13, 4, 10, 22, 19, 14, 8, 18, 3, 6, 15]
        #one more rerank
        rerank_train_id=[]
        for i in range(len(rerank_train_idx)):
            rerank_train_id.append(train_ids[rerank_train_idx[i]-1])


        total=[]
        total.extend(rerank_train_id)
        total.extend(test_ids)
        for i,idx in enumerate(total):
            files=sort_glob(f"{ori_dir}/subject_{idx}_*")
            for f in files:
                f_name=os.path.basename(f)
                f_terms=f_name.split('_')
                f_terms[1]=str(i+1)
                f_name="_".join(f_terms)
                shutil.copy(f,f"{self.datadir}/{f_name}")



    def process(self, roi_ids, out_put_dir,crop_dir):
        # self.split_train_and_test_like_myops()
        self.datadir=(f"{self.datadir}")
        self.crop_ms_by_lab_3D(roi_ids,out_put_dir,crop_dir,normalize=False)
        # self.crop_by_lab_3D(self.c0_imgs,self.c0_labs,)
        # self.crop_by_lab_3D(self.t2_imgs,self.t2_labs,roi_ids,out_put_dir)
        # self.crop_by_lab_3D(self.de_imgs,self.de_labs,roi_ids,out_put_dir)
    def process_old(self, roi_ids, out_put_dir,crop_dir):
        self.split_train_and_test_like_myops()
        self.crop_ms_by_lab_3D(roi_ids,out_put_dir,crop_dir,normalize=False)
        # self.crop_by_lab_3D(self.c0_imgs,self.c0_labs,)
        # self.crop_by_lab_3D(self.t2_imgs,self.t2_labs,roi_ids,out_put_dir)
        # self.crop_by_lab_3D(self.de_imgs,self.de_labs,roi_ids,out_put_dir)




class MyoPSPreProcess_aff_algined(RJPreProcess_un_aligned):
    def __init__(self, args, datadir,aug=False):
        self.c0_imgs=sort_glob(f'{datadir}/*subject[0-9]*_C0.nii.gz')
        self.t2_imgs=sort_glob(f'{datadir}/*subject[0-9]*_T2.nii.gz')
        self.de_imgs=sort_glob(f'{datadir}/*subject[0-9]*_DE.nii.gz')
        self.c0_labs=sort_glob(f"{datadir}/*ana_c0.nii.gz")
        self.t2_labs=sort_glob(f"{datadir}/*ana_patho_t2_*.nii.gz")
        self.de_labs=sort_glob(f"{datadir}/*ana_patho_de_*.nii.gz")
        n_val = int(len(self.c0_imgs) * args.val_percent)
        n_train = len(self.c0_imgs) - n_val
        self.op=SkimageOP_Base()
        self.aug=aug
        self.args=args

class MyoPSPreProcess_algined(MyoPSPreProcess_aff_algined):
    def __init__(self, args, datadir,aug=False):
        self.c0_imgs=sort_glob(f'{datadir}/*subject[0-9]*_C0.nii.gz')
        self.t2_imgs=sort_glob(f'{datadir}/*subject[0-9]*_T2.nii.gz')
        self.de_imgs=sort_glob(f'{datadir}/*subject[0-9]*_DE.nii.gz')
        self.c0_labs=sort_glob(f"{datadir}/*subject[0-9]*_C0_manual.nii.gz")
        self.t2_labs=sort_glob(f"{datadir}/*subject[0-9]*_T2_manual.nii.gz")
        self.de_labs=sort_glob(f"{datadir}/*subject[0-9]*_DE_manual.nii.gz")
        n_val = int(len(self.c0_imgs) * args.val_percent)
        n_train = len(self.c0_imgs) - n_val
        self.op=SkimageOP_Base()
        self.aug=aug
        self.args=args


from tools.dir import mk_or_cleardir,get_name_wo_suffix
class MyoPS20PreProcess():

    def __init__(self, args,datadir):
        self.c0_imgs=sort_glob(f'{datadir}/*_C0*.nii.gz')
        self.t2_imgs=sort_glob(f'{datadir}/*_T2*.nii.gz')
        self.de_imgs=sort_glob(f'{datadir}/*_DE*.nii.gz')
        self.c0_labs=sort_glob(f"{datadir}/*_gd*.nii.gz")
        self.t2_labs=sort_glob(f"{datadir}/*_gd*.nii.gz")
        self.de_labs=sort_glob(f"{datadir}/*_gd*.nii.gz")
        self.labs=sort_glob(f"{datadir}/*_gd*.nii.gz")

        # n_val = int(len(self.c0_imgs) * args.val_percent)
        # n_train = len(self.c0_imgs) - n_val
        n_train=20
        self.op=SkimageOP_Base()
        self.args=args
        # self.aug=False
        # if aug == False:
        #     self.c0_imgs = self.c0_imgs[:n_train]
        #     self.t2_imgs = self.t2_imgs[:n_train]
        #     self.de_imgs = self.de_imgs[:n_train]
        #     self.labs = self.labs[:n_train]
        # else:
        #     self.c0_imgs = self.c0_imgs[n_train:]
        #     self.t2_imgs = self.t2_imgs[n_train:]
        #     self.de_imgs = self.de_imgs[n_train:]
        #     self.labs = self.labs[n_train:]

    def reindex_for_myo_scar_edema(self,img,to_nn=True):
        arr=sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        if to_nn:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo.value, MyoPSLabelIndex.myo_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema.value,MyoPSLabelIndex.edema_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar.value,MyoPSLabelIndex.scar_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p.value, MyoPSLabelIndex.lv_p_nn.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.rv.value, MyoPSLabelIndex.rv_nn.value, 0)
        else:
            new_array = new_array + np.where(arr == MyoPSLabelIndex.myo_nn.value, MyoPSLabelIndex.myo.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.edema_nn.value, MyoPSLabelIndex.edema.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.scar_nn.value, MyoPSLabelIndex.scar.value, 0)
            new_array = new_array + np.where(arr == MyoPSLabelIndex.lv_p_nn.value,  MyoPSLabelIndex.lv_p.value, 0)
            new_array = new_array + np.where(arr ==  MyoPSLabelIndex.rv_nn.value,  MyoPSLabelIndex.rv.value, 0)

        new_img=sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    def random_affine(self,img,mask):
        img_array=sitk.GetArrayFromImage(img).astype("float")
        mask_array=sitk.GetArrayFromImage(mask).astype("float")
        img_array,mask_array=self.op.random_affine(img_array,mask_array)
        new_img=sitk.GetImageFromArray(img_array)
        new_mask=sitk.GetImageFromArray(mask_array)
        new_img.CopyInformation(img)
        new_mask.CopyInformation(mask)
        return new_img,new_mask

    def process(self, ids,out_put_dir,crop_dir=None,need_reindex=True):
        mk_or_cleardir(out_put_dir)
        for p_img_c0,p_img_t2,p_img_lge,p_lab in zip(self.c0_imgs,self.t2_imgs,self.de_imgs,self.labs):

            medical=MyoPS20MultiModalityImage(p_img_c0,p_img_t2,p_img_lge,p_lab)
            lab=medical.get_data(DataInfo.lab)
            lab_array=sitk.GetArrayViewFromImage(lab)
            bbox=get_bounding_box_by_ids(lab_array,15,ids)
            crop_lab = crop_by_bbox(lab, bbox)

            if need_reindex==True:
                crop_lab=self.reindex_for_myo_scar_edema(crop_lab)

            crop_c0=crop_by_bbox(medical.get_data(DataInfo.C0_img),bbox)
            crop_t2=crop_by_bbox(medical.get_data(DataInfo.T2_img),bbox)
            crop_lge=crop_by_bbox(medical.get_data(DataInfo.LGE_img),bbox)

            if crop_dir is not None:
                sitk_write_image(crop_c0,dir=crop_dir,name=get_name_wo_suffix(p_img_c0))
                sitk_write_image(crop_t2,dir=crop_dir,name=get_name_wo_suffix(p_img_t2))
                sitk_write_image(crop_lge,dir=crop_dir,name=get_name_wo_suffix(p_img_lge))
                sitk_write_image(sitk.Cast(crop_lab,sitk.sitkUInt16),dir=crop_dir,name=get_name_wo_suffix(p_lab))

            sub_out_out_dir=f"{out_put_dir}/{(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0]).replace('_gd','')}"
            for i in range(crop_lab.GetSize()[-1]):
                C0=(crop_c0[:,:,i])
                T2=(crop_t2[:, :, i])
                LGE=(crop_lge[:, :, i])
                lab=(crop_lab[:,:,i])
                # C0,T2,LGE,lab=self.op.normalize_multiseq(C0, T2, LGE, lab,(self.args.image_size,self.args.image_size))

                # if self.aug==True:
                #     C0,  C0_lab, LGE,  LGE_lab, T2, T2_lab=self.op.aug_multiseq(C0, LGE, T2, lab)
                # else:
                C0_lab=lab
                T2_lab=lab
                LGE_lab=lab

                sitk_write_image(C0, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.C0_path)).split('.')[0],i))
                sitk_write_image(sitk.Cast(C0_lab,sitk.sitkUInt16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','C0_gd'),i))

                sitk_write_image(T2, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.T2_path)).split('.')[0],i))
                sitk_write_image(sitk.Cast(T2_lab,sitk.sitkUInt16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','T2_gd'),i))

                sitk_write_image(LGE, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.LGE_path)).split('.')[0],i))
                sitk_write_image(sitk.Cast(LGE_lab,sitk.sitkUInt16),dir=sub_out_out_dir,name="%s_%d"%(os.path.basename(medical.get_data(DataInfo.lab_path)).split('.')[0].replace('gd','DE_gd'),i))


    def process_test(self, out_put_dir):
        mk_or_cleardir(out_put_dir)
        for p_img_c0,p_img_t2,p_img_lge in zip(self.c0_imgs,self.t2_imgs,self.de_imgs):

            # medical=MyoPS20MultiModalityImage(p_img_c0,p_img_t2,p_img_lge,p_lab)
            # lab=medical.get_data(DataInfo.lab)
            # lab_array=sitk.GetArrayViewFromImage(lab)
            # bbox=get_bounding_box_by_ids(lab_array,15,ids)
            # crop_lab = crop_by_bbox(lab, bbox)
            # crop_lab=self.reindex_for_myo_scar_edema(crop_lab)
            crop_c0=sitk.ReadImage(p_img_c0)
            crop_t2=sitk.ReadImage(p_img_t2)
            crop_lge=sitk.ReadImage(p_img_lge)

            sub_out_out_dir=f"{out_put_dir}/{(os.path.basename(p_img_c0).split('.')[0]).replace('_gd','')}"

            for i in range(crop_t2.GetSize()[-1]):
                C0=(crop_c0[:,:,i])
                T2=(crop_t2[:, :, i])
                LGE=(crop_lge[:, :, i])
                # lab=(crop_lab[:,:,i])
                # C0,T2,LGE,lab=self.op.normalize_multiseq(C0, T2, LGE, lab,(self.args.image_size,self.args.image_size))

                # if self.aug==True:
                #     C0,  C0_lab, LGE,  LGE_lab, T2, T2_lab=self.op.aug_multiseq(C0, LGE, T2, lab)
                # else:
                # C0_lab=lab
                # T2_lab=lab
                # LGE_lab=lab

                sitk_write_image(C0, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(p_img_c0).split('.')[0],i))

                sitk_write_image(T2, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(p_img_t2).split('.')[0],i))

                sitk_write_image(LGE, dir=sub_out_out_dir, name="%s_%d"%(os.path.basename(p_img_lge).split('.')[0],i))
