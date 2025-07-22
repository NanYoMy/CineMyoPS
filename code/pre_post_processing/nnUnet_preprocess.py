import os

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunet.paths import nnUNet_raw_data,nnUNet_cropped_data

from dataloader.util import SkimageOP_Base
from tools.np_sitk_tools import  get_bounding_box_by_ids, crop_by_bbox
from tools.dir import sort_glob, mk_or_cleardir
from tools.np_sitk_tools import get_mask_bounding_box,reindex_label_array ,reindex_label_by_dict,reindex_label_array_by_dict
from tools.np_sitk_tools import binarize_img
from tools.dir import mkdir_if_not_exist
from tools.np_sitk_tools import binarize_numpy_array
from tools.dir import mkcleardir,mk_or_cleardir
import skimage
import nibabel as nib
class nnUNetPrepare():
    def __init__(self,args):
        self.spacing = (0.75, 0.75)
        self.foldername = "Task%03.0d_%s_%s" % (args.task_id, args.task_name,args.target)
        self.taskname= "%s_%s" % ( args.task_name,args.target)
        # self.spacing=[0.0025,0.0025]
        self.out_base = os.path.join(nnUNet_raw_data, self.foldername)
        mk_or_cleardir(self.out_base)
        self.crop_out_base=os.path.join(nnUNet_cropped_data,self.foldername)
        mk_or_cleardir(self.crop_out_base)
        self.imagestr = os.path.join(self.out_base, "imagesTr")
        self.imagests = os.path.join(self.out_base, "imagesTs")
        self.imagesvalid = os.path.join(self.out_base, "imagesValid")
        self.labelstr = os.path.join(self.out_base, "labelsTr")
        self.labelsts = os.path.join(self.out_base, "labelsTs")
        self.labelsvalid = os.path.join(self.out_base, "labelsValid")
        self.train_patient_names = []
        self.test_patient_names = []
        self.res = []
    def reindex_label(self,img,ids,to_nn=True):
        arr=sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        for k in ids.keys():
            for i in ids[k]:
                new_array = new_array + np.where(arr== i,k , 0)
        new_img=sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    # def write_array_for_nnunet2D(self, img, out_dir, name, id="0000", is_label=False,para=None):
    #
    #     if is_label==False:
    #         img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
    #     else:
    #         img_itk = sitk.GetImageFromArray(img.astype(np.uint8)[None])
    #
    #
    #
    #     if para is not None:
    #         # img_itk.CopyInformation(para)
    #         # img_itk.SetSpacing(list(para.GetSpacing())[::-1] + [999])
    #         img_itk.SetSpacing(list(self.spacing)[::-1] + [999])
    #     else:
    #         img_itk.SetSpacing(list(self.spacing)[::-1] + [999])
    #
    #
    #     if is_label==False:
    #         sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}_{id}.nii.gz"))
    #     else:
    #         sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}.nii.gz"))


    def write_array_for_nnunet(self, img, out_dir, name, id="0000", is_label=False,para=None):

        # if is_label==False:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
        # else:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.uint8)[None])

        if is_label==False:
            img_itk = sitk.GetImageFromArray(img.astype(np.float32))
        else:
            img_itk = sitk.GetImageFromArray(img.astype(np.uint8))

        if para is not None:
            img_itk.CopyInformation(para)
            # img_itk.SetSpacing(list(self.spacing)[::-1] + [999])
            # img_itk.SetSpacing(list(para.GetSpacing())[:2]+ [999])
        else:
            img_itk.SetSpacing(list(self.spacing)[::-1] + [999])


        if is_label==False:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}_{id}.nii.gz"))
        else:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}.nii.gz"))

    def write_array_for_nnunet2D(self, img, out_dir, name, id="0000", is_label=False,para=None):

        if is_label==False:
            img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
        else:
            img_itk = sitk.GetImageFromArray(img.astype(np.uint8)[None])

        # if is_label==False:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.float32))
        # else:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.uint8))

        if para is not None:
            # img_itk.CopyInformation(para)
            # img_itk.SetSpacing(list(self.spacing)[::-1] + [999])
            img_itk.SetSpacing(list(para.GetSpacing())[:2]+ [999])
        else:
            img_itk.SetSpacing(list(self.spacing)[::-1] + [999])


        if is_label==False:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}_{id}.nii.gz"))
        else:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}.nii.gz"))

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"{self.args.modality}",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names)>0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

# json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class MyoPSPreCropProcess_nnUNet(nnUNetPrepare):
    def __init__(self, args, aug=False):
        super(MyoPSPreCropProcess_nnUNet, self).__init__(args)
        self.imgs = sort_glob(f'{args.data_dir}/train25/*{args.modality}*.nii.gz')
        self.labs = sort_glob(f"{args.data_dir}/train25_myops_gd/*.nii.gz")
        self.op = SkimageOP_Base()
        self.aug = aug
        self.args=args
        



    def crop_by_lab_3D(self, ids_for_crop, ids_for_seg):

        for p_img,p_lab in zip(self.imgs,self.labs):

            lab=sitk.ReadImage(p_lab)
            lab_array=sitk.GetArrayViewFromImage(lab)

            img=sitk.ReadImage(p_img)

            bbox=get_bounding_box_by_ids(lab_array, 15, ids_for_crop)
            crop_lab = crop_by_bbox(lab, bbox)
            crop_lab=self.reindex_label(crop_lab, ids_for_seg)
            crop_img=crop_by_bbox(img,bbox)

            for i in range(crop_lab.GetSize()[-1]):
                img=sitk.GetArrayFromImage(crop_img[:,:,i])
                lab=sitk.GetArrayFromImage(crop_lab[:,:,i])
                img,lab=self.op.normalize_image(img, lab,(self.args.image_size,self.args.image_size))
                if self.aug==True:
                    img,  lab=self.op.aug_img(img, lab)
                    # C0_lab=lab
                else:
                    lab=lab
                case_name="%s_%d"%(os.path.basename(p_img).split('.')[0],i)
                self.train_patient_names.append(case_name)
                self.write_array_for_nnunet(img, self.imagestr, case_name)#img, dir=sub_out_out_dir, name=case_name)
                self.write_array_for_nnunet(np.round(lab).astype(np.int16), self.labelstr, case_name, is_label=True)

        self.write_json()

class MyoPSPreProcess_nnUNet(nnUNetPrepare):
    def __init__(self, args, aug=False):
        super(MyoPSPreProcess_nnUNet, self).__init__(args)
        self.imgs = sort_glob(f'{args.data_dir}/train25/*{args.modality}*.nii.gz')
        self.labs = sort_glob(f"{args.data_dir}/train25_myops_gd/*.nii.gz")

        self.test_img=sort_glob(f'{args.data_dir}/test20/*{args.modality}*.nii.gz')

        self.op = SkimageOP_Base()
        self.aug = aug
        self.args = args

    def extract_slice_from_volum(self,  ids_for_seg):
        mkdir_if_not_exist(self.labelstr)
        mkdir_if_not_exist(self.imagestr)
        mkdir_if_not_exist(self.imagests)
        for p_img, p_lab in zip(self.imgs, self.labs):

            lab = sitk.ReadImage(p_lab)

            img = sitk.ReadImage(p_img)
            lab = self.reindex_label(lab, ids_for_seg)
            para=sitk.ReadImage(p_img)
            for i in range(lab.GetSize()[-1]):
                img_slice = sitk.GetArrayFromImage(img[:, :, i])
                lab_slice = sitk.GetArrayFromImage(lab[:, :, i])
                # img, lab = self.op.normalize_image(img, lab, (self.args.image_size, self.args.image_size))
                # if self.aug == True:
                #     img, lab = self.op.aug_img(img, lab)
                # else:
                #     lab = lab

                case_name = "%s_%d" % (os.path.basename(p_img).split('.')[0], i)
                self.train_patient_names.append(case_name)
                self.write_2D_array_for_nnunet(img_slice, self.imagestr, case_name,para=para)  # img, dir=sub_out_out_dir, name=case_name)
                self.write_2D_array_for_nnunet(np.round(lab_slice).astype(np.int16), self.labelstr, case_name, is_label=True,para=para)

#test dataset
        for p_img in self.test_img:


            img = sitk.ReadImage(p_img)
            para=sitk.ReadImage(p_img)
            for i in range(img.GetSize()[-1]):
                img_slice = sitk.GetArrayFromImage(img[:, :, i])
                case_name = "%s_%d" % (os.path.basename(p_img).split('.')[0], i)
                self.test_patient_names.append(case_name)
                self.write_2D_array_for_nnunet(img_slice, self.imagests, case_name,para=para)  # img, dir=sub_out_out_dir, name=case_name)

        self.write_json()

    def write_2D_array_for_nnunet(self, img, out_dir, name, id="0000", is_label=False,para=None):

        if is_label==False:
            img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
        else:
            img_itk = sitk.GetImageFromArray(img.astype(np.uint8)[None])

        if para is not None:
            # img_itk.CopyInformation(para)
            # img_itk.SetSpacing(list(self.spacing)[::-1] + [999])
            img_itk.SetSpacing(list(para.GetSpacing())[:2]+ [999])
        else:
            img_itk.SetSpacing(list(self.spacing)[::-1] + [999])


        if is_label==False:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}_{id}.nii.gz"))
        else:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}.nii.gz"))



    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"{self.args.modality}",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2":"lv",
            "3":"rv"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names)>0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class MSCMR_AS_PreProcess_nnUNet(nnUNetPrepare):
    def __init__(self, args, aug=False):
        super(MSCMR_AS_PreProcess_nnUNet, self).__init__(args)
        self.args=args

        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    # def _getsample(self, subjects, modality):
    #     # here mask was predicted by jrs
    #     dict = {"img": [],  'lab': []}
    #
    #     for s in subjects:
    #         dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
    #         dict["lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
    #
    #     print(dict)
    #
    #     list = []
    #     for i, l, g in zip(dict["img"], dict["asn_lab"], dict['gt_lab']):
    #         list.append({"img": i, "asn_lab": l, 'gt_lab': g})
    #
    #     return list

    def prepare(self):

        self.imgs  = sort_glob(f'{self.args.data_dir}/*/*img_{self.args.modality}*')
        if self.args.modality=='c0':
            self.labs  = sort_glob(f'{self.args.data_dir}/*/*ana_{self.args.modality}*')
        else:
            self.labs  = sort_glob(f'{self.args.data_dir}/*/*ana_patho_{self.args.modality}*')


        for img,lab in zip(self.imgs,self.labs):

            terms=os.path.basename(img).split('.')[0].split('_')
            # case_name = "%s_%s_%s_%s" % (terms[0], terms[1], terms[2], terms[3], terms[4])
            case_name = f"{terms[0]}_{terms[1]}_{terms[2]}_{terms[3]}_{terms[4]}"
            if int(terms[1])>25:
                self.test_patient_names.append(case_name)
                self.save_one_img_lab(img,lab,case_name,self.imagests,self.labelsts)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_img_lab(img, lab,case_name,self.imagestr,self.labelstr)


        self.write_json()




    def save_one_img_lab(self, img,lab,case_name,output_img_dir,output_lab_dir):
        lab_sitk=sitk.ReadImage(lab)
        lab_array = sitk.GetArrayFromImage(lab_sitk)
        lab_array = binarize_numpy_array(lab_array, [1220,200,500,2221])
        img_array=sitk.GetArrayFromImage(sitk.ReadImage(img))
        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        self.write_array_for_nnunet2D(img_array, output_img_dir, case_name, f'0000', False, para=lab_sitk)
        self.write_array_for_nnunet2D(lab_array, output_lab_dir, case_name, is_label=True, para=lab_sitk)

    # def save_one_train_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
    #     lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
    #     lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
    #     lab_de_lab = sitk.ReadImage([de['gt_lab']])
    #     lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
    #     lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
    #     lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200, 500, 1220, 2221])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [200, 500, 1220, 2221])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [200, 500, 1220, 2221])
    #
    #     mask = (binary_myo_array + binary_edema_array + binary_scar_array)
    #     mask = np.where(mask > 0, 1, 0)
    #
    #     kernel = skimage.morphology.disk(5)
    #     mask[0, :, :] = skimage.morphology.dilation(mask[0, :, :], kernel)
    #
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [2221])
    #
    #     lab_array = np.zeros_like(binary_myo_array)
    #     lab_array[np.where(binary_myo_array == 1)] = 1
    #     lab_array[np.where(binary_edema_array == 1)] = 2
    #     lab_array[np.where(binary_scar_array == 1)] = 3
    #
    #     self.save_masked_img(c0, mask, case_name, 0, imagestr)
    #     self.save_masked_img(t2, mask, case_name, 1, imagestr)
    #     self.save_masked_img(de, mask, case_name, 2, imagestr)
    #
    #     # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
    #     self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
    #
    # def write_mask_img(self, pathes, mask, case_name, id, imagestr):
    #     img = sitk.ReadImage(pathes['img'])
    #     mask = sitk.GetImageFromArray(mask)
    #     mask = binarize_img(mask, [1, 2, 3, 5])
    #     masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)
    #
    #     self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')
    #
    # def _write(self, pathes, case_name, id, imagestr):
    #     img = sitk.ReadImage(pathes['img'])
    #     lab = sitk.ReadImage(pathes['lab'])
    #     mask = binarize_img(lab, [1, 2, 3, 5])
    #     masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)
    #
    #     self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')
    #
    # def save_masked_img(self, pathes, mask, case_name, id, outputdir):
    #     img = sitk.ReadImage(pathes['img'])
    #     masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)


    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"{self.args.modality}",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "lv",
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class MyoPSPreProcess_nnUNet_Multiseq(MyoPSPreCropProcess_nnUNet):
    def __init__(self,args,aug=False):
        super().__init__(args,aug)
        self.imgs_C0 = sort_glob(f'{args.data_dir}/train25/*C0*.nii.gz')
        self.imgs_T2 = sort_glob(f'{args.data_dir}/train25/*T2*.nii.gz')
        self.imgs_DE = sort_glob(f'{args.data_dir}/train25/*DE*.nii.gz')
        self.labs = sort_glob(f"{args.data_dir}/train25_myops_gd/*.nii.gz")

    def reindex_label(self, img, ids, to_nn=True):
        arr = sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        for k in ids.keys():
            for i in ids[k]:
                new_array = new_array + np.where(arr == i, k, 0)
        new_img = sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)

        # mask=arr>0
        # mask = sitk.GetImageFromArray(mask.astype(np.int16))
        # mask.CopyInformation(mask)

        return new_img

    def generate_LV_mask(self, img, ids={1: [200], 2: [1220], 3: [2221],4:[500]}):
        arr = sitk.GetArrayFromImage(img)
        new_array = np.zeros(arr.shape, np.uint16)
        for k in ids.keys():
            for i in ids[k]:
                new_array = new_array + np.where(arr == i, 1, 0)
        new_img = sitk.GetImageFromArray(new_array)
        new_img.CopyInformation(img)
        return new_img

    def crop_by_lab_3D(self, ids_for_crop, ids_for_seg, mask_img=False):

        for p_img_C0, p_img_T2,p_img_DE,p_lab in zip(self.imgs_C0,self.imgs_T2,self.imgs_DE, self.labs):

            lab = sitk.ReadImage(p_lab)
            lab_array = sitk.GetArrayViewFromImage(lab)
            bbox = get_bounding_box_by_ids(lab_array, 10, ids_for_crop)
            crop_lab = crop_by_bbox(lab, bbox)
            mask_volume=self.generate_LV_mask(crop_lab)
            crop_lab = self.reindex_label(crop_lab, ids_for_seg)


            for i in range(crop_lab.GetSize()[-1]):
                case_name = "%s_%d" % (os.path.basename(p_lab).split('.')[0].replace('gd', ''), i)
                self.train_patient_names.append(case_name)
                lab = sitk.GetArrayFromImage(crop_lab[:, :, i])
                mask = sitk.GetArrayFromImage(mask_volume[:, :, i])
                for index,p_img in enumerate([p_img_C0, p_img_T2, p_img_DE]):
                    img = sitk.ReadImage(p_img)
                    crop_img = crop_by_bbox(img, bbox)
                    img = sitk.GetArrayFromImage(crop_img[:, :, i])
                    if mask_img == True:
                        img = img * mask
                    img, lab = self.op.normalize_image(img, lab,(self.args.image_size,self.args.image_size))
                    if self.aug == True:
                        img, lab = self.op.aug_img(img, lab)

                    self.write_array_for_nnunet(img, self.imagestr, case_name, f'000{index}')  # img, dir=sub_out_out_dir, name=case_name)


                self.write_array_for_nnunet(np.round(lab).astype(np.int16), self.labelstr, case_name, is_label=True)


        self.write_json()

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"C0",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3": "scar"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class PSNTrainDataPrepare_MyoPS(nnUNetPrepare):

    def __init__(self,args):
        super(PSNTrainDataPrepare_MyoPS, self).__init__(args)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.imagesvalid)
        mk_or_cleardir(self.labelstr)
        mk_or_cleardir(self.labelsvalid)
        self.op = SkimageOP_Base()
        self.args=args


    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.upper()}_[0-9].nii.gz"))
            dict["lab"].extend(sort_glob(f"{s}/*{modality.upper()}_gd_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["img"],dict["lab"]):
            list.append({"img":i,"lab":l})

        return list


    def prepare(self):
        subjects= sort_glob(f'{self.args.data_dir_train}/*')+sort_glob(f'{self.args.data_dir_valid}/*')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        assert len(c0s) == len(des) and len(c0s) == len(t2s)
        self.train_patient_names=self.save_multi_modality_maskimage(c0s, t2s, des, self.imagestr, self.labelstr)

        # subjects= sort_glob(f'{self.args.data_dir_valid}/*')
        # c0s=self._getsample(subjects, "c0")
        # t2s=self._getsample(subjects, "t2")
        # des=self._getsample(subjects, "de")
        # self.test_patient_names=self.save_multi_modality_maskimage(c0s, t2s, des, self.imagesvalid, self.labelsvalid)
        self.write_json()
    def save_multi_modality_maskimage(self, c0s, t2s, des, imagestr, labelstr):
        train_patient_names=[]
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[5])
            train_patient_names.append(case_name)
            lab_img=sitk.ReadImage([c0['lab']])
            para=sitk.ReadImage([c0['img']])
            lab_array = sitk.GetArrayFromImage(lab_img)#c0,de,t2
            myo_lab_array=reindex_label_array_by_dict(lab_array,{1:[1],2:[2],3:[3]})
            self.write_array_for_nnunet(myo_lab_array, labelstr, case_name, is_label=True,para=para)
            mask=reindex_label_array_by_dict(lab_array,{1:[1,2,3,5]})

            # a dilation is required
            if self.args.dali==True:
                kernel = skimage.morphology.disk(5)
                mask[0, :, :] = skimage.morphology.dilation(mask[0, :, :], kernel)

            self.save_masked_img(c0, mask, case_name, 0, imagestr)
            self.save_masked_img(t2, mask, case_name, 1, imagestr)
            self.save_masked_img(de, mask, case_name, 2, imagestr)

        return train_patient_names

    def _write_img(self,pathes,mask,case_name,id,outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"C0",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3": "scar"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))
    def save_masked_img(self, pathes, mask, case_name, id, outputdir):
        img = sitk.ReadImage([pathes['img']])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)


class PSNTestDataAfterASNPrepare_MyoPS(nnUNetPrepare):
    def __init__(self,args):
        super(PSNTestDataAfterASNPrepare_MyoPS, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"lab": []}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.upper()}_assn_img_[0-9].nii.gz"))
            dict["lab"].extend(sort_glob(f"{s}/*{modality.upper()}_assn_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["img"],dict["lab"]):
            list.append({"img":i,"lab":l})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des, self.imagests, self.labelsts)

        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des, imagestr, labelstr):
        train_patient_names=[]
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])
            train_patient_names.append(case_name)
            labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['lab']))#c0,de,t2
            labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['lab']))#c0,de,t2
            labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['lab']))#c0,de,t2

            labC0 = binarize_numpy_array(labC0,[1,2,3,5])
            labT2 = binarize_numpy_array(labT2,[1,2,3,5])
            labDE = binarize_numpy_array(labDE,[1,2,3,5])



            if self.args.dali==True:
                mergelab = (labC0 + labT2 + labDE)
                mergelab = np.where(mergelab > 0, 1, 0)
                kernel = skimage.morphology.disk(5)
                mergelab[0, :, :] = skimage.morphology.dilation(mergelab[0, :, :], kernel)
            else:
                mergelab = np.floor((labC0 + labT2 + labDE) / 2)

            self.save_masked_img(c0, mergelab, case_name, 0, imagestr)
            self.save_masked_img(t2, mergelab, case_name, 1, imagestr)
            self.save_masked_img(de, mergelab, case_name, 2, imagestr)
            #merge label delet
        return train_patient_names

    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_masked_img(self, pathes, mask, case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

class PSNDataAfterASNPrepare_MSCMR(nnUNetPrepare):
    def __init__(self,args):
        super(PSNDataAfterASNPrepare_MSCMR, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_train_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def save_one_test_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        asn_labC0 = binarize_numpy_array(asn_labC0, [1, 2, 3, 5])
        asn_labT2 = binarize_numpy_array(asn_labT2, [1, 2, 3, 5])
        asn_labDE = binarize_numpy_array(asn_labDE, [1, 2, 3, 5])

        mask = (asn_labC0 + asn_labT2 + asn_labDE)
        mask = np.where(mask>0,1,0)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #

        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        lab_array[np.where(binary_myo_array == 1)] = 1
        lab_array[np.where(binary_edema_array == 1)] = 2
        lab_array[np.where(binary_scar_array == 1)] = 3

        self.save_masked_img(c0, mask, case_name, 0, imagestr)
        self.save_masked_img(t2, mask, case_name, 1, imagestr)
        self.save_masked_img(de, mask, case_name, 2, imagestr)

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)

    def save_one_train_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200,500,1220,2221])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [200,500,1220,2221])
        binary_scar_array = binarize_numpy_array(lab_de_array, [200,500,1220,2221])


        mask = (binary_myo_array + binary_edema_array + binary_scar_array)
        mask = np.where(mask>0,1,0)

        kernel = skimage.morphology.disk(5)
        mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)

        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        lab_array[np.where(binary_myo_array == 1)] = 1
        lab_array[np.where(binary_edema_array == 1)] = 2
        lab_array[np.where(binary_scar_array == 1)] = 3

        self.save_masked_img(c0, mask, case_name, 0, imagestr)
        self.save_masked_img(t2, mask, case_name, 1, imagestr)
        self.save_masked_img(de, mask, case_name, 2, imagestr)

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_masked_img(self, pathes, mask, case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)
    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"C0",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3": "scar"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class RJPSNPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(RJPSNPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def _check_parameter(self,a:sitk.Image,b:sitk.Image):
        assert  a.GetSize()==b.GetSize()
        assert  a.GetSpacing()==b.GetSpacing()
        assert a.GetOrigin()==b.GetOrigin()
        assert a.GetDirection()==b.GetDirection()

    def save_one_test_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        # self.save_img(c0,  case_name, 0, imagestr)


        #the predict ROI from ASN, it will be propagated to PSN as Attention for pathology segmentation.
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        self._check_parameter(sitk.ReadImage(c0['asn_lab']),sitk.ReadImage(t2['asn_lab']))
        self._check_parameter(sitk.ReadImage(de['asn_lab']),sitk.ReadImage(t2['asn_lab']))
        asn_labC0 = binarize_numpy_array(asn_labC0, [1, 2, 3, 5])
        asn_labT2 = binarize_numpy_array(asn_labT2, [1, 2, 3, 5])
        asn_labDE = binarize_numpy_array(asn_labDE, [1, 2, 3, 5])

        #prior
        prior = (asn_labC0 + asn_labT2 + asn_labDE)
        prior = np.where(prior>0,1,0)


        #i add prior and use it as attention for psn. do not perform data augmentation operation on prior channal
        #see nnuent/preprocessing/preprocessing.py  search KEYWORD scheme=="prior":
        if self.args.comspace=="de":
            self.save_prior(prior, de ,case_name, 0, imagestr)
        elif self.args.comspace=="t2":
            self.save_prior(prior, t2 ,case_name, 0, imagestr)
        else:
            exit(-912)

        self.save_img(t2,  case_name, 1, imagestr)
        self.save_img(de,  case_name, 2, imagestr)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #
        # the ground truth of pathology and predict segmentation mask
        #
        # lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        # lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2

        # binary_myo_array = binarize_numpy_array(lab_c0_array, [200],outindex=1)

        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220],outindex=2)
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221],outindex=4)

        # lab_array = binary_edema_array+binary_scar_array
        # i add prior here to avoid intensity-based data augmentation operation, it could be used as attention for psn segmentation
        lab_array = binary_edema_array+binary_scar_array+prior

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        if self.args.comspace=="de":
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
        elif self.args.comspace=="t2":
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_t2_lab)
        else:
            exit(-912)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_img(self, pathes,  case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

    def save_prior(self,prior,para,case_name,id,outputdir):
        img = sitk.ReadImage(para['img'])
        self.write_array_for_nnunet(prior, outputdir, case_name, f'000{id}',False,img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"prior",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3" : "myo_edema",
            "4": "scar",
            "5":"myo_scar",
            '6':"scar_edema",
            '7':"myo_scar_edema_"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class RJMyoPriorPSNPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(RJMyoPriorPSNPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def _check_parameter(self,a:sitk.Image,b:sitk.Image):
        assert  a.GetSize()==b.GetSize()
        assert  a.GetSpacing()==b.GetSpacing()
        assert a.GetOrigin()==b.GetOrigin()
        assert a.GetDirection()==b.GetDirection()

    def save_one_test_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        # self.save_img(c0,  case_name, 0, imagestr)


        #the predict ROI from ASN, it will be propagated to PSN as Attention for pathology segmentation.
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        self._check_parameter(sitk.ReadImage(c0['asn_lab']),sitk.ReadImage(t2['asn_lab']))
        self._check_parameter(sitk.ReadImage(de['asn_lab']),sitk.ReadImage(t2['asn_lab']))



        #i add prior and use it as attention for psn. do not perform data augmentation operation on prior channal
        #see nnuent/preprocessing/preprocessing.py  search KEYWORD scheme=="prior":
        if self.args.comspace=="de":
            prior=asn_labDE
            self.save_prior(prior, de ,case_name, 0, imagestr)
        elif self.args.comspace=="t2":
            # prior
            prior =  asn_labT2
            self.save_prior(prior, t2 ,case_name, 0, imagestr)
        else:
            exit(-912)

        self.save_img(t2,  case_name, 1, imagestr)
        self.save_img(de,  case_name, 2, imagestr)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #
        # the ground truth of pathology and predict segmentation mask
        #
        # lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        # lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2

        # binary_myo_array = binarize_numpy_array(lab_c0_array, [200],outindex=1)

        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220],outindex=2)
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221],outindex=4)

        # lab_array = binary_edema_array+binary_scar_array
        # i add prior here to avoid intensity-based data augmentation operation, it could be used as attention for psn segmentation
        prior = np.where(prior > 0.5, 1, 0)
        lab_array = binary_edema_array+binary_scar_array+prior

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        if self.args.comspace=="de":
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
        elif self.args.comspace=="t2":
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_t2_lab)
        else:
            exit(-912)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_img(self, pathes,  case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

    def save_prior(self,prior,para,case_name,id,outputdir):
        img = sitk.ReadImage(para['img'])
        self.write_array_for_nnunet(prior, outputdir, case_name, f'000{id}',False,img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"prior",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3" : "myo_edema",
            "4": "scar",
            "5":"myo_scar",
            '6':"scar_edema",
            '7':"myo_scar_edema_"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class NNUnetDataAfterASNPrepare_MSCMR(nnUNetPrepare):
    def __init__(self,args):
        super(NNUnetDataAfterASNPrepare_MSCMR, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>25:
                self.test_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def save_one_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        asn_labC0 = binarize_numpy_array(asn_labC0, [1, 2, 3, 5])
        asn_labT2 = binarize_numpy_array(asn_labT2, [1, 2, 3, 5])
        asn_labDE = binarize_numpy_array(asn_labDE, [1, 2, 3, 5])

        mask = (asn_labC0 + asn_labT2 + asn_labDE)
        mask = np.where(mask>0,1,0)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #

        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220,2221])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        if self.args.modality=="t2":
            self.save_masked_img(t2, mask, case_name, 0, imagestr)
            lab_array[np.where(binary_edema_array == 1)] = 1
        elif self.args.modality=='de':
            self.save_masked_img(de, mask, case_name, 0, imagestr)
            lab_array[np.where(binary_scar_array == 1)] = 1
        else:
            print("unsupport type")
            exit(-912)

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)

    # def save_one_train_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
    #     lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
    #     lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
    #     lab_de_lab = sitk.ReadImage([de['gt_lab']])
    #     lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
    #     lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
    #     lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200,500,1220,2221])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [200,500,1220,2221])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [200,500,1220,2221])
    #
    #
    #     mask = (binary_myo_array + binary_edema_array + binary_scar_array)
    #     mask = np.where(mask>0,1,0)
    #
    #     kernel = skimage.morphology.disk(5)
    #     mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
    #
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [2221])
    #
    #     lab_array = np.zeros_like(binary_myo_array)
    #     lab_array[np.where(binary_myo_array == 1)] = 1
    #     lab_array[np.where(binary_edema_array == 1)] = 2
    #     lab_array[np.where(binary_scar_array == 1)] = 3
    #
    #     self.save_masked_img(c0, mask, case_name, 0, imagestr)
    #     self.save_masked_img(t2, mask, case_name, 1, imagestr)
    #     self.save_masked_img(de, mask, case_name, 2, imagestr)
    #
    #     # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
    #     self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_masked_img(self, pathes, mask, case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)
    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": self.args.modality
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "scar" if self.args.modality=='de' else "edema",

        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class NNUnetMMDataAfterASNPrepare_MSCMR(nnUNetPrepare):
    def __init__(self,args):
        super(NNUnetMMDataAfterASNPrepare_MSCMR, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list




    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>25:
                self.test_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def save_one_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        asn_labC0 = binarize_numpy_array(asn_labC0, [1, 2, 3, 5])
        asn_labT2 = binarize_numpy_array(asn_labT2, [1, 2, 3, 5])
        asn_labDE = binarize_numpy_array(asn_labDE, [1, 2, 3, 5])

        mask = (asn_labC0 + asn_labT2 + asn_labDE)
        mask = np.where(mask>0,1,0)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #

        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        lab_array[np.where(binary_myo_array == 1)] = 1
        lab_array[np.where(binary_edema_array == 1)] = 2
        lab_array[np.where(binary_scar_array == 1)] = 3

        self.save_masked_img(c0, mask, case_name, 0, imagestr)
        self.save_masked_img(t2, mask, case_name, 1, imagestr)
        self.save_masked_img(de, mask, case_name, 2, imagestr)

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_masked_img(self, pathes, mask, case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img) * (mask).astype(np.float)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"C0",
            "1": f"T2",
            "2": f"DE"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "myo",
            "2": "edema",
            "3": "scar"
        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class RJSingleModalitynnuentPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(RJSingleModalitynnuentPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list


    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def save_one_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        asn_labC0 = binarize_numpy_array(asn_labC0, [1, 2, 3, 5])
        asn_labT2 = binarize_numpy_array(asn_labT2, [1, 2, 3, 5])
        asn_labDE = binarize_numpy_array(asn_labDE, [1, 2, 3, 5])

        mask = (asn_labC0 + asn_labT2 + asn_labDE)
        mask = np.where(mask>0,1,0)

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #

        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220,2221])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        if self.args.modality=="t2":
            self.save_my_img(t2, case_name, 0, imagestr)
            lab_array[np.where(binary_edema_array == 1)] = 1
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_t2_lab)
        elif self.args.modality=='de':
            self.save_my_img(de, case_name, 0, imagestr)
            lab_array[np.where(binary_scar_array == 1)] = 1
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
        else:
            print("unsupport type")
            exit(-912)

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})


    # def save_one_train_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
    #     lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
    #     lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
    #     lab_de_lab = sitk.ReadImage([de['gt_lab']])
    #     lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
    #     lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
    #     lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200,500,1220,2221])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [200,500,1220,2221])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [200,500,1220,2221])
    #
    #
    #     mask = (binary_myo_array + binary_edema_array + binary_scar_array)
    #     mask = np.where(mask>0,1,0)
    #
    #     kernel = skimage.morphology.disk(5)
    #     mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
    #
    #     binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
    #     binary_edema_array = binarize_numpy_array(lab_t2_array, [1220])
    #     binary_scar_array = binarize_numpy_array(lab_de_array, [2221])
    #
    #     lab_array = np.zeros_like(binary_myo_array)
    #     lab_array[np.where(binary_myo_array == 1)] = 1
    #     lab_array[np.where(binary_edema_array == 1)] = 2
    #     lab_array[np.where(binary_scar_array == 1)] = 3
    #
    #     self.save_masked_img(c0, mask, case_name, 0, imagestr)
    #     self.save_masked_img(t2, mask, case_name, 1, imagestr)
    #     self.save_masked_img(de, mask, case_name, 2, imagestr)
    #
    #     # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
    #     self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)


    def save_my_img(self, pathes,  case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        array = sitk.GetArrayFromImage(img)
        self.write_array_for_nnunet(array, outputdir, case_name, f'000{id}',False,img)
    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": self.args.modality
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "scar" if self.args.modality=='de' else "edema",

        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class RJMvMMnnuentPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(RJMvMMnnuentPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.upper()}_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.upper()}_manual_[0-9].nii.gz"))

        print(dict)

        list=[]
        for i,g in zip(dict["img"],dict['gt_lab']):
            list.append({"img":i,'gt_lab':g})

        return list


    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir}/*[0-9]')
        c0s=self._getsample(subjects, "C0")
        t2s=self._getsample(subjects, "T2")
        des=self._getsample(subjects, "DE")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            c0_terms=os.path.basename(c0['gt_lab']).split('.')[0].split('_')
            t2_terms=os.path.basename(t2['gt_lab']).split('.')[0].split('_')
            de_terms=os.path.basename(de['gt_lab']).split('.')[0].split('_')
            assert c0_terms[1]==t2_terms[1]
            assert c0_terms[1]==de_terms[1]
            case_name = "%s_%s" % (c0_terms[1],c0_terms[-1])
            print(f"processing: {case_name}")
            if int(c0_terms[1][7:])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def save_one_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):

        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #


        lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2
        binary_myo_array = binarize_numpy_array(lab_c0_array, [200])
        binary_edema_array = binarize_numpy_array(lab_t2_array, [1220,2221])
        binary_scar_array = binarize_numpy_array(lab_de_array, [2221])

        lab_array = np.zeros_like(binary_myo_array)
        if self.args.modality=="t2":
            lab_array[np.where(binary_edema_array == 1)] = 1
            self.write_2Darray_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_t2_lab)
        elif self.args.modality=='de':
            lab_array[np.where(binary_scar_array == 1)] = 1
            self.write_2Darray_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
        else:
            print("unsupport type")
            exit(-912)
        self.save_my_img(de, case_name, 0, imagestr)
        self.save_my_img(t2, case_name, 1, imagestr)

    def write_2Darray_for_nnunet(self, img, out_dir, name, id="0000", is_label=False,para=None):

        # if is_label==False:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
        # else:
        #     img_itk = sitk.GetImageFromArray(img.astype(np.uint8)[None])

        img=np.squeeze(img)
        img=np.expand_dims(img,0)

        if is_label==False:
            img_itk = sitk.GetImageFromArray(img.astype(np.float32))
        else:
            img_itk = sitk.GetImageFromArray(img.astype(np.uint8))


        img_itk.SetSpacing(list(self.spacing)+ [999])



        if is_label==False:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}_{id}.nii.gz"))
        else:
            sitk.WriteImage(img_itk, os.path.join(out_dir+ f"/{name}.nii.gz"))


    def save_my_img(self, pathes,  case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        array = sitk.GetArrayFromImage(img)
        self.write_2Darray_for_nnunet(array, outputdir, case_name, f'000{id}',False,img)
    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "de",
            "1": "t2"
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "scar" if self.args.modality=='de' else "edema",

        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class RJMyoPrior2MPSNPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(RJMyoPrior2MPSNPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects, modality):
        #here mask was predicted by jrs
        dict = {"img": [],"asn_lab": [],'gt_lab':[]}

        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_img_[0-9].nii.gz"))
            dict["asn_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_lab_[0-9].nii.gz"))
            dict["gt_lab"].extend(sort_glob(f"{s}/*{modality.lower()}_assn_gt_lab_[0-9].nii.gz"))
        print(dict)

        list=[]
        for i,l,g in zip(dict["img"],dict["asn_lab"],dict['gt_lab']):
            list.append({"img":i,"asn_lab":l,'gt_lab':g})

        return list

    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir_ASN_Out}/*[0-9]')
        c0s=self._getsample(subjects, "c0")
        t2s=self._getsample(subjects, "t2")
        des=self._getsample(subjects, "de")
        self.save_for_nnunet(c0s, t2s, des)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, c0s, t2s, des):
        for c0,t2,de in zip( c0s, t2s, des):
            terms=os.path.basename(c0['asn_lab']).split('.')[0].split('_')
            case_name = "%s_%s_%s_%s" % (terms[0],terms[1],terms[2],terms[6])

            if int(terms[1])>1025:
                self.test_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagests, self.labelsts, t2)
            else:
                self.train_patient_names.append(case_name)
                self.save_one_test_img_lab(c0, case_name, de, self.imagestr, self.labelstr, t2)

    def _check_parameter(self,a:sitk.Image,b:sitk.Image):
        assert  a.GetSize()==b.GetSize()
        assert  a.GetSpacing()==b.GetSpacing()
        assert a.GetOrigin()==b.GetOrigin()
        assert a.GetDirection()==b.GetDirection()

    def save_one_test_img_lab(self, c0, case_name, de, imagestr, labelstr, t2):
        # self.save_img(c0,  case_name, 0, imagestr)


        #the predict ROI from ASN, it will be propagated to PSN as Attention for pathology segmentation.
        asn_labC0 = sitk.GetArrayFromImage(sitk.ReadImage(c0['asn_lab']))  # c0,de,t2
        asn_labT2 = sitk.GetArrayFromImage(sitk.ReadImage(t2['asn_lab']))  # c0,de,t2
        asn_labDE = sitk.GetArrayFromImage(sitk.ReadImage(de['asn_lab']))  # c0,de,t2
        self._check_parameter(sitk.ReadImage(c0['asn_lab']),sitk.ReadImage(t2['asn_lab']))
        self._check_parameter(sitk.ReadImage(de['asn_lab']),sitk.ReadImage(t2['asn_lab']))



        #i add prior and use it as attention for psn. do not perform data augmentation operation on prior channal
        #see nnuent/preprocessing/preprocessing.py  search KEYWORD scheme=="prior":
        if self.args.comspace=="de":
            prior=asn_labDE
            self.save_prior(prior, de ,case_name, 0, imagestr)
            self.save_img(de,  case_name, 1, imagestr)
        elif self.args.comspace=="t2":
            # prior
            prior = asn_labT2
            self.save_prior(prior, t2 ,case_name, 0, imagestr)
            self.save_img(t2,  case_name, 1, imagestr)
        else:
            exit(-912)


        # kernel = skimage.morphology.disk(5)
        # mask[0,:,:] = skimage.morphology.dilation(mask[0,:,:], kernel)
        #
        # the ground truth of pathology and predict segmentation mask
        #
        # lab_c0_lab = sitk.ReadImage([c0['gt_lab']])
        lab_t2_lab = sitk.ReadImage([t2['gt_lab']])
        lab_de_lab = sitk.ReadImage([de['gt_lab']])
        # lab_c0_array = sitk.GetArrayFromImage(lab_c0_lab)  # c0,de,t2
        lab_t2_array = sitk.GetArrayFromImage(lab_t2_lab)  # c0,de,t2
        lab_de_array = sitk.GetArrayFromImage(lab_de_lab)  # c0,de,t2

        # binary_myo_array = binarize_numpy_array(lab_c0_array, [200],outindex=1)


        # lab_array = binary_edema_array+binary_scar_array
        # i add prior here to avoid intensity-based data augmentation operation, it could be used as attention for psn segmentation
        # prior = np.where(prior > 0.5, 1, 0)
        # lab_array = binary_edema_array+binary_scar_array+prior

        # myo_lab_array = reindex_label_array_by_dict(lab_array, {1: [200], 2: [1220], 3: [2221]})
        if self.args.comspace=="de":
            binary_scar_array = binarize_numpy_array(lab_de_array, [2221], outindex=1)
            lab_array = binary_scar_array
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_de_lab)
        elif self.args.comspace=="t2":
            binary_edema_array = binarize_numpy_array(lab_t2_array, [1220], outindex=1)
            lab_array=binary_edema_array
            self.write_array_for_nnunet(lab_array, labelstr, case_name, is_label=True, para=lab_t2_lab)
        else:
            exit(-912)


    def write_mask_img(self,pathes,mask,case_name,id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        mask=sitk.GetImageFromArray(mask)
        mask = binarize_img(mask, [1, 2, 3, 5])
        masked_array = sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')


    def _write(self, pathes, case_name, id,imagestr):
        img = sitk.ReadImage(pathes['img'])
        lab = sitk.ReadImage(pathes['lab'])
        mask = binarize_img(lab, [1, 2, 3, 5])
        masked_array= sitk.GetArrayFromImage(img) * sitk.GetArrayFromImage(mask).astype(np.float)

        self.write_array_for_nnunet(masked_array, imagestr, case_name, f'000{id}')

    def save_img(self, pathes,  case_name, id, outputdir):
        img = sitk.ReadImage(pathes['img'])
        masked_array = sitk.GetArrayFromImage(img)
        self.write_array_for_nnunet(masked_array, outputdir, case_name, f'000{id}',False,img)

    def save_prior(self,prior,para,case_name,id,outputdir):
        img = sitk.ReadImage(para['img'])
        self.write_array_for_nnunet(prior, outputdir, case_name, f'000{id}',False,img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": f"prior",
            "1": f"{self.args.comspace}" ,
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "pathology",

        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

from tools.np_sitk_tools import sitkResize3D
class LascarPrepare(nnUNetPrepare):
    def __init__(self,args):
        super(LascarPrepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args=args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects,target):
        #here mask was predicted by jrs
        dict = {"image": [],"label": []}

        for s in subjects:
            dict["image"].extend(sort_glob(f"{s}/enhanced.nii.gz"))
            if target=="LA":
                dict["label"].extend(sort_glob(f"{s}/atriumSegImgMO.nii.gz"))
            else:
                dict["label"].extend(sort_glob(f"{s}/scarSegImgM.nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["image"],dict["label"]):
            list.append({"img":i,"lab":l})

        return list


    def prepare(self):

        subjects= sort_glob(f'{self.args.data_dir}/*')
        datas=self._getsample(subjects, self.args.target)
        self.save_for_nnunet(datas)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, datas):
        for data in datas:
            terms=os.path.basename(os.path.dirname(data['lab'])).split("_")[-1]
            case_name = terms.zfill(4)
            self.train_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)

            # if int(terms) < 51:
            #     self.train_patient_names.append(case_name)
            #     self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)
            # else:
            #     self.test_patient_names.append(case_name)
            #     self.save_one_img_lab(data, case_name, self.imagests, self.labelsts)



    def save_one_img_lab(self, data, case_name, imagestr, labelstr):

        self.save_my_img(data, case_name, 0, imagestr)
        self.save_my_lab(data, case_name,  labelstr)


    def resove_bug(self,path):
        img=nib.load(path)
        qform=img.get_qform()
        img.set_qform(qform)
        sform=img.get_sform()
        img.set_sform(sform)
        nib.save(img,path)

    def save_my_img(self, pathes,  case_name, id, outputdir):

        self.resove_bug(pathes['img'])
        img = sitk.ReadImage(pathes['img'])
        img=sitkResize3D(img,(128,128,44),sitk.sitkLinear)
        array = sitk.GetArrayFromImage(img)
        self.write_array_for_nnunet(array, outputdir, case_name, f'000{id}',False,img)

    def save_my_lab(self, pathes,  case_name, outputdir):
        self.resove_bug(pathes['lab'])
        img = sitk.ReadImage(pathes['lab'])
        img = sitkResize3D(img, (128, 128, 44),sitk.sitkNearestNeighbor)
        array = sitk.GetArrayFromImage(img)
        array=np.where(array>0,1,0)
        self.write_array_for_nnunet(array, outputdir, case_name, is_label=True,para=img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "lge"
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "forground"

        }

        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))

class myo_LV_Prepare(nnUNetPrepare):
    def __init__(self, args):
        super(myo_LV_Prepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args = args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects,target):
        #here mask was predicted by jrs
        dict = {"image": [],"label": []}

        for s in subjects:
            dict["image"].extend(sort_glob(f"{s}/*_image.nii.gz"))
            dict["label"].extend(sort_glob(f"{s}/*_label.nii.gz"))
            # dict["mask"].extend(sort_glob(f"{s}/Contours/*.nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["image"],dict["label"]):
            list.append({"img":i,"lab":l})

        return list

    # def crop(self, p_img1, p_lab, p_img2='none'):
    #     # print(p_img)
    #     img = sitk.ReadImage(p_img1)
    #     if p_img2 != 'none':
    #         img2 = sitk.ReadImage(p_img2)
    #     lab = sitk.ReadImage(p_lab)
    #     # lab_mask = sitk.BinaryThreshold(lab, 0.1, 10, 1, 0)
    #     # lab_mask_array=sitk.GetArrayFromImage(lab_mask)
    #     lab_mask_array = sitk.GetArrayFromImage(lab)
    #     lab_mask_array = np.where(lab_mask_array > 0.1, 1, 0)
    #     bbox = get_bounding_box_by_ids(lab_mask_array, 15, [1])
    #     crop_lab = crop_by_bbox(lab, bbox)
    #     crop_img = crop_by_bbox(img, bbox)
    #     if p_img2 != 'none':
    #         crop_img2 = crop_by_bbox(img2, bbox)
    #     # sitk.WriteImage(crop_lab, p_lab)
    #     sitk.WriteImage(crop_img, p_img1)
    #     if p_img2 != 'none':
    #         sitk.WriteImage(crop_img2, p_img2)
    #     # return crop_img,crop_lab

    def prepare(self):
        # print(self.args.data_dir)
        # subjects= sort_glob(f'{self.args.data_dir}/*')
        subjects = sort_glob(f'{self.args.data_dir}')
        subjects2 = sort_glob(f'{self.args.test_dir}')

        datas_train = self._getsample(subjects, self.args.target)
        datas_test = self._getsample(subjects2, self.args.target)
        self.save_for_nnunet(datas_train, datas_test)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, datas_train, datas_test):
        for data in datas_train:
            case_name = os.path.basename(data['lab']).split("_")[0]+"_"+os.path.basename(data['lab']).split("_")[1]
            # case_name ="case"+str(terms).zfill(4)
            # self.train_patient_names.append(case_name)
            # self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)
            # print(int(os.path.basename(data['lab']).split("_")[0][4:]))
            self.train_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)

        for data in datas_test:
            case_name = os.path.basename(data['lab']).split("_")[0] + "_" + \
                        os.path.basename(data['lab']).split("_")[1]
            self.test_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagests, self.labelsts)



    def save_one_img_lab(self, data, case_name, imagestr, labelstr):

        self.save_my_img(data, case_name,  imagestr)
        self.save_my_lab(data, case_name,  labelstr)

    def save_my_img(self, pathes, case_name, outputdir):

        print(pathes['img'])
        self.resove_bug(pathes['img'])
        # self.resove_bug(pathes['mask'])
        img = sitk.ReadImage(pathes['img'])
        # mask=sitk.ReadImage(pathes['mask'])
        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # img=crop_by_bbox(img,bbox)
        array = sitk.GetArrayFromImage(img)
        for i in range(1):
            self.write_array_for_nnunet(array[i,:, :, :], outputdir, case_name, f'{i:04d}', False, img[:, :,:, i])

    def resove_bug(self,path):
        img=nib.load(path)
        qform=img.get_qform()
        img.set_qform(qform)
        sform=img.get_sform()
        img.set_sform(sform)
        nib.save(img,path)

    def save_my_lab(self, pathes, case_name, outputdir):

        self.resove_bug(pathes['lab'])
        # self.resove_bug(pathes['mask'])

        lab = sitk.ReadImage(pathes['lab'])
        # mask=sitk.ReadImage(pathes['mask'])

        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # lab=crop_by_bbox(lab,bbox)
        array = sitk.GetArrayFromImage(lab)
        # array = np.where(array == 500, 1, array)     # !
        # array = np.where(array == 600, 2, array)
        # array = np.where(array == 420, 3, array)
        # array = np.where(array == 550, 4, array)
        # array = np.where(array == 205, 5, array)
        # array = np.where(array == 820, 6, array)
        # array = np.where(array == 850, 7, array)

        array = np.where(array == 200, 1, array)  # !
        array = np.where(array == 500, 2, array)
        array = np.where(array == 600, 3, array)
        array = np.where(array == 1220, 4, array)
        array = np.where(array == 2221, 5, array)


        # array = np.where(array ==1 , 0, array)
        # mask=crop_by_bbox(mask,bbox)

        # array_mask= sitk.GetArrayFromImage(mask)
        # array_mask = np.where(array_mask > 0, 1, 0)
        #
        # array_lab = sitk.GetArrayFromImage(lab)
        # array_lab = np.where(array_lab > 0, 2, 0)
        #
        # array = array_mask + array_lab


        self.write_array_for_nnunet(array, outputdir, case_name, f'000{id}', True, lab)

        # if self.args.task_id == 854:
        #     img_scar_path=os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     img_scar=sitk.ReadImage(img_scar_path)
        #     img_scar = sitk.BinaryThreshold(img_scar, lowerThreshold=1, upperThreshold=1, insideValue=2,
        #                                      outsideValue=0)
        #     img_scar = sitk.Cast(img_scar, sitk.sitkFloat32)
        #     img = img + img_scar
        #
        #     print('kk',img_scar_path)

        # array = sitk.GetArrayFromImage(img)
        # array = np.where(array > 0, 1, 0)
        # if self.args.task_id in [854]:
        #     img_scar_path = os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     self.resove_bug(img_scar_path)
        #     img_scar = sitk.ReadImage(img_scar_path)
        #     array1 = sitk.GetArrayFromImage(img_scar)
        #     array1 = np.where(array1 > 0, 2, 0)
        #     array = array + array1
        # self.write_array_for_nnunet(array, outputdir, case_name, is_label=True, para=img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "cine0",
            # "1": "cine1",
            # "2": "cine2",
            # "3": "cine3",
            # "4": "cine4",
            # "5": "cine5",
            # "6": "cine6",
            # "7": "cine7",
            # "8": "cine8",
            # "9": "cine9",
            # "10": "cine10",
            # "11": "cine11",
            # "12": "cine12",
            # "13": "cine13",
            # "14": "cine14",
            # "15": "cine15",
            # "16": "cine16",
            # ....
            #TODO
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "lv",
            "2": "rv",
            "3": "la",
            "4": "ra",
            "5": "myo",
            # "6": "aao",
            # "7": "pa",
            # "9": "ivc",
            # "10": "pv"

        }


        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class myo_ana_Prepare(nnUNetPrepare):
    def __init__(self, args):
        super(myo_ana_Prepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args = args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects,target):
        #here mask was predicted by jrs
        dict = {"image": [],"label": []}

        for s in subjects:
            dict["image"].extend(sort_glob(f"{s}/image/*.nii.gz"))
            dict["label"].extend(sort_glob(f"{s}/label/*.nii.gz"))
            # dict["mask"].extend(sort_glob(f"{s}/Contours/*.nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["image"],dict["label"]):
            list.append({"img":i,"lab":l})

        return list

    # def crop(self, p_img1, p_lab, p_img2='none'):
    #     # print(p_img)
    #     img = sitk.ReadImage(p_img1)
    #     if p_img2 != 'none':
    #         img2 = sitk.ReadImage(p_img2)
    #     lab = sitk.ReadImage(p_lab)
    #     # lab_mask = sitk.BinaryThreshold(lab, 0.1, 10, 1, 0)
    #     # lab_mask_array=sitk.GetArrayFromImage(lab_mask)
    #     lab_mask_array = sitk.GetArrayFromImage(lab)
    #     lab_mask_array = np.where(lab_mask_array > 0.1, 1, 0)
    #     bbox = get_bounding_box_by_ids(lab_mask_array, 15, [1])
    #     crop_lab = crop_by_bbox(lab, bbox)
    #     crop_img = crop_by_bbox(img, bbox)
    #     if p_img2 != 'none':
    #         crop_img2 = crop_by_bbox(img2, bbox)
    #     # sitk.WriteImage(crop_lab, p_lab)
    #     sitk.WriteImage(crop_img, p_img1)
    #     if p_img2 != 'none':
    #         sitk.WriteImage(crop_img2, p_img2)
    #     # return crop_img,crop_lab

    def prepare(self):
        # print(self.args.data_dir)
        # subjects= sort_glob(f'{self.args.data_dir}/*')
        subjects = sort_glob(f'{self.args.data_dir}')
        subjects2 = sort_glob(f'{self.args.test_dir}')

        datas_train = self._getsample(subjects, self.args.target)
        datas_test = self._getsample(subjects2, self.args.target)
        self.save_for_nnunet(datas_train, datas_test)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, datas_train, datas_test):
        for data in datas_train:
            case_name = os.path.basename(data['lab']).split(".")[0]
            # case_name ="case"+str(terms).zfill(4)
            # self.train_patient_names.append(case_name)
            # self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)
            # print(int(os.path.basename(data['lab']).split("_")[0][4:]))
            self.train_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)

        for data in datas_test:
            case_name = os.path.basename(data['lab']).split(".")[0]
            self.test_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagests, self.labelsts)



    def save_one_img_lab(self, data, case_name, imagestr, labelstr):

        self.save_my_img(data, case_name,  imagestr)
        self.save_my_lab(data, case_name,  labelstr)

    def save_my_img(self, pathes, case_name, outputdir):

        print(pathes['img'])
        self.resove_bug(pathes['img'])
        # self.resove_bug(pathes['mask'])
        img = sitk.ReadImage(pathes['img'])
        # mask=sitk.ReadImage(pathes['mask'])
        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # img=crop_by_bbox(img,bbox)
        array = sitk.GetArrayFromImage(img)
        for i in range(17):
            self.write_array_for_nnunet(array[i, :, :], outputdir, case_name, f'{i:04d}', False, img[:, :, i])

    def resove_bug(self,path):
        img=nib.load(path)
        qform=img.get_qform()
        img.set_qform(qform)
        sform=img.get_sform()
        img.set_sform(sform)
        nib.save(img,path)

    def save_my_lab(self, pathes, case_name, outputdir):

        self.resove_bug(pathes['lab'])
        # self.resove_bug(pathes['mask'])

        lab = sitk.ReadImage(pathes['lab'])
        # mask=sitk.ReadImage(pathes['mask'])

        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # lab=crop_by_bbox(lab,bbox)
        array = sitk.GetArrayFromImage(lab)
        # array = np.where(array == 500, 1, array)     # !
        # array = np.where(array == 600, 2, array)
        # array = np.where(array == 420, 3, array)
        # array = np.where(array == 550, 4, array)
        # array = np.where(array == 205, 5, array)
        # array = np.where(array == 820, 6, array)
        # array = np.where(array == 850, 7, array)

        # array = np.where(array == 200, 1, array)  # !
        # array = np.where(array == 500, 2, array)
        # array = np.where(array == 600, 3, array)
        # array = np.where(array == 1220, 4, array)
        # array = np.where(array == 2221, 5, array)


        # array = np.where(array ==1 , 0, array)
        # mask=crop_by_bbox(mask,bbox)

        # array_mask= sitk.GetArrayFromImage(mask)
        # array_mask = np.where(array_mask > 0, 1, 0)
        #
        # array_lab = sitk.GetArrayFromImage(lab)
        # array_lab = np.where(array_lab > 0, 2, 0)
        #
        # array = array_mask + array_lab

        # array = [array[:1], array[8:9]]
        # array = np.concatenate(array, axis=0)

        self.write_array_for_nnunet(array, outputdir, case_name, f'000{id}', True, lab)

        # if self.args.task_id == 854:
        #     img_scar_path=os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     img_scar=sitk.ReadImage(img_scar_path)
        #     img_scar = sitk.BinaryThreshold(img_scar, lowerThreshold=1, upperThreshold=1, insideValue=2,
        #                                      outsideValue=0)
        #     img_scar = sitk.Cast(img_scar, sitk.sitkFloat32)
        #     img = img + img_scar
        #
        #     print('kk',img_scar_path)

        # array = sitk.GetArrayFromImage(img)
        # array = np.where(array > 0, 1, 0)
        # if self.args.task_id in [854]:
        #     img_scar_path = os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     self.resove_bug(img_scar_path)
        #     img_scar = sitk.ReadImage(img_scar_path)
        #     array1 = sitk.GetArrayFromImage(img_scar)
        #     array1 = np.where(array1 > 0, 2, 0)
        #     array = array + array1
        # self.write_array_for_nnunet(array, outputdir, case_name, is_label=True, para=img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "cine0",
            "1": "cine1",
            "2": "cine2",
            "3": "cine3",
            "4": "cine4",
            "5": "cine5",
            "6": "cine6",
            "7": "cine7",
            "8": "cine8",
            "9": "cine9",
            "10": "cine10",
            "11": "cine11",
            "12": "cine12",
            "13": "cine13",
            "14": "cine14",
            "15": "cine15",
            "16": "cine16",
            # ....
            #TODO
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "lv",
            "2": "rv",
            "3": "la",
            # "4": "ra",
            # "5": "myo",
            # "6": "aao",
            # "7": "pa",
            # "9": "ivc",
            # "10": "pv"

        }


        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))


class myo_path_Prepare(nnUNetPrepare):
    def __init__(self, args):
        super(myo_path_Prepare, self).__init__(args)
        self.op = SkimageOP_Base()
        self.args = args
        mk_or_cleardir(self.imagests)
        mk_or_cleardir(self.labelsts)
        mk_or_cleardir(self.imagestr)
        mk_or_cleardir(self.labelstr)

    def _getsample(self, subjects,target):
        #here mask was predicted by jrs
        dict = {"image": [],"label": []}

        for s in subjects:
            dict["image"].extend(sort_glob(f"{s}/*_image.nii.gz"))
            dict["label"].extend(sort_glob(f"{s}/*_label.nii.gz"))
            # dict["mask"].extend(sort_glob(f"{s}/Contours/*.nii.gz"))
        print(dict)

        list=[]
        for i,l in zip(dict["image"],dict["label"]):
            list.append({"img":i,"lab":l})

        return list

    # def crop(self, p_img1, p_lab, p_img2='none'):
    #     # print(p_img)
    #     img = sitk.ReadImage(p_img1)
    #     if p_img2 != 'none':
    #         img2 = sitk.ReadImage(p_img2)
    #     lab = sitk.ReadImage(p_lab)
    #     # lab_mask = sitk.BinaryThreshold(lab, 0.1, 10, 1, 0)
    #     # lab_mask_array=sitk.GetArrayFromImage(lab_mask)
    #     lab_mask_array = sitk.GetArrayFromImage(lab)
    #     lab_mask_array = np.where(lab_mask_array > 0.1, 1, 0)
    #     bbox = get_bounding_box_by_ids(lab_mask_array, 15, [1])
    #     crop_lab = crop_by_bbox(lab, bbox)
    #     crop_img = crop_by_bbox(img, bbox)
    #     if p_img2 != 'none':
    #         crop_img2 = crop_by_bbox(img2, bbox)
    #     # sitk.WriteImage(crop_lab, p_lab)
    #     sitk.WriteImage(crop_img, p_img1)
    #     if p_img2 != 'none':
    #         sitk.WriteImage(crop_img2, p_img2)
    #     # return crop_img,crop_lab

    def prepare(self):
        # print(self.args.data_dir)
        # subjects= sort_glob(f'{self.args.data_dir}/*')
        subjects = sort_glob(f'{self.args.data_dir}')
        subjects2 = sort_glob(f'{self.args.test_dir}')

        datas_train = self._getsample(subjects, self.args.target)
        datas_test = self._getsample(subjects2, self.args.target)
        self.save_for_nnunet(datas_train, datas_test)

        self.write_json()
        # self.write_json()
    def save_for_nnunet(self, datas_train, datas_test):
        for data in datas_train:
            case_name = os.path.basename(data['lab']).split("_label")[0]
            # case_name ="case"+str(terms).zfill(4)
            # self.train_patient_names.append(case_name)
            # self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)
            # print(int(os.path.basename(data['lab']).split("_")[0][4:]))
            self.train_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagestr, self.labelstr)

        for data in datas_test:
            case_name = os.path.basename(data['lab']).split("_label")[0]
            self.test_patient_names.append(case_name)
            self.save_one_img_lab(data, case_name, self.imagests, self.labelsts)



    def save_one_img_lab(self, data, case_name, imagestr, labelstr):

        self.save_my_img(data, case_name,  imagestr)
        self.save_my_lab(data, case_name,  labelstr)

    def save_my_img(self, pathes, case_name, outputdir):

        print(pathes['img'])
        self.resove_bug(pathes['img'])
        # self.resove_bug(pathes['mask'])
        img = sitk.ReadImage(pathes['img'])
        # mask=sitk.ReadImage(pathes['mask'])
        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # img=crop_by_bbox(img,bbox)
        array = sitk.GetArrayFromImage(img)
        for i in range(17):
            self.write_array_for_nnunet(array[i, :, :, :], outputdir, case_name, f'{i:04d}', False, img[:, :, :, i])

    def resove_bug(self,path):
        img=nib.load(path)
        qform=img.get_qform()
        img.set_qform(qform)
        sform=img.get_sform()
        img.set_sform(sform)
        nib.save(img,path)

    def save_my_lab(self, pathes, case_name, outputdir):

        self.resove_bug(pathes['lab'])
        # self.resove_bug(pathes['mask'])

        lab = sitk.ReadImage(pathes['lab'])
        # mask=sitk.ReadImage(pathes['mask'])

        # bbox=get_bounding_box_by_idsV3(sitk.Cast(mask,sitk.sitkInt32),[0.1,0.1,0.1],[1,2,3,4])
        # lab=crop_by_bbox(lab,bbox)
        array = sitk.GetArrayFromImage(lab)
        # array = np.where(array == 500, 1, array)     # !
        # array = np.where(array == 600, 2, array)
        # array = np.where(array == 420, 3, array)
        # array = np.where(array == 550, 4, array)
        # array = np.where(array == 205, 5, array)
        # array = np.where(array == 820, 6, array)
        # array = np.where(array == 850, 7, array)

        array = np.where(array == 200, 1, array)  # !
        array = np.where(array == 500, 2, array)
        array = np.where(array == 600, 3, array)
        array = np.where(array == 1220, 4, array)
        array = np.where(array == 2221, 5, array)


        # array = np.where(array ==1 , 0, array)
        # mask=crop_by_bbox(mask,bbox)

        # array_mask= sitk.GetArrayFromImage(mask)
        # array_mask = np.where(array_mask > 0, 1, 0)
        #
        # array_lab = sitk.GetArrayFromImage(lab)
        # array_lab = np.where(array_lab > 0, 2, 0)
        #
        # array = array_mask + array_lab

        # array = [array[:1], array[8:9]]
        # array = np.concatenate(array, axis=0)

        self.write_array_for_nnunet(array, outputdir, case_name, f'000{id}', True, lab)

        # if self.args.task_id == 854:
        #     img_scar_path=os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     img_scar=sitk.ReadImage(img_scar_path)
        #     img_scar = sitk.BinaryThreshold(img_scar, lowerThreshold=1, upperThreshold=1, insideValue=2,
        #                                      outsideValue=0)
        #     img_scar = sitk.Cast(img_scar, sitk.sitkFloat32)
        #     img = img + img_scar
        #
        #     print('kk',img_scar_path)

        # array = sitk.GetArrayFromImage(img)
        # array = np.where(array > 0, 1, 0)
        # if self.args.task_id in [854]:
        #     img_scar_path = os.path.dirname(pathes['lab']) + '/' + r'scarSegImgM.nii.gz'
        #     self.resove_bug(img_scar_path)
        #     img_scar = sitk.ReadImage(img_scar_path)
        #     array1 = sitk.GetArrayFromImage(img_scar)
        #     array1 = np.where(array1 > 0, 2, 0)
        #     array = array + array1
        # self.write_array_for_nnunet(array, outputdir, case_name, is_label=True, para=img)

    def write_json(self):
        json_dict = {}
        json_dict['name'] = self.taskname
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "cine0",
            "1": "cine1",
            "2": "cine2",
            "3": "cine3",
            "4": "cine4",
            "5": "cine5",
            "6": "cine6",
            "7": "cine7",
            "8": "cine8",
            "9": "cine9",
            "10": "cine10",
            "11": "cine11",
            "12": "cine12",
            "13": "cine13",
            "14": "cine14",
            "15": "cine15",
            "16": "cine16",
            # ....
            #TODO
        }

        json_dict['labels'] = {
            "0": "background",
            "1": "lv",
            "2": "rv",
            "3": "la",
            "4": "ra",
            "5": "myo",
            # "6": "aao",
            # "7": "pa",
            # "9": "ivc",
            # "10": "pv"

        }


        json_dict['numTraining'] = len(self.train_patient_names)
        json_dict['numTest'] = len(self.test_patient_names)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                 self.train_patient_names]
        if len(self.test_patient_names) > 0:
            json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in self.test_patient_names]
        else:
            json_dict['test'] = []

        # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in test_patient_names]

        save_json(json_dict, os.path.join(self.out_base, "dataset.json"))