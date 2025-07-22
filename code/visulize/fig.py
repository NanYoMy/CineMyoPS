import cv2
import SimpleITK as sitk
from tools.np_sitk_tools import crop_by_bbox,get_bounding_box_by_ids,sitkRespacing
from visulize.helper import show_img,extract_semantic_contour,draw_coutours
from visulize.color import palette

ref_img=sitk.ReadImage("E:\homework2\myopos-segnetwork\img\intro\mis-aligned\subject_33_ana_c0.nii.gz")
# ref_img=sitkRespacing(ref_img,)
ref_img=sitkRespacing(ref_img, sitk.sitkNearestNeighbor, [1, 1, ref_img.GetSpacing()[-1]])
ref_bbox=get_bounding_box_by_ids(ref_img, padding=20, ids=[200,1220,2221, 500, 600])
ref_bbox2=get_bounding_box_by_ids(ref_img, padding=20, ids=[200,1220,2221, 500])
from tools.np_sitk_tools import clipseScaleSitkImage, sitkResize
import numpy as np


# def resample(self, img, lab):
#     print(img.GetSpacing())
#     n_lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])
#
#     n_img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
#     return n_img, n_lab

from tools.np_sitk_tools import reindex_label_array_by_dict

def show_img_slice(p_img, p_lab, slice, name):
    img=sitk.ReadImage(p_img)
    lab=sitk.ReadImage(p_lab)
    img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
    lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])
    img=crop_by_bbox(img, ref_bbox)
    lab=crop_by_bbox(lab, ref_bbox)
    img=clipseScaleSitkImage(img,0,100)

    img = sitkResize(img, [500, 500, img.GetSize()[-1]], sitk.sitkLinear)
    lab = sitkResize(lab, [500, 500, lab.GetSize()[-1]], sitk.sitkNearestNeighbor)

    img=sitk.GetArrayFromImage(img).astype(np.uint8)
    lab=sitk.GetArrayFromImage(lab)

    lab=reindex_label_array_by_dict(lab,{200:[200,1220,2221],600:[600],500:[500]})

    tmp_img=img[slice,...]
    tmp_lab=lab[slice,...]
    contours=extract_semantic_contour(tmp_lab)
    # tmp_img=draw_coutours(np.tile(np.expand_dims(tmp_img,-1),(1,1,3)),contours,{500:palette[0],200:palette[1],600:palette[7],1222:palette[3],2221:palette[4]})
    # show_img(tmp_img)
    # tmp_img=cv2.resize(tmp_img,(500,500))
    cv2.imwrite(name,tmp_img)

def show_masked_img_slice(p_img,p_lab,slice,name):
    img = sitk.ReadImage(p_img)
    lab = sitk.ReadImage(p_lab)
    img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
    lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])
    img = crop_by_bbox(img, ref_bbox2)
    lab = crop_by_bbox(lab, ref_bbox2)
    img = clipseScaleSitkImage(img, 0, 100)

    img = sitkResize(img, [500, 500, img.GetSize()[-1]], sitk.sitkLinear)
    lab = sitkResize(lab, [500, 500, lab.GetSize()[-1]], sitk.sitkNearestNeighbor)

    img = sitk.GetArrayFromImage(img).astype(np.uint8)
    lab = sitk.GetArrayFromImage(lab)

    lab = reindex_label_array_by_dict(lab, {200: [200, 1220, 2221], 600: [600], 500: [500]})

    tmp_img = img[slice, ...]
    tmp_lab = lab[slice, ...]
    tmp_img=tmp_img*reindex_label_array_by_dict(tmp_lab,{1: [200, 1220, 2221,500]})

    cv2.imwrite(name, tmp_img)

def show_img_slice_with_contour(p_img, p_lab, slice, name):
    img=sitk.ReadImage(p_img)
    lab=sitk.ReadImage(p_lab)
    img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
    lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])
    img=crop_by_bbox(img, ref_bbox)
    lab=crop_by_bbox(lab, ref_bbox)
    img=clipseScaleSitkImage(img,0,100)

    img = sitkResize(img, [500, 500, img.GetSize()[-1]], sitk.sitkLinear)
    lab = sitkResize(lab, [500, 500, lab.GetSize()[-1]], sitk.sitkNearestNeighbor)

    img=sitk.GetArrayFromImage(img).astype(np.uint8)
    lab=sitk.GetArrayFromImage(lab)

    lab=reindex_label_array_by_dict(lab,{200:[200,1220,2221,500],600:[600]})

    tmp_img=img[slice,...]
    tmp_lab=lab[slice,...]
    contours=extract_semantic_contour(tmp_lab)
    tmp_img=draw_coutours(np.tile(np.expand_dims(tmp_img,-1),(1,1,3)),contours,{500:palette[0],200:palette[7],600:palette[7],1222:palette[3],2221:palette[4]})
    # show_img(tmp_img)
    # tmp_img=cv2.resize(tmp_img,(500,500))
    cv2.imwrite(name,tmp_img)




from visulize.helper import add_label_2_img
def show_img_slice_with_pathology(p_img,p_lab,slice,name,show_lab= {1: [1220]}):
    img = sitk.ReadImage(p_img)
    lab = sitk.ReadImage(p_lab)
    img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
    lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])



    img = crop_by_bbox(img, ref_bbox)
    lab = crop_by_bbox(lab, ref_bbox)
    img = clipseScaleSitkImage(img, 0, 100)

    img = sitkResize(img, [500, 500, img.GetSize()[-1]], sitk.sitkLinear)
    lab = sitkResize(lab, [500, 500, lab.GetSize()[-1]], sitk.sitkNearestNeighbor)

    img=sitk.GetArrayFromImage(img).astype(np.uint8)
    lab=sitk.GetArrayFromImage(lab)
    lab = reindex_label_array_by_dict(lab,show_lab)
    tmp_img=img[slice,...]
    tmp_lab=lab[slice,...]

    tmp_img=add_label_2_img(tmp_img,tmp_lab,{1:palette[4],2:palette[1],3:palette[2],4:palette[3],5:palette[6]})

    cv2.imwrite(name,tmp_img)


def show_mask_img_slice_with_pathology(p_img,p_lab,slice,name,show_lab= {1: [1220]}):
    img = sitk.ReadImage(p_img)
    lab = sitk.ReadImage(p_lab)
    img = sitkRespacing(img, sitk.sitkLinear, [1, 1, img.GetSpacing()[-1]])
    lab = sitkRespacing(lab, sitk.sitkNearestNeighbor, [1, 1, img.GetSpacing()[-1]])



    img = crop_by_bbox(img, ref_bbox2)
    lab = crop_by_bbox(lab, ref_bbox2)
    img = clipseScaleSitkImage(img, 0, 100)

    img = sitkResize(img, [500, 500, img.GetSize()[-1]], sitk.sitkLinear)
    lab = sitkResize(lab, [500, 500, lab.GetSize()[-1]], sitk.sitkNearestNeighbor)

    img=sitk.GetArrayFromImage(img).astype(np.uint8)
    lab=sitk.GetArrayFromImage(lab)
    lab = reindex_label_array_by_dict(lab,show_lab)
    tmp_img=img[slice,...]
    tmp_lab=lab[slice,...]


    tmp_img=np.ones_like(tmp_img)*255
    tmp_img=add_label_2_img(tmp_img,tmp_lab,{1:palette[4],2:palette[1],3:palette[2],4:palette[3],5:palette[6]})
    cv2.imwrite(name,tmp_img)


from tools.dir import get_name_wo_suffix



if __name__=="__main__":
    slice=1

    #unaligned
    basedir="E:\\homework2\\myopos-segnetwork\\img\\intro\\mis-aligned\\"
    img_path=basedir+"subject_33_img_c0.nii.gz"
    lab_path=basedir+"subject_33_ana_c0.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")


    img_path=basedir+"subject_33_img_de.nii.gz"
    lab_path=basedir+"subject_33_ana_patho_de_scar.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")

    img_path=basedir+"subject_33_img_t2.nii.gz"
    lab_path=basedir+"subject_33_ana_patho_t2_edema.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")

    #aligned
    basedir="E:\\homework2\\myopos-segnetwork\\img\\intro\\aligned\\"
    img_path=basedir+"align_subject33_C0.nii.gz"
    lab_path=basedir+"align_subject33_C0_manual.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")
    show_masked_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_mask_img.png")


    img_path=basedir+"align_subject33_DE.nii.gz"
    lab_path=basedir+"align_subject33_DE_manual.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")
    show_masked_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_mask_img.png")


    img_path=basedir+"align_subject33_T2.nii.gz"
    lab_path=basedir+"align_subject33_T2_manual.nii.gz"
    show_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_img.png")
    show_masked_img_slice(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_mask_img.png")


    #pathology region

    img_path=basedir+"align_subject33_DE.nii.gz"
    lab_path=basedir+"align_subject33_DE_manual_scar.nii.gz"
    show_img_slice_with_pathology(img_path,lab_path,slice,name=get_name_wo_suffix(img_path)+"_label_img.png",show_lab={1: [200]})
    show_mask_img_slice_with_pathology(img_path,lab_path,slice,name=get_name_wo_suffix(img_path)+"_label_mask_img.png",show_lab={1: [200]})

    img_path=basedir+"align_subject33_T2.nii.gz"
    lab_path=basedir+"align_subject33_T2_manual_edema.nii.gz"
    show_img_slice_with_pathology(img_path,lab_path,slice,name=get_name_wo_suffix(img_path)+"_label_img.png",show_lab={5: [200]})
    show_mask_img_slice_with_pathology(img_path,lab_path,slice,name=get_name_wo_suffix(img_path)+"_label_mask_img.png",show_lab={5: [200]})

    img_path = basedir + "align_subject33_C0.nii.gz"
    lab_path = basedir + "align_subject33_C0_manual.nii.gz"
    show_img_slice_with_pathology(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_label_img.png",show_lab={2: [200]})
    show_mask_img_slice_with_pathology(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_label_mask_img.png",show_lab={2: [200]})

    #lv region

    basedir="E:\\homework2\\myopos-segnetwork\\img\\intro\\aligned\\"
    img_path=basedir+"align_subject33_C0.nii.gz"
    lab_path=basedir+"align_subject33_C0_manual.nii.gz"
    show_img_slice_with_contour(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_contour_lv_img.png")

    img_path=basedir+"align_subject33_DE.nii.gz"
    lab_path=basedir+"align_subject33_DE_manual.nii.gz"
    show_img_slice_with_contour(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_contour_lv_img.png")


    img_path=basedir+"align_subject33_T2.nii.gz"
    lab_path=basedir+"align_subject33_T2_manual.nii.gz"
    show_img_slice_with_contour(img_path, lab_path, slice, name=get_name_wo_suffix(img_path) + "_contour_lv_img.png")