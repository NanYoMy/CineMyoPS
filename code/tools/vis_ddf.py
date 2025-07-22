from tools.itkdatawriter import sitk_write_images
import os
import SimpleITK as sitk

def save_ddf(input_,parameter_img,dir,name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if dir is not None:

        if not isinstance(input_,sitk.Image):
            # if len(input_.shape)==2:
            #     input_=np.expand_dims(input_,axis=0)
            img = sitk.GetImageFromArray(input_,isVector=True)
        else:
            img=input_
        if parameter_img is not None:
            img.CopyInformation(parameter_img)
        sitk.WriteImage(img, os.path.join(dir, name+'.nii.gz'))
    return os.path.join(dir, name+'.nii.gz')