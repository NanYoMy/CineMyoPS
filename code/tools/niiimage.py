
import SimpleITK as sitk
'''
所有nii图像的基础类别
'''
class NiiImage():
    def __init__(self,path,mask_path=None):
        self.path=path
        self.sitk_image=sitk.ReadImage(path)
        self.space=self.sitk_image.GetSpacing()
        self.array=sitk.GetArrayFromImage(self.sitk_image)
        if not mask_path:
            self.mask_path=mask_path
            self.array_mask=sitk.GetArrayFromImage(sitk.ReadImage(self.mask_path))
