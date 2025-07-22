import os

from tools.dir import sort_glob


def reorient():
    pathes=sort_glob("../data/alignmscmr_myops_aff_aligned/*41*.nii.gz")
    for path in pathes:
        os.system(f"E:\consistent_workspace\myops20\JRSNet\mmvm\zxhimageop.exe -int {path} -o {path}  -raw")