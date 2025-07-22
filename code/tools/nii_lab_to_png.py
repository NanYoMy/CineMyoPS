from PIL import Image, ImageOps
import cv2
from skimage import color,segmentation
from tools.np_sitk_tools import clipseScaleSArray
import numpy as np
def save_img(img, outdir, name,  img_size=128, border=0):

    img = Image.fromarray(img).convert('RGB').resize((img_size, img_size))
    # source_points = (source_points + 1) / 2 * 128 + 64
    # source_image = Image.fromarray(img).convert('RGB').resize((128, 128))
    # img=ImageOps.flip(img)# 调整正常的视角
    img.save(f"{outdir}/{name}")

def save_img_with_mask(img,lab,dir,name):
    img=np.squeeze(img)
    lab=np.squeeze(lab)
    img = cv2.resize(img, (500, 500))
    lab = cv2.resize(lab, (500, 500),interpolation=cv2.INTER_NEAREST)
    img = clipseScaleSArray(img, 0, 100).astype('uint8')
    img = color.label2rgb(lab.astype(np.uint8), img, colors=[(255, 255, 0),(0, 255, 255)], alpha=0.001,
                          bg_label=0, bg_color=None)
    # img = clipseScaleSArray(img, 0, 100)
    # img = cv2.flip(img, 0)
    cv2.imwrite(f"{dir}/{name}", img * 255)

def save_img_with_contorus( img, lab, dir, name):

    img=cv2.resize(img, (500, 500))
    lab=cv2.resize(lab, (500, 500),interpolation=cv2.INTER_NEAREST)
    img = clipseScaleSArray(img, 0, 100).astype('uint8')
    img = segmentation.mark_boundaries(img, lab.astype(np.uint8),color=[(1, 0, 0)], mode='thick')
    # img = cv2.flip(img, 0)
    cv2.imwrite(f"{dir}/{name}",img*255 )

def save_img_with_pred_get_contour(img, pred_lab, gt_lab, dir, name):
    pred_lab = cv2.resize(pred_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
    gt_lab = cv2.resize(gt_lab, (500, 500), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (500, 500))
    img = clipseScaleSArray(img, 0, 100).astype(np.uint8)
    img = segmentation.mark_boundaries(img, pred_lab.astype(np.uint8), color=[(0, 0, 1)], mode='thick')
    img = segmentation.mark_boundaries(img, gt_lab.astype(np.uint8), color=[(1, 0, 0)], mode='thick')
    # img = cv2.flip(img, 0)
    cv2.imwrite(f"{dir}/{name}", img * 255)