
import numpy as np
import cv2
import cv2
from visulize.color import palette

def extract_semantic_contour(label_array):

    # get semantic contours
    label_l = np.unique(label_array)
    contour_d = {}
    for label in label_l:
        l ,c = np.where(label_array == label)
        mask = np.zeros(label_array.shape).astype(np.uint8)
        mask[l ,c] = 255
        # show_img(mask)

        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

        # contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours=smooth_contours(contours)
        # keep only long contours
        big_contours_l = []
        for c in contours:
            if c.shape[0] > 0:
                big_contours_l.append(c)
        if len(big_contours_l ) >0:
            contour_d[label] = np.vstack(big_contours_l)

    return contour_d

def save(array,name):
    pass

def show_img(img,display_size=None):
    if display_size is not None:
        img=cv2.resize(img,display_size)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import numpy
import cv2
from scipy.interpolate import splprep, splev
def smooth_contours(contours):
    smoothened = []
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = numpy.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(numpy.asarray(res_array, dtype=numpy.int32))
    return smoothened

def draw_coutours(img,contours,palette):
    # contours=extract_semantic_contour(lab)
    # cv2.findContours(lab,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours.keys():
        if c not in palette.keys():
            continue
        img = cv2.drawContours(img, contours[c], -1, palette[c], 2)
    return img
from skimage import color,io
import matplotlib.pyplot as plt
from skimage import segmentation
from tools.np_sitk_tools import binarize_numpy_array
from skimage.color.colorlabel import label2rgb
from tools.np_sitk_tools import clipseScaleSArray
def draw_mask(img,lab,palette):
    # contours=extract_semantic_contour(lab)
    # cv2.findContours(lab,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img=np.tile(np.expand_dims(img,axis=-1),[1,1,3])
    # color_lab=np.zeros_like(img)
    # ori_lab=lab
    # lab=np.tile(np.expand_dims(lab,axis=-1),[1,1,3])
    #
    # for c in np.unique(lab):
    #     if c==0:
    #         continue
    #     color_lab[ori_lab==c,:] = palette[c]
    #     over_lap=cv2.addWeighted(img.astype(np.float32),0.8,lab.astype(np.float32),0.2,0)
    #     # img=img*lab+over_lap*(np.where(lab==c,0,1))
    #     img=over_lap
    # return img

    #
    # io.imshow(color.label2rgb(lab.astype(np.uint8), img, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None))
    # plt.show()
    result_image = color.label2rgb(lab.astype(np.uint8), img, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)
    result_image=clipseScaleSArray(result_image,0,100)
    return result_image


def add_label_2_img(img,lab,palette):
    mask=binarize_numpy_array(lab)
    mask=mask.astype(np.uint8)
    color_lab=np.tile(np.expand_dims(np.zeros_like(lab),-1),(1,1,3))

    for k in palette.keys():
        c,l=np.where(lab==k)
        color_lab[c,l]=palette[k]
    img=np.tile(np.expand_dims(img,-1),(1,1,3))
    # img=cv2.bitwise_or(img,color_lab.astype(np.uint8),mask=mask.astype(np.uint8))
    img=img*(1-np.tile(np.expand_dims(mask,-1),(1,1,3)))
    img=cv2.bitwise_or(img,color_lab.astype(np.uint8))
    return  img
