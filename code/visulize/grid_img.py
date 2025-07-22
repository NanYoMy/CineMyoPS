from PIL import Image,ImageEnhance,ImageOps


# from visulize.brightness_adjustV2 import adjust_brightness_with_gamma
import cv2
import SimpleITK as sitk
import numpy as np
from tools.np_sitk_tools import clipseScaleSitkImage
def _open(p:str):
    if p.endswith("nii.gz"):
        img=sitk.ReadImage(p)
        img=clipseScaleSitkImage(img,0,100)
        array=np.squeeze(sitk.GetArrayFromImage(img))
        new_img=Image.fromarray(array).convert('RGB')
        return new_img
    else:
        return Image.open(p).convert('RGB')



def generate_img_by_row(all_path,numrows,numcols,output,pad=10):
    numimages=len(all_path)
    sampleimage = _open(all_path[0])  # Open first image, just to get dimensions
    width, height = sampleimage.size  # PIL uses (x, y)
    pad = 10
    outimg = Image.new("RGBA", (numcols * width + (numcols - 1) * pad, numrows * height + (numrows - 1) * pad),
                       (0, 0, 0, 0))  # Initialize to transparent
    # Write to output image. This approach copies pixels from the source image
    for i in range(numimages):
        print(f"{numimages}-{i}:{all_path[i]}")

        # img = cv2.imread(all_path[i])
        # bright = adjust_brightness_with_gamma(img)
        # cv2.imwrite(all_path[i], img)

        currimage = _open(all_path[i])


        for k in range(height):
            for j in range(width):
                currimgpixel = currimage.getpixel((j, k))
                # outimg.putpixel(((i % height * width+(i % height)*pad) + j, (i // height * numcols+(i // height)*pad) + k), currimgpixel)
                outimg.putpixel(((i // numrows * width + (i // numrows) * pad) + j,
                                 (i % numrows * height + (i % numrows) * pad) + k), currimgpixel)
    outimg.save(output)

def tranposition(files,numrows,numbcols):
    # 重新排列数组
    mid = len(files) // 2
    first_half = files[:mid]
    second_half = files[mid:]

    # 交替合并列表
    result = []
    for f, s in zip(first_half, second_half):
        result.append(f)
        result.append(s)

    # 打印结果
    print(result)

    return  result,numbcols,numrows

def generate_img(all_path,numrows,numcols,output,pad=10):
    numimages=len(all_path)
    sampleimage = _open(all_path[0])  # Open first image, just to get dimensions
    width, height = sampleimage.size  # PIL uses (x, y)

    outimg = Image.new("RGBA", (numcols * width + (numcols - 1) * pad, numrows * height + (numrows - 1) * pad),
                       (0, 0, 0, 0))  # Initialize to transparent
    # Write to output image. This approach copies pixels from the source image
    for i in range(numimages):
        print(f"{numimages}-{i}:{all_path[i]}")

        # img = cv2.imread(all_path[i])
        # bright = adjust_brightness_with_gamma(img)
        # cv2.imwrite(all_path[i], img)

        currimage = _open(all_path[i])


        for k in range(height):
            for j in range(width):
                currimgpixel = currimage.getpixel((j, k))
                # outimg.putpixel(((i % height * width+(i % height)*pad) + j, (i // height * numcols+(i // height)*pad) + k), currimgpixel)
                outimg.putpixel(((i // numrows * width + (i // numrows) * pad) + j,
                                 (i % numrows * height + (i % numrows) * pad) + k), currimgpixel)
    outimg.save(output)

from tools.np_sitk_tools import sitkResize
def generate_variable_size_img(all_path,numrows,numcols,output,pad=10):
    numimages=len(all_path)
    sampleimage = _open(all_path[0])  # Open first image, just to get dimensions
    s_width, s_height = sampleimage.size  # PIL uses (x, y)
    pad = 10
    outimg = Image.new("RGBA", (numcols * s_width + (numcols - 1) * pad, numrows * s_height + (numrows - 1) * pad),
                       (0, 0, 0, 0))  # Initialize to transparent
    # Write to output image. This approach copies pixels from the source image
    for i in range(numimages):
        print(f"{numimages}-{i}:{all_path[i]}")

        # img = cv2.imread(all_path[i])

        # bright = adjust_brightness_with_gamma(img)
        # cv2.imwrite(all_path[i], img)

        currimage = _open(all_path[i])

        c_w,c_h=currimage.size
        new_width=s_width
        new_heihgt=s_height
        currimage=currimage.resize((new_width,new_heihgt) )



        for k in range(new_heihgt):
            for j in range(new_width):
                currimgpixel = currimage.getpixel((j, k))
                # outimg.putpixel(((i % height * width+(i % height)*pad) + j, (i // height * numcols+(i // height)*pad) + k), currimgpixel)
                outimg.putpixel(((i // numrows * s_width + (i // numrows) * pad) + j,
                                 (i % numrows * s_height + (i % numrows) * pad) + k), currimgpixel)
    outimg.save(output)