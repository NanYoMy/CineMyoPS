from skimage import io,data,color
import numpy as np
import cv2
img=cv2.imread('../../data/test/a.jpg')
gray=color.rgb2gray(img)
rows,cols=gray.shape
labels=np.zeros([rows,cols])
for i in range(rows):
    for j in range(cols):
        if(gray[i,j]<0.4):
            labels[i,j]=-1
        elif(gray[i,j]<0.75):
            labels[i,j]=-1
        else:
            labels[i,j]=2
dst=color.label2rgb(labels,img,alpha=1)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()