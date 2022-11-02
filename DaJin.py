#encoding=utf-8
import cv2 as cv
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()


img = cv.imread('Harry.jpg', 0)
#Step3. 运行封装OTSU函数并输出灰度化后的阈值
ret, th2 = cv.threshold(img, 0, 255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print(f"Best threshold = {ret}")
show(th2)