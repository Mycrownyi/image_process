import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()

img = cv.imread('Harry.jpg', 0)

T = img.mean()

while True:
    t0 = img[img < T].mean()
    t1 = img[img >= T].mean()
    t  = (t0 + t1) / 2
    if abs(T - t) < 1:
        break
    T = t
T = int(T)

print(f"Best threshold = {T}")
th, img_bin = cv.threshold(img, T, 255, 0)
show(img_bin)