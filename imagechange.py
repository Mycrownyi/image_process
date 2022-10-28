# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图片
src = cv2.imread('sheep.jpg')
image = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)

# 原图的高、宽 以及通道数
rows, cols, channel = image.shape

# 绕图像的中心旋转
# 参数：旋转中心 旋转度数 scale
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
# 参数：原始图像 旋转参数 元素图像宽高
img1 = cv2.warpAffine(image, M, (cols, rows))


#图像平移
M = np.float32([[1, 0, 0], [0, 1, 100]])
img2 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


#图像翻转
img3 = cv2.flip(image, 0)


#复合变换
img4 = cv2.flip(cv2.warpAffine(img1, M, (image.shape[1], image.shape[0])),0)

#显示图形
titles = [ 'Image1', 'Image2', 'Image3', 'Image4']
images = [img1, img2, img3, img4]
for i in range(4):
   plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
