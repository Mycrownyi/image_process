# coding=gbk
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2
from scipy.fftpack import fft

def make_PSF(kernel_size=15, angle=60):
    PSF = np.diag(np.ones(kernel_size))  # 初始模糊核的方向是-45度
    angle = angle + 45  # 抵消-45度的影响
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)  # 生成旋转算子
    PSF = cv2.warpAffine(PSF, M, (kernel_size, kernel_size), flags=cv2.INTER_LINEAR)  # 实现旋转变换
    PSF = PSF / PSF.sum()  # 使模糊核的权重和为1
    return PSF


# 此函数扩展PSF0，使之与image0一样大小
def extension_PSF(image0, PSF0):
    [img_h, img_w] = image0.shape
    [h, w] = PSF0.shape
    PSF = np.zeros((img_h, img_w))
    PSF[0:h, 0:w] = PSF0
    return PSF


# 在频域对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # 对输入图像进行傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps  # 对运动模糊核进行傅里叶变换，并加上一个很小的数
    blurred = np.fft.ifft2(input_fft * PSF_fft)  # 在频域进行运动模糊
    blurred = np.abs(blurred)
    return blurred


#获取图像自相关函数
def getAC(image):
    f = np.fft.fft2(image)
    AC = np.fft.ifft2(np.abs(f) ** 2)
    AC = np.fft.fftshift(AC)
    return AC


def inverse(input, PSF, eps):  # 逆滤波
    input_fft = np.fft.fft2(input)  # 对退化图像进行傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps  # 对运动模糊核进行傅里叶变换，并加上一个很小的数
    Output_fft = input_fft / PSF_fft  # 在频域进行逆滤波
    result = np.fft.ifft2(Output_fft)  # 进行傅里叶反变换
    result = np.abs(result)
    return result


def wiener(input, PSF, eps, K=0.01):  # 已知信噪比的维纳滤波，k为信噪比
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = (np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)) * input_fft
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result

def wiener1(input, PSF, eps, K=0.01):  # 未知信噪比的维纳滤波，使用近似方式
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = (((np.abs(PSF_fft) ** 2) / (np.abs(PSF_fft) ** 2 + K))/ PSF_fft) * input_fft
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result

def wiener2(input, PSF, eps,NCORR,ICORR):  # 未知信噪比的维纳滤波，利用未退化图像以及噪声图自相关函数功率谱来求k
    prx1 = np.abs(np.array(np.fft.fft2(NCORR)))**2
    prx2 = np.abs(np.array(np.fft.fft2(ICORR)))**2
    K = prx1/prx2*16876
    # K = prx1 / prx2*1000
    # print(K)

    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = (((np.abs(PSF_fft) ** 2) / (np.abs(PSF_fft) ** 2 + K))/ PSF_fft) * input_fft
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result

if __name__ == "__main__":
    image = cv2.imread(r'cat.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eps = 1e-3
    # plt.figure(1)
    # 进行运动模糊处理
    PSF = make_PSF(35, 60)
    # 扩展PSF，使其与图像一样大小
    PSF = extension_PSF(image, PSF)
    blurred = make_blurred(image, PSF, eps)  # 在频域对图像进行运动模糊

    # 添加噪声,standard_normal产生随机的函数
    blurred_noisy = blurred + 0.1 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)
    noisy = 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    plt.figure(figsize=(8, 6))
    plt.subplot(2,3,1)
    plt.axis("off")
    plt.gray(), plt.title("motion & noisy blurred"), plt.imshow(blurred_noisy)  # 显示添加噪声且运动模糊的图像

    result = wiener(blurred_noisy, PSF, eps, K=0.01)  # 对添加噪声的图像进行信噪比已知维纳滤波
    plt.subplot(2,3,2)
    plt.axis("off"), plt.title("wiener deblurred(k=0.01)"), plt.imshow(result)


    result = wiener1(blurred_noisy, PSF, eps, K=0.01)  # 对添加噪声的图像进行信噪比未知的维纳滤波
    plt.subplot(2,3,3)
    plt.axis("off"), plt.title("wiener deblurred(without k)"), plt.imshow(result)

    pic = np.asarray(image)

    #获取未退化图以及噪声图的自相关函数
    ICORR = getAC(pic)
    NCORR = getAC(noisy)
    result = wiener2(blurred_noisy, PSF, eps, NCORR, ICORR)
    plt.subplot(2,3,4)
    plt.axis("off"), plt.title("wiener deblurred(compute k)"), plt.imshow(result)

plt.show()