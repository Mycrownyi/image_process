# coding=gbk
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2
from scipy.fftpack import fft

def make_PSF(kernel_size=15, angle=60):
    PSF = np.diag(np.ones(kernel_size))  # ��ʼģ���˵ķ�����-45��
    angle = angle + 45  # ����-45�ȵ�Ӱ��
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)  # ������ת����
    PSF = cv2.warpAffine(PSF, M, (kernel_size, kernel_size), flags=cv2.INTER_LINEAR)  # ʵ����ת�任
    PSF = PSF / PSF.sum()  # ʹģ���˵�Ȩ�غ�Ϊ1
    return PSF


# �˺�����չPSF0��ʹ֮��image0һ����С
def extension_PSF(image0, PSF0):
    [img_h, img_w] = image0.shape
    [h, w] = PSF0.shape
    PSF = np.zeros((img_h, img_w))
    PSF[0:h, 0:w] = PSF0
    return PSF


# ��Ƶ���ͼƬ�����˶�ģ��
def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # ������ͼ����и���Ҷ�任
    PSF_fft = np.fft.fft2(PSF) + eps  # ���˶�ģ���˽��и���Ҷ�任��������һ����С����
    blurred = np.fft.ifft2(input_fft * PSF_fft)  # ��Ƶ������˶�ģ��
    blurred = np.abs(blurred)
    return blurred


#��ȡͼ������غ���
def getAC(image):
    f = np.fft.fft2(image)
    AC = np.fft.ifft2(np.abs(f) ** 2)
    AC = np.fft.fftshift(AC)
    return AC


def inverse(input, PSF, eps):  # ���˲�
    input_fft = np.fft.fft2(input)  # ���˻�ͼ����и���Ҷ�任
    PSF_fft = np.fft.fft2(PSF) + eps  # ���˶�ģ���˽��и���Ҷ�任��������һ����С����
    Output_fft = input_fft / PSF_fft  # ��Ƶ��������˲�
    result = np.fft.ifft2(Output_fft)  # ���и���Ҷ���任
    result = np.abs(result)
    return result


def wiener(input, PSF, eps, K=0.01):  # ��֪����ȵ�ά���˲���kΪ�����
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = (np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)) * input_fft
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result

def wiener1(input, PSF, eps, K=0.01):  # δ֪����ȵ�ά���˲���ʹ�ý��Ʒ�ʽ
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    PSF_fft_1 = (((np.abs(PSF_fft) ** 2) / (np.abs(PSF_fft) ** 2 + K))/ PSF_fft) * input_fft
    result = np.fft.ifft2(PSF_fft_1)
    result = np.abs(result)
    return result

def wiener2(input, PSF, eps,NCORR,ICORR):  # δ֪����ȵ�ά���˲�������δ�˻�ͼ���Լ�����ͼ����غ�������������k
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
    # �����˶�ģ������
    PSF = make_PSF(35, 60)
    # ��չPSF��ʹ����ͼ��һ����С
    PSF = extension_PSF(image, PSF)
    blurred = make_blurred(image, PSF, eps)  # ��Ƶ���ͼ������˶�ģ��

    # �������,standard_normal��������ĺ���
    blurred_noisy = blurred + 0.1 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)
    noisy = 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    plt.figure(figsize=(8, 6))
    plt.subplot(2,3,1)
    plt.axis("off")
    plt.gray(), plt.title("motion & noisy blurred"), plt.imshow(blurred_noisy)  # ��ʾ����������˶�ģ����ͼ��

    result = wiener(blurred_noisy, PSF, eps, K=0.01)  # �����������ͼ������������֪ά���˲�
    plt.subplot(2,3,2)
    plt.axis("off"), plt.title("wiener deblurred(k=0.01)"), plt.imshow(result)


    result = wiener1(blurred_noisy, PSF, eps, K=0.01)  # �����������ͼ����������δ֪��ά���˲�
    plt.subplot(2,3,3)
    plt.axis("off"), plt.title("wiener deblurred(without k)"), plt.imshow(result)

    pic = np.asarray(image)

    #��ȡδ�˻�ͼ�Լ�����ͼ������غ���
    ICORR = getAC(pic)
    NCORR = getAC(noisy)
    result = wiener2(blurred_noisy, PSF, eps, NCORR, ICORR)
    plt.subplot(2,3,4)
    plt.axis("off"), plt.title("wiener deblurred(compute k)"), plt.imshow(result)

plt.show()