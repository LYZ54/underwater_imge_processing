import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import color


# # 计算每个通道的平均像素值
def average_color(channel):
    h, w = channel.shape
    sum = 0
    for i in range(h):
        for j in range(w):
            sum = sum + channel[i][j]
    ave = sum / (w * h)
    return ave


# 根据论文公式 进行红 蓝 通道的补偿
def channel_compensate(channel, g):
    h, w = channel.shape
    g_ave = average_color(g)
    now_channel_ave = average_color(channel)
    for i in range(h):
        for j in range(w):
            channel[i][j] = channel[i][j] + (g_ave - now_channel_ave) * (1 - channel[i][j]) * g[i][j]
    return channel


# gray world
def GW(orgImg):
    B, G, R = np.double(orgImg[:, :, 0]), np.double(orgImg[:, :, 1]), np.double(orgImg[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    dst_img = np.uint8(np.zeros_like(orgImg))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


# 改变像素范围
def normalization(channel, new_max, new_min):
    new_channel = (channel - np.min(channel)) * (new_max - new_min) / (np.max(channel) - np.min(channel)) - new_min
    print(new_channel)
    return new_channel


# 对图像进行归一化处理
def image_normalization(img, new_max, new_min):
    b, g, r = cv.split(img)
    b = normalization(b, new_max, new_min)
    g = normalization(g, new_max, new_min)
    r = normalization(r, new_max, new_min)
    img = cv.merge([b, g, r])
    print(img)
    return img


# 总函数 使用该函数进行白平衡
def white_balance(img_path, tag):
    img = cv.imread(img_path, 1)
    img = image_normalization(img, 1, 0)
    h, w, c = img.shape
    b, g, r = cv.split(img)
    r_h, r_w = r.shape
    r_c = channel_compensate(r, g)
    b_c = channel_compensate(b, g)
    g_c = g
    img_compen = cv.merge([b_c, g_c, r_c])
    # img_compen = img_compen * 255
    img_compen = image_normalization(img_compen, 255, 0)
    img_compen = img_compen.astype(np.uint8)
    if tag == 1:
        img_compen = GW(img_compen)  # gray world
    return img, img_compen


img_path = './img_3.jpg'
img, img_compen = white_balance(img_path, tag=1)  # 白平衡
cv.imshow('white_balance', img_compen.astype(np.uint8))
cv.waitKey(0)
