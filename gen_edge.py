import cv2
import numpy as np
from tqdm import tqdm
import os

src_gt = ''
src_edge = ''
for image_name in tqdm(os.listdir(src_gt)):
    gt = cv2.imread(src_gt + image_name)

    # 使用算子

    # 形态学：边缘检测
    _, Thr_img = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 定义矩形结构元素
    gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度

    cv2.imwrite(src_edge + image_name, gradient)
