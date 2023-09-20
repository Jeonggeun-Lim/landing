#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d  import Axes3D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import time

img_h, img_w   = 384, 576

def convolve(image, kernel, scale):

    stride = 1
    image_decreased   = image
    image_decreased_2 = image
    # image_decreased_2 = np.zeros((96, 144))
    # image_decreased_2 = np.zeros((196//2,290//2))
    # image_decreased_2 = np.zeros((196//16,290//16))

    (iH, iW) = image_decreased.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    
    # ave_filter_size = (105, 105)
    # ave_filter_kernel = np.ones(ave_filter_size, dtype=np.float32) / (ave_filter_size[0] * ave_filter_size[1])
    # image = cv2.filter2D(image, -1, ave_filter_kernel)
    
    # image = cv2.GaussianBlur(image, (175,175), 0)
    
    image = cv2.copyMakeBorder(image_decreased_2, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    # start = time.time()  # 시작 시간 저장

    for y in np.arange(pad, iH + pad, stride):
        for x in np.arange(pad, iW + pad, stride):

            slope1 = slope_func(pad ,image[y-pad,x    ],image[y+pad,x-pad],image[y+pad,x+pad], 1)
            slope2 = slope_func(pad ,image[y-pad,x-pad],image[y    ,x-pad],image[y+pad,x-pad], 2)
            slope3 = slope_func(pad ,image[y-pad,x-pad],image[y-pad,x+pad],image[y+pad,x    ], 3)
            slope4 = slope_func(pad ,image[y-pad,x+pad],image[y    ,x-pad],image[y+pad,x+pad], 4)

            # slope = int((slope1+slope3)/2)
            slope = int((slope1+slope2+slope3+slope4)/4)
            image_decreased_2[(y-pad)//stride-1, (x-pad)//stride - 1] = slope

            # image_decreased_2[y//2-pad,x//2-pad] = slope
            # image_decreased_2[int((y-pad)//2-pad//2),int((x-pad)//2-pad//2)] = slope

        print('slope: ',  x-pad, y-pad, '/', iH, slope)

    # print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    output = (image_decreased_2).astype("uint8")
    # output = (image_decreased_2 * 255/90).astype("uint8")
    # image_increased = cv2.resize(output, dsize=(img_w,img_h), interpolation=cv2.INTER_AREA)

    return output

def slope_func(pad ,z1, z2, z3, num):

    if num == 1:
        vec_01 = np.array([ pad, -pad*2, z1-z2])
        vec_02 = np.array([-pad, -pad*2, z1-z3])
    elif num == 2:
        vec_01 = np.array([-2*pad, -1*pad, z1-z2])
        vec_02 = np.array([     0, -2*pad, z1-z3])
    elif num == 3:
        vec_01 = np.array([-2*pad,      0, z1-z2])
        vec_02 = np.array([-1*pad, -2*pad, z1-z3])
    else:
        vec_01 = np.array([ 2*pad, -1*pad, z1-z2])
        vec_02 = np.array([     0, -2*pad, z1-z3])

    xx = vec_01[1]*vec_02[2] - vec_01[2]*vec_02[1]
    yy = vec_01[2]*vec_02[1] - vec_01[0]*vec_02[2]
    zz = vec_01[0]*vec_02[1] - vec_01[1]*vec_02[0]

    normal_vec3_length = np.sqrt((xx ** 2) + (yy ** 2) + (zz ** 2))
    normal_vec2_length = np.sqrt((xx ** 2) + (yy ** 2))

    slope = (abs((np.pi/2 - np.arccos(normal_vec2_length/normal_vec3_length)) * 180.0 / np.pi))

    if slope > 70:
        slope = 0

    return slope

def correct_image(blur, slope_range):
    height, width = blur[slope_range[0]:slope_range[1], :].shape  # y, x
    blur = blur[slope_range[0]:slope_range[1], :]

    for x in range(height):
        for y in range(width):
            blur[x][y] = blur[x][y] - int(25 - x * 25/576)
            if (blur[x][y] > 150):
                blur[x][y] = 0
            print(blur[x][y])

    return blur

def estimate_slope(z):
    ## Padding
    scale = 1.0
    k = 15
    erosion_k = k//2

    kernel = np.ones((k,k))/(k*k)
    kernel_erosion = np.ones((erosion_k,erosion_k))/(erosion_k*erosion_k)
    output = convolve(z, kernel, scale)
    
    total_pixel_value = 0
    count = 0
    h, w = output.shape
    diff = 10
    for y in range(h - diff*2):
        for x in range(w - diff*2):
            if (output[y + diff, x + diff] > 10):
                count = count + 1
                total_pixel_value += output[y + diff, x + diff]
    print("ave: ", total_pixel_value, total_pixel_value / count)
    # print("ave: ", total_pixel_value, total_pixel_value / (img_h * img_w))
    
    return output

def main():
    
    vor_image_gray = cv2.imread('images_10/voro_lidar_img_2023_09_16_06_41_42_01800.jpg', 0).astype(np.uint8)   # 10 deg
    # slope_range = (50, 230)   # 10 deg
    # vor_image_gray = cv2.imread('images_20/voro_lidar_img_2023_09_16_06_54_29_02000.jpg', 0).astype(np.uint8)   # 20 deg
    # slope_range = (0, 200)   # 20 deg
    # vor_image_gray = cv2.imread('images_30/voro_lidar_img_2023_09_16_07_04_35_02300.jpg', 0).astype(np.uint8)   # 30 deg
    # slope_range = (50, 200)   # 30 deg
    # vor_image_gray = vor_image_gray[0:400][:]
    vor_image_gray = vor_image_gray[0:400, 0:400]

    
    # img_h, img_w = vor_image_gray.shape
    # for y in range(img_h):
    #     vor_image_gray[y, :] = vor_image_gray[y, 100]
    slope_range = (0, 384)
    
    blur = cv2.GaussianBlur(vor_image_gray, (45,45), 0)
    z = correct_image(blur, slope_range)
    # z = estimate_slope(z)
    z = cv2.GaussianBlur(z, (45,45),0)
    z = cv2.resize(z, dsize=(576,384), interpolation=cv2.INTER_AREA)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.5])

    img_h, img_w = z.shape
    x = np.arange(0, img_w, 1)
    y = np.arange(0, img_h, 1)
    x, y = np.meshgrid(x, y)

    # np.savetxt('slope.txt', blur[1][y], fmt='%d', delimiter='/t')
    # z = cv2.GaussianBlur(vor_image_gray,(45,45),0)

    ax.plot_surface(x, y, z, cmap='gray')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Slope')

    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_zlim(0, np.max(z))
    # ax.set_zlim(0, 150)

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
