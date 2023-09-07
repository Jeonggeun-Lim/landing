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
    image = cv2.copyMakeBorder(image_decreased, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    image_bor = cv2.copyMakeBorder(image_decreased, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    print('image.shape: ', image.shape)
    print('image_bor.shape: ', image_bor.shape)
        
        
    output = np.zeros((iH, iW), dtype="float32")
    # start = time.time()  # 시작 시간 저장

    for y in np.arange(pad, iH + pad, stride):
        for x in np.arange(pad, iW + pad, stride):

            slope1 = slope_func(pad ,image[y-pad,x    ],image[y+pad,x-pad],image[y+pad,x+pad], 1)
            slope2 = slope_func(pad ,image[y-pad,x-pad],image[y    ,x-pad],image[y+pad,x-pad], 2)
            slope3 = slope_func(pad ,image[y-pad,x-pad],image[y-pad,x+pad],image[y+pad,x    ], 3)
            slope4 = slope_func(pad ,image[y-pad,x+pad],image[y    ,x-pad],image[y+pad,x+pad], 4)

            # slope = int((slope1+slope2)/2)
            slope = int((slope1+slope2+slope3+slope4)/4)
            image_decreased_2[(y-pad)//stride-1, (x-pad)//stride - 1] = slope

            # image_decreased_2[y//2-pad,x//2-pad] = slope
            # image_decreased_2[int((y-pad)//2-pad//2),int((x-pad)//2-pad//2)] = slope

        print('slope: ',  y-pad, '/', iH, slope)

    # print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    output = (image_decreased_2).astype("uint8")
    # output = (image_decreased_2 * 255/90).astype("uint8")
    image_increased = cv2.resize(output, dsize=(img_w,img_h), interpolation=cv2.INTER_AREA)

    return image_increased, image_bor

# def slope_func(p1, p2, p3):
# def slope_func(pad ,z1, z2, z3):
# def slope_func(v1x ,v1y, v1z, v2x ,v2y, v2z):
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

    return slope


def main():

    # vor_image_gray = cv2.imread('./images_slope_4.47m_video2/voro_lidar_img_2021_11_04_01_52_43_23900.jpg',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('./images_slope_8.0m/voro_lidar_img_2021_11_04_02_26_33_15500.jpg',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('./images_slope/voro_lidar_img_2021_11_17_00_27_15_14200.jpg',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('img1.png',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('depth_map.png',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('voro_lidar_img_2021_07_21_19_07_26_15400.png',0).astype(np.float32)   # 3.51m
    vor_image_gray = cv2.imread('landing_experiment_20230901/voro_lidar_img_2021_07_21_19_07_26_15400.jpg', 0).astype(np.float32)   # 3.51m

    


    # vor_image_gray = cv2.imread('./images_slope_3.51m/voro_lidar_img_2021_11_01_02_31_03_21500.jpg',0).astype(np.float32)   # 3.51m
    # vor_image_gray = cv2.imread('./images_slope_4.57m/voro_lidar_img_2021_11_01_06_39_40_17100.jpg',0).astype(np.float32)   # 4.57m  k=85 
    # vor_image_gray = cv2.imread('./images_slope_3.51m/voro_lidar_img_2021_11_01_02_31_03_21500.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_slope_7.30m/voro_lidar_img_2021_11_01_07_53_43_14800.jpg',0).astype(np.float32)








    # vor_image_gray = cv2.imread('./images_alt_1/voro_lidar_img_2021_09_30_02_20_20_15000.jpg',0).astype(np.float32)   #  5.11m  k=51
    # vor_image_gray = cv2.imread('./images_alt_2/voro_lidar_img_2021_09_30_03_37_50_11050.jpg',0).astype(np.float32)   # 10.00m  k = 19
    # vor_image_gray = cv2.imread('./images_alt_3/voro_lidar_img_2021_09_30_05_03_34_10600.jpg',0).astype(np.float32)   #  6.00m
    # vor_image_gray = cv2.imread('./images_alt_4/voro_lidar_img_2021_09_30_06_29_28_10690.jpg',0).astype(np.float32)   #  7.87m  k = 35
    # vor_image_gray = cv2.imread('./images_alt_5/voro_lidar_img_2021_10_02_04_30_51_10380.jpg',0).astype(np.float32)   #  6.68m  k = 41
    # vor_image_gray = cv2.imread('./images_alt_6/voro_lidar_img_2021_10_02_05_04_31_12350.jpg',0).astype(np.float32)   #  6.17m
    # vor_image_gray = cv2.imread('./images_alt_7/voro_lidar_img_2021_10_02_06_18_57_13210.jpg',0).astype(np.float32)   #  8.47m
    # vor_image_gray = cv2.imread('./images_alt_8/voro_lidar_img_2021_10_03_04_52_16_10810.jpg',0).astype(np.float32)   #  9.13m
    # vor_image_gray = cv2.imread('./images_alt_9/voro_lidar_img_2021_10_03_06_02_39_11130.jpg',0).astype(np.float32)   #  8.20m
    # vor_image_gray = cv2.imread('./images_alt_10/voro_lidar_img_2021_10_03_07_31_58_12830.jpg',0).astype(np.float32)  #  8.57m
    # vor_image_gray = cv2.imread('./images_alt_11/voro_lidar_img_2021_10_04_05_58_00_12380.jpg',0).astype(np.float32)  #  8.24m
    
    # vor_image_gray = cv2.imread('./voro_lidar_img_2021_10_02_10_21_20_00690.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_alt_4/voro_lidar_img_2021_09_30_06_29_29_10710.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_slope_depth_bush/voro_lidar_img_2021_09_17_07_09_45_08660.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_outdoor/voro_lidar_img_2021_09_17_02_09_40_08970.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_slope_depth_on_water/voro_lidar_img_2021_09_19_05_24_33_09630.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_depth_slope/delau_lidar_img_2021_09_20_08_26_43_10530.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_depth_slope/voro_lidar_img_2021_09_20_08_26_48_10630.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_slope/voro_lidar_img_2021_09_22_06_02_42_23490.jpg',0).astype(np.float32)
    # vor_image_gray = cv2.imread('./images_depth_slope/delau_lidar_img_2021_09_20_08_26_43_10540.jpg',0).astype(np.float32)
    cv2.imshow('vor_image', vor_image_gray/255)
    
    ## GaussianBlur
    blur = cv2.GaussianBlur(vor_image_gray,(5,5),0)
    cv2.imshow('blur', blur/255)

    

    ## Padding
    # size: 1.0
    # scale = 1.0
    scale = 1.0
    # k, erosion_k = 17, 9
    # k = 46
    k = 15
    # erosion_k = 17
    erosion_k = k//2
    # erosion_k = k//2

    kernel = np.ones((k,k))/(k*k)
    
    kernel_erosion = np.ones((erosion_k,erosion_k))/(erosion_k*erosion_k)

    output, image_bor = convolve(blur, kernel, scale)
    # # output = convolve(blur*90/255, kernel)

    # cv2.imshow('output', output)

    # # output_thresold = np.where(output < 40.5*2, 0, output)
    # output_thresold = np.where(output < 10, 0, output)
    # cv2.imshow('output_thresold', output_thresold)
    # # output_thresold_erode = cv2.erode(output_thresold, kernel_erosion, iterations=1)
    # output_thresold_opening = cv2.morphologyEx(output_thresold, cv2.MORPH_OPEN, kernel_erosion)

    # aaaaa = np.where(output_thresold_opening < 1, 255, output_thresold_opening)
    # aaaaa = np.where(aaaaa < 200, 0, aaaaa)
    # # aaaaa = np.where(output_thresold_opening < 10, 0, aaaaa)
    # print(aaaaa)


    # # img = cv2.imread('landable_image.png')
    # # gray = cv2.cvtColor(image_bor, cv2.COLOR_BGR2GRAY)
    # # gray = np.pad(aaaaa, (1,1), 'constant', constant_values=0)
    # # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # 이진화된 결과를 dist_transform 함수의 입력으로 사용합니다. 
    # # dist_transform = cv2.distanceTransform(vor_image_gray, cv2.DIST_L2, 5)

    # print(vor_image_gray.shape)
    # print(vor_image_gray[1][1])
    # print(np.info(vor_image_gray))
    
    # blur *= 255
    # blur = blur.astype(np.uint8)

    dist_transform = cv2.distanceTransform(blur, cv2.DIST_L2, 5)
    
    # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다. 
    result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("dist_transform", result)
    # cv2.imshow("src", img)




    # cv2.imshow('output_thresold_erode', aaaaa)

    # cv2.imshow('image_bor', image_bor/255)

    # cv2.imwrite('image_bor.jpg', aaaaa)




    # ## BilateralFilter
    # blur = cv2.bilateralFilter(vor_image_gray,9,75,75)
    # cv2.imshow('blur', blur)

    # vor_image_gray = vor_image_gray*2.5

    ## Gray to 3D
    # https://discourse.matplotlib.org/t/surface-plot-interactive-chart-is-very-slow/21332/3
    # cv2.imwrite('output.jpg', output)
    
    x, y = np.mgrid[0:result.shape[0], 0:result.shape[1]]
    fig = plt.figure(figsize=(11, 11))
    ax = fig.gca(projection='3d')
    # ax.plot_wireframe(x, y, vor_image_gray, rstride=10, cstride=10, cmap=plt.cm.gray, linewidth=1)
    ax.plot_surface(x, y, blur, rstride=10, cstride=10, cmap=plt.cm.gray, linewidth=1)
    ax.view_init(30, -20)
    # ax.view_init(30, 30)
    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
