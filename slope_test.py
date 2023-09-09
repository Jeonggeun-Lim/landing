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

    if slope >30:
        slope = 0

    return slope


def main():
    vor_image_gray = cv2.imread('landing_experiment_20230901/voro_lidar_img_2021_07_21_19_07_26_15400.jpg', 0).astype(np.uint8)   # 3.51m
    # cv2.imshow('vor_image', vor_image_gray/255)

    ## GaussianBlur
    blur = cv2.GaussianBlur(vor_image_gray,(15,15),0)
    cv2.imshow('blur', blur/255)

    # ## Padding
    scale = 1.0
    k = 90
    erosion_k = k//2

    kernel = np.ones((k,k))/(k*k)
    kernel_erosion = np.ones((erosion_k,erosion_k))/(erosion_k*erosion_k)
    output, image_bor = convolve(vor_image_gray, kernel, scale)

    # dist_transform = cv2.distanceTransform(blur, cv2.DIST_L2, 5)  

    # # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다. 
    # result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # # cv2.imshow("dist_transform", result)


    ## Gray to 3D
    # https://discourse.matplotlib.org/t/surface-plot-interactive-chart-is-very-slow/21332/3
    # cv2.imwrite('output.jpg', output)

    # x, y = np.mgrid[0:blur.shape[0], 0:blur.shape[1]]
    # fig = plt.figure(figsize=(11, 11))
    # ax = fig.gca(projection='3d')
    # # ax.plot_wireframe(x, y, blur, rstride=10, cstride=10, cmap=plt.cm.gray, linewidth=1)
    # ax.set_box_aspect([1, 1, 1])  # x, y, z 축 스케일을 동일하게 설정
    # ax.plot_surface(x, y, blur, rstride=10, cstride=10, cmap=plt.cm.gray, linewidth=1)
    # ax.view_init(30, -20)
    # # ax.view_init(30, 30)
    # plt.show()

# z = blur
    height, width = blur.shape

    for x in range(height):
        for y in range(width):
            # print(blur[x][y])
            blur[x][y] = blur[x][y]  - int(50 - x * 40/350)
            if (blur[x][y] > 50):
                blur[x][y] = 0
        print(blur[x][y])
        # print(blur[x][y])
    z = blur

    
    grid_size = 10
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    # z = np.linspace(0, 1, grid_size)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_box_aspect([1, 0.5, 0.05])

    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    x, y = np.meshgrid(x, y)
    

    ax.plot_surface(x, y, z, cmap='gray')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Gray Value')

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, np.max(z))

    plt.show()



    # ax.plot(x, y, blur)
    # ax.set_box_aspect([1, 1, 1])  # x, y, z 축 스케일을 동일하게 설정

    # plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
