import numpy as np
import cv2

def slop_estimation(self):
    # ======== Slope Estimation =========
    print('Slope Estimation START!!!')
    k = int(0.5 * img_w / (max_alt * np.tan(42.2 * np.pi / 180)))
    erosion_k = k // 2

    print('k: ', k, 'erosion_k: ', erosion_k)

    # k, erosion_k = 35, 17
    kernel = np.ones((k,k))/(k*k)
    kernel_erosion = np.ones((erosion_k,erosion_k))/(erosion_k*erosion_k)
    scale = 1.0
    ## GaussianBlur
    # blur = cv2.GaussianBlur(voro_lidar_img,(5,5),0)
    output = self.convolve(voro_lidar_img, kernel, scale)


def convolve(self, image, kernel, scale):
    stride = 4
    image_decreased   = image
    # image_decreased_2 = image
    image_decreased_2 = np.zeros((96, 144))
    # image_decreased_2 = np.zeros((196//2,290//2))
    # image_decreased_2 = np.zeros((196//16,290//16))

    (iH, iW) = image_decreased.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image_decreased, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    
    output = np.zeros((iH, iW), dtype="float32")
    # start = time.time()  # 시작 시간 저장

    for y in np.arange(pad, iH + pad, stride):
        for x in np.arange(pad, iW + pad, stride):

            slope1 = self.slope_func(pad ,image[y-pad,x    ],image[y+pad,x-pad],image[y+pad,x+pad], 1)
            slope2 = self.slope_func(pad ,image[y-pad,x-pad],image[y    ,x-pad],image[y+pad,x-pad], 2)
            slope3 = self.slope_func(pad ,image[y-pad,x-pad],image[y-pad,x+pad],image[y+pad,x    ], 3)
            slope4 = self.slope_func(pad ,image[y-pad,x+pad],image[y    ,x-pad],image[y+pad,x+pad], 4)

            # slope = int((slope1+slope2)/2)
            slope = int((slope1+slope2+slope3+slope4)/4)
            image_decreased_2[(y-pad)//stride-1, (x-pad)//stride - 1] = slope

            # image_decreased_2[y//2-pad,x//2-pad] = slope
            # image_decreased_2[int((y-pad)//2-pad//2),int((x-pad)//2-pad//2)] = slope

        # print('slope: ',  y-pad, '/', iH, slope)

    # print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    output = (image_decreased_2).astype("uint8")
    # output = (image_decreased_2 * 255/90).astype("uint8")
    image_increased = cv2.resize(output, dsize=(img_w,img_h), interpolation=cv2.INTER_AREA)

    return image_increased

    def slope_func(self, pad ,z1, z2, z3, num):

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