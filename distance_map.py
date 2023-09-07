import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import time

img_h, img_w   = 384, 576


def main():

    pad = 10
    vor_image_gray = cv2.imread('img1.png',0).astype(np.uint8)   # 3.51m
    cv2.imshow("vor_image_gray", vor_image_gray)

    image_bor = cv2.copyMakeBorder(vor_image_gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    cv2.imshow("image_bor", image_bor)

    

    dist_transform = cv2.distanceTransform(image_bor, cv2.DIST_L2, 5)
    # dist_transform  함수를 사용하면 실수 타입(float32)의 이미지가 생성됩니다. 화면에 보여주려면 normalize 함수를 사용해야 합니다. 
    result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("dist_transform", result)
    cv2.imwrite('image_bor.jpg', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
