# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:16:52 2020

@author: LEE
"""
import cv2
import numpy as np

def imshow(windowname,image):
    cv2.imshow(windowname,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_circles_demo(image):
    '''
    输入一幅图片
    
    返回画好圆心的原图
    '''
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)  # 均值偏移滤波
#    cv2.imshow('dst',dst)
    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('cimage',cimage)
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 200, 
                               param1=30, param2=80, minRadius=0, maxRadius=0)
    # 整数化，#把circles包含的圆心和半径的值变成整数
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画出外边圆
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 画出圆心
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 0), 2)
#    cv2.imshow("circles", image)
    return image

def edge_demo(image,color_edge=False):
    '''
    输入一幅图片
    
    返回边缘图片
    '''
    image = cv2.blur(image, (3, 3))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度     
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv2.Canny(gray, 50, 150)
    cv2.imshow("Canny Edge", edge_output)
    if color_edge:
        dst = cv2.bitwise_and(image, image, mask= edge_output)
        cv2.imshow("Color Edge", dst)
    return edge_output

def color_extract_demo(image,lower=[0,0,0],upper=[180,255,255],color='custom'):
    '''
    image：原图
    lower: HSV空间最低值
    upper: HSV空间最高值
    
    return：原图指定颜色区域
    '''
    color_dic = {'black':[[0,0,0],[180,255,46]],'gray':[[0,0,46],[180,43,220]],
           'white':[[0,0,221],[180,30,255]],'orange':[[11,43,46],[25,255,255]],
           'yellow':[[26,43,46],[34,255,255]],'green':[[35,43,46],[99,255,255]],
           'blue':[[100,43,46],[124,255,255]],'purple':[[125,43,46],[155,255,255]],
           'custom':[lower,upper]}
    
    if color == 'red':
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        
        lower_red = [np.array([0,43,46]),np.array([156,43,46])]
        upper_red = [np.array([10,255,255]),np.array([180,255,255])]
        
        mask_red0 = cv2.inRange(hsv,lower_red[0],upper_red[0])
        mask_red1 = cv2.inRange(hsv,lower_red[1],upper_red[1])
        mask_red = cv2.add(mask_red0,mask_red1)
        inv_mask = cv2.bitwise_not(mask_red)
        inv_mask = cv2.cvtColor(inv_mask,cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.bitwise_and(hsv,hsv,mask=mask_red)
        
        mask_color = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        mask_color = cv2.add(mask_color,inv_mask)
    else:
        lower = color_dic[color][0]
        upper = color_dic[color][1]
        
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_color = np.array(lower)
        upper_color = np.array(upper)
        
        mask_color = cv2.inRange(hsv,lower_color,upper_color) #范围内的像素点置255
        inv_mask = cv2.bitwise_not(mask_color)
        inv_mask = cv2.cvtColor(inv_mask,cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.bitwise_and(hsv,hsv,mask=mask_color)  
           
        mask_color = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        mask_color = cv2.add(mask_color,inv_mask)
    return mask_color

def color_mask_demo(image,lower=[0,0,0],upper=[180,255,255],color='custom'):
    '''
    :param image:原图像
    :param lower:最低颜色阈值
    :param upper:最高颜色阈值
    :param color:选择已设定好的颜色

    :return:返回去除颜色后的图像
    '''
    color_dic = {'black': [[0, 0, 0], [180, 255, 46]], 'gray': [[0, 0, 46], [180, 43, 220]],
                 'white': [[0, 0, 221], [180, 30, 255]], 'orange': [[11, 43, 46], [25, 255, 255]],
                 'yellow': [[26, 43, 46], [34, 255, 255]], 'green': [[35, 43, 46], [99, 255, 255]],
                 'blue': [[100, 43, 46], [124, 255, 255]], 'purple': [[125, 43, 46], [155, 255, 255]],
                 'custom': [lower, upper]}

    if color == 'red':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = [np.array([0, 43, 46]), np.array([156, 43, 46])]
        upper_red = [np.array([10, 255, 255]), np.array([180, 255, 255])]

        mask_red0 = cv2.inRange(hsv, lower_red[0], upper_red[0])
        mask_red1 = cv2.inRange(hsv, lower_red[1], upper_red[1])
        mask_red = cv2.add(mask_red0, mask_red1)

        inv_mask = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)

        mask_color = cv2.bitwise_or(image, inv_mask)
    else:
        lower = color_dic[color][0]
        upper = color_dic[color][1]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_color = np.array(lower)
        upper_color = np.array(upper)

        mask_color = cv2.inRange(hsv, lower_color, upper_color)  # 范围内的像素点置255

        inv_mask = cv2.cvtColor(mask_color, cv2.COLOR_GRAY2BGR)

        mask_color = cv2.bitwise_or(image, inv_mask)
    return mask_color


def sobel_egde_demo(image):
    '''
    image:原图或者灰度化图片

    return：三通道或二值化边界图片
    '''
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回unit8
    absY = cv2.convertScaleAbs(y)

    img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return img

def adjust_imshow(namewidonws,image):
    '''
    namewidows:窗口名字
    image:图片
    '''
    cv2.namedWindow(namewidonws,0)
    cv2.imshow(namewidonws,image)

def high_contrast(image,iterations=1):
    '''
    高反差保留算法
    :param image: 原图片
    :param iterations: 迭代次数

    :return: 高反差保留后的原图片
    '''
    img = cv2.GaussianBlur(image,(5,5),1)
    temp = cv2.subtract(image,img)
    for i in range(iterations):
        result = cv2.addWeighted(image,1,temp,1,0)
    return result

def get_contours_rectangle_demo(image,th=0):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,127,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    _,cnts,hir = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    record = [0]
    temp = 0
    for i in range(len(cnts)):
        c = cnts[i]
        if cv2.contourArea(c) > temp:
            temp = cv2.contourArea(c)
            record = c
    img_size = img.shape[0] * img.shape[1]
    if len(record) and temp < 0.95*img_size and temp > 0.01*img_size:
        '待优化'
        x,y,w,h = cv2.boundingRect(record)
        bottom_right = (x+w,y+h)
        # cv2.rectangle(img,(x,y),bottom_right,[255,255,255]) #画出目标区域
        xmin = max(x - th, 0)
        xmax = min(x+w + th, image.shape[1])
        ymin = max(0, y - th)
        ymax = min(y+h + th, image.shape[0])
        img = image[ymin:ymax,xmin:xmax]
    else:
        img = image
    return img
#def detect_lines_demo(image):
#    '''
#    image: 原图
#    
#    return: 画了直线的原图
#    '''
#    image = 