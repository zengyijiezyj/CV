# -*- coding: utf-8 -*-
"""
-------------------
Description:
File Name:T1
Author:YJ
Date:2018/8/28
Time:19:12
-------------------
"""
import cv2
import sys

imagePath = sys.argv[1]
cascPath = sys.argv[2]

# 创建级联并使用face cascade初始化它。这会将面部级联加载到内存中，以便可以使用。级联只是一个包含检测面部数据的XML文件。xml中存放的是训练后的特征池
faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 颜色空间转换

faces = faceCascade.detectMultiScale(
    imgGray,
    scaleFactor=1.1,  # 表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%
    minNeighbors=5,  # 构成检测目标的相邻矩形的最小个数
    minSize=(30, 30),  # 目标区域的范围
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE  # 设置值为0或者此值
)

print "You have found {0} faces!".format(len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),
                  2)  # （x，y）是矩阵的左上点坐标 （x+w，y+h）是矩阵的右下点坐标 （0,255,0）是画线对应的rgb颜色 2是所画的线的宽度

cv2.imshow("found", image)
cv2.waitKey(0)
