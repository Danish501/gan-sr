# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:34:45 2023

@author: Danish
"""
import os
import cv2
import config
a=os.listdir("test")

for img in a:
    img_arr=cv2.imread("test/"+img)
    img_arr=cv2.resize(img_arr,(160,120))
    cv2.imwrite("test_new/"+img,img_arr)
    
    
    
    
 
