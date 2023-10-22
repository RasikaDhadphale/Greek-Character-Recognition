#!/usr/bin/env python
# coding: utf-8

import glob
import re
import cv2
import pandas as pd
import numpy as np


def get_data_tabular():
    # Update the path of dataset
    path = "D:/Study/Final Project/Dataset/Final Datasets/Greek Dataset"
    train_data = pd.read_csv(path + "/train.csv", header=None)
    test_data = pd.read_csv(path + "/test.csv",  header=None)
    X_train = train_data.iloc[:, :196]
    Y_train = train_data.iloc[:, 196]
    X_test = test_data.iloc[:, :196]
    Y_test = test_data.iloc[:, 196]

    return X_train, Y_train, X_test, Y_test


def get_data_gcdb():
    # Update the path of dataset
    path = "D:\Study\Final Project\Dataset\Final Datasets\GCDB\Query"
    pattern = r'LETT_([A-Z]+)_NORM\.([A-Z]+)'
    
    images = []
    label = []
    for folder in glob.glob(path + '\LETT_*_NORM.*') :
        match = re.search(pattern, folder)
        if match:
            temp_label = match.group(2).lower()
        for filename in glob.glob(folder +'\*.bmp'):
            image = cv2.imread(filename)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (40, 40), interpolation=cv2.INTER_AREA)
            images.append(gray_image)
            label.append(temp_label)
            flip_image_0 = cv2.flip(gray_image, 0)
            images.append(flip_image_0)
            label.append(temp_label)
            flip_image_1 = cv2.flip(gray_image, 1)
            images.append(flip_image_1)
            label.append(temp_label)

    data = pd.DataFrame(list(zip(images, label)), columns=["Image", "Label"])
    
    uni_labels = pd.DataFrame({'Label': list(data.Label.unique()), 'Label_id': np.arange(1, 25)})
    
    data = data.merge(uni_labels, how='left', on= 'Label')  

    return data
    
    
    