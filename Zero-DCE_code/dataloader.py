import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):

	image_list_lowlight = glob.glob(lowlight_images_path + "*.JPG")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path, channel):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 256
        self.channel = channel
        
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))
        
    def __getitem__(self, index):
        
        data_lowlight_path = self.data_list[index]
        data_lowlight = cv2.imread(data_lowlight_path)
        data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
        data_lowlight = cv2.resize(data_lowlight,(self.size,self.size),interpolation = cv2.INTER_AREA)
        if self.channel=="RGB":
            data_lowlight = (np.asarray(data_lowlight)/255.0)
        else: 
            if self.channel=="HSV":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2HSV)
                H, S, V = cv2.split(data_lowlight)
                data_lowlight = np.asarray(data_lowlight).copy()
                data_lowlight_1 = ((H)/(180.0)) 
                data_lowlight_2 = ((S)/(255.0))
                data_lowlight_3 = ((V)/(255.0))
            elif self.channel=="HLS":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2HLS)
                H, L, S = cv2.split(data_lowlight)
                data_lowlight = np.asarray(data_lowlight).copy()
                data_lowlight_1 = ((H)/(180.0)) 
                data_lowlight_2 = ((L)/(255.0))
                data_lowlight_3 = ((S)/(255.0))
            elif self.channel=="YCbCr":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2YCrCb)
                Y, Cr, Cb = cv2.split(data_lowlight)
                data_lowlight_1 = ((Y)/(255.0)) 
                data_lowlight_2 = ((Cr)/(255.0))
                data_lowlight_3 = ((Cb)/(255.0))
            elif self.channel=="YUV":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2YUV)
                Y, U, V = cv2.split(data_lowlight)
                data_lowlight_1 = ((Y-16.0)/(235.0-16.0)) 
                data_lowlight_2 = ((U-16.0)/(235.0-16.0))
                data_lowlight_3 = ((V-16.0)/(235.0-16.0))
            elif self.channel=="LAB":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2Lab)
                L, A, B = cv2.split(data_lowlight)
                data_lowlight_1 = ((L-0.0)/(255.0-0.0)) 
                data_lowlight_2 = ((A-1.0)/(255.0-1.0))
                data_lowlight_3 = ((B-1.0)/(255.0-1.0))
            elif self.channel=="LUV":
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2Luv)
                L, U, V = cv2.split(data_lowlight)
                data_lowlight_1 = ((L)/(255.0)) 
                data_lowlight_2 = ((U)/(255.0))
                data_lowlight_3 = ((V)/(255.0))

            data_lowlight = cv2.merge([data_lowlight_1,data_lowlight_2,data_lowlight_3])       
            
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2,0,1)
        return data_lowlight
    def __len__(self):
        return len(self.data_list)
