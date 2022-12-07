import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import argparse
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2
import tensorflow as tf
	


def Color_Choice(color_space,data_lowlight_path):
    #data_lowlight = Image.open(data_lowlight_path).convert(color)
    data_lowlight = cv2.imread(data_lowlight_path)
    data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
    if color_space == "RGB":
        data_lowlight = (np.asarray(data_lowlight)/255.0)
        n = [255,0,255,0,255,0]
        back = cv2.COLOR_RGB2BGR
    else: 
        if color_space == "HSV":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2HSV)
            n = [180,0,255,0,255,0]
            back = cv2.COLOR_HSV2BGR
        elif color_space == "HLS":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2HLS)
            n = [180,0,255,0,255,0]
            back = cv2.COLOR_HLS2BGR
        elif color_space == "YCbCr":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2YCrCb)
            n = [255,0,255,0,255,0]
            back = cv2.COLOR_YCrCb2BGR
        elif color_space == "YUV":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2YUV)
            n = [235-16,16,235-16,16,235-16,16]
            back = cv2.COLOR_YUV2BGR
        elif color_space == "LAB":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2Lab)
            n = [255-1,1,255-1,1,255-1,1]
            back = cv2.COLOR_Lab2BGR
        elif color_space == "Luv":
            data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_RGB2Luv)
            n = [255,0,255,0,255,0]
            back = cv2.COLOR_Luv2BGR
        c1,c2,c3 = cv2.split(data_lowlight)
        data_lowlight_1 = ((c1-n[1])/(n[0])) 
        data_lowlight_2 = ((c2-n[3])/(n[2]))
        data_lowlight_3 = ((c3-n[5])/(n[4]))
        data_lowlight = cv2.merge([data_lowlight_1,data_lowlight_2,data_lowlight_3])       
            
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    return data_lowlight,n,back



def lowlight(color_channel,lowlight_images_path,save_images_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    data_lowlight,con,inchan = Color_Choice(color_channel,lowlight_images_path)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    
    if config.channel=="RGB":
        DCE_net = model.enhance_net_nopool_3().cuda()
    elif config.channel=="HSV":
        DCE_net = model.enhance_net_nopool_1_3().cuda()

    elif config.channel=="HLS":
        DCE_net = model.enhance_net_nopool_1_2().cuda()
    elif config.channel=="YCbCr" or  config.channel=="YUV" or  config.channel=="LAB" or  config.channel=="Luv":
        DCE_net = model.enhance_net_nopool_1_1().cuda()
    
    DCE_net.load_state_dict(torch.load("snapshots/"+color_channel+".pth"))
    start = time.time()
    
    _,enhanced_image,_ = DCE_net(data_lowlight)
    data_lowlight = enhanced_image[0].permute(1,2,0).cpu().numpy()
    temp1 = np.zeros((data_lowlight[:,:,0].shape[0],data_lowlight[:,:,0].shape[1]), dtype="uint8")
    temp2= np.zeros((data_lowlight[:,:,0].shape[0],data_lowlight[:,:,0].shape[1]), dtype="uint8")
    temp3 = np.zeros((data_lowlight[:,:,0].shape[0],data_lowlight[:,:,0].shape[1]), dtype="uint8")
    temp1[:,:] = (data_lowlight[:,:,0]*con[0]+con[1]).astype(dtype="uint8")
    temp2[:,:] = (data_lowlight[:,:,1]*con[2]+con[3]).astype(dtype="uint8")
    temp3[:,:] = (data_lowlight[:,:,2]*con[4]+con[5]).astype(dtype="uint8")

    data_lowlight = cv2.cvtColor(cv2.merge([temp1,temp2,temp3]), inchan)
    result_path = lowlight_images_path.replace('test_data','result')
    if not os.path.exists(result_path.replace('/'+result_path.split("/")[-1],'')):
        os.makedirs(result_path.replace('/'+result_path.split("/")[-1],''))
    cv2.imwrite(result_path,data_lowlight)
    
    end_time = (time.time() - start)
    print(end_time)
    """
    result_path = lowlight_images_path.replace('test_data','result')
    if not os.path.exists(result_path.replace('/'+result_path.split("/")[-1],'')):
        os.makedirs(result_path.replace('/'+result_path.split("/")[-1],''))
        
    torchvision.utils.save_image(enhanced_image, result_path)
    """


if __name__ == '__main__':
# test_images
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowlight_images_path", type=str, default="data/test_data/")
    parser.add_argument("--mode", type=str, default="image")
    parser.add_argument("--channel", type=str, default="RGB")
    parser.add_argument("--save_images_path", type=str, default="data/result")
    config = parser.parse_args()
    with torch.no_grad():
        if config.mode == "image":
            file_path = config.lowlight_images_path
            file_list = os.listdir(config.lowlight_images_path)
            for file_name in file_list:
                test_list = glob.glob(file_path+file_name+"/*") 
                for image in test_list:
				# image = image
                    print(image)
                    lowlight(config.channel,image,config.save_images_path)
        #elif config.mode == "video":


		

