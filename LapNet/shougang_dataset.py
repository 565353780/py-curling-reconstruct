import torch
from torch.utils import data

import warnings

import numpy as np
import os
import cv2
import glob
import time


class ShougangDataset(data.Dataset):
    # refer from : 
    # https://github.com/vxy10/ImageAugmentation
    # https://github.com/TuSimple/tusimple-benchmark/blob/master/example/lane_demo.ipynb
    def __init__(self, file_path, size=[320, 160], train=True):
        warnings.simplefilter("ignore")

        self.width = size[0]
        self.height = size[1]
        #"D:\datasets\CULANE\**\*.jpg"
        self.rng = np.random.RandomState(int(time.time()))
        self.path_list = glob.glob(file_path+"/**/*.jpg", recursive=True)
        self.rng.shuffle(self.path_list)
        
        self.flags = {'size':size, 'train':train}

        self.img = np.zeros(size, np.uint8)
        
        self.label_img = np.zeros(size, np.uint8)
        self.ins_img = np.zeros((0,size[0],size[1]), np.uint8)
        
        self.len = len(self.path_list)
        self.mainpath=file_path
        print(self.len)
    
    def next(self,path):


        img_path = path+".jpg"
        ins_path = path+".png"
        # print(path)
        # print(img_path)

        frame = cv2.imread(img_path)

        self.rng = np.random.RandomState(int(time.time()))
        seg_g = cv2.imread(ins_path,cv2.IMREAD_GRAYSCALE)
        # self.rng = np.random.RandomState(int(time.time()))
        if ((frame is None) or (seg_g is None)):
            print("Failed to read:",img_path)
            #./ShougangDataset/shougang20190504/im4in25id8935708126.jpg
            frame = cv2.imread(self.mainpath+"/failsafe.jpg")
            seg_g = cv2.imread(self.mainpath+"/failsafe.png",cv2.IMREAD_GRAYSCALE)

        gamma = self.rng.uniform(0.8,1.4)
        gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        #实现映射用的是Opencv的查表函数
        frame = cv2.LUT(frame,gamma_table)

        p = self.rng.uniform(0.0,1.0)
        
        if (p < 0.5):
            frame = cv2.flip(frame, -1)
            seg_g = cv2.flip(seg_g, -1)
                       
        frame = cv2.resize(frame,(self.width,self.height),interpolation=cv2.INTER_AREA)
        seg_g = cv2.resize(seg_g,(self.width,self.height),interpolation=cv2.INTER_AREA)
        frame_h, frame_w, _ = frame.shape
        crop_factor_h = self.rng.uniform(0.0, 0.01)
        crop_factor_w = self.rng.uniform(0.0, 0.01)
        h = frame_h - frame_h*crop_factor_h
        w = frame_w - frame_w*crop_factor_w
        x = self.rng.uniform(0, int(frame_w - w))
        y = int(frame_h - h)//2
        # y = self.rng.uniform(y-3, y+3)
        #self.rng.uniform(int(frame_h - h/2-1), int(frame_h - h/2+1))
        crop = np.array([y,y+h,x,x+w]).astype('int')
        frame = frame[crop[0]:crop[1],crop[2]:crop[3]]
        seg_g = seg_g[crop[0]:crop[1],crop[2]:crop[3]]

        seg_b = np.zeros((self.height,self.width), dtype=np.float32)
        seg_i = list()
        seg_b = cv2.normalize(seg_g,seg_g,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # # for i in range(seg_g.max()):
        #     _bin = np.where(seg_g == i, 1, 0).astype(np.uint8)
        #     _bin = cv2.resize(_bin,(self.width,self.height),interpolation=cv2.INTER_NEAREST)
        #     __bin = np.zeros((self.height,self.width), dtype=np.float32)
        #     if(i != 0):
        #         contours,hierarchy = cv2.findContours(_bin, 1, 2)
        #         if len(contours):    
        #             con = contours[0]
        #             for c in contours:
        #                 con = np.vstack((con,c))
        #             x,y,w,h = cv2.boundingRect(con)
        #             center = (int(x+w/2),int(y+h/2))
                    
        #             cv2.circle(seg_b,center,1,color = (1), thickness=-1, lineType=8)
        #             py,px = (int(y+h/2),int(x+w/2))
        #             #seg_b[py+(-1),px+(-1)] = 0.453542 * 2 #px+(-1),py+(-1)
        #             #seg_b[py+(-1),px+(0)] = 0.566406 * 2 #px+(-1),py+(0)
        #             #seg_b[py+(-1),px+(1)] = 0.453542 * 2 #px+(-1),py+(1)
        #             #seg_b[py+( 0),px+(-1)] = 0.566406 * 2 #px+( 0),py+(-1)
        #             #seg_b[py+( 0),px+(0)] = 0.707355 * 2 #px+( 0),py+(0)
        #             #seg_b[py+( 0),px+(1)] = 0.566406 * 2 #px+( 0),py+(1)
        #             #seg_b[py+( 1),px+(-1)] = 0.453542 * 2 #px+( 1),py+(-1)
        #             #seg_b[py+( 1),px+(0)] = 0.566406 * 2 #px+( 1),py+(0)
        #             #seg_b[py+( 1),px+(1)] = 0.453542 * 2 #px+( 1),py+(1)
        #             cv2.circle(__bin,center,2,color = (1), thickness=-1, lineType=16)
        #             seg_b[py+( 0),px+(0)] = 1
        #     seg_i.append(cv2.resize(__bin,(self.width//2,self.height//2),interpolation=cv2.INTER_NEAREST))
        #     # = cv2.GaussianBlur(seg_b,(3,3),0)
        # seg_i = np.array(seg_i)        

         
        # print(frame.shape)
        # print(seg_g.shape)
        # print(self.width,self.height)
        frame = cv2.resize(frame,(self.width,self.height),interpolation=cv2.INTER_AREA)
        seg_b = cv2.resize(seg_b,(self.width,self.height),interpolation=cv2.INTER_NEAREST)
        seg_view = seg_b

        
        
            
        # #stage 1 random ROI
        # #// 去除OSD区域和车引擎盖
        # frame = frame[0:frame_h-100,:]
        # seg_b = seg_b[0:frame_h-100,:]
        # frame_h,frame_w,_ = frame.shape
        # canvas = frame
        
        # #CROP ROI
        # crop_factor_h = self.rng.uniform(0.15, 0.55)
        # crop_factor_w = self.rng.uniform(0.10, 0.4)
        # h = frame_h - frame_h*crop_factor_h
        # w = frame_w - frame_w*crop_factor_w
        # x = self.rng.uniform(int(frame_w - w)*0.2, int(frame_w - w)*0.8)
        # y = frame_h*crop_factor_h
        # crop = np.array([y,y+h,x,x+w]).astype('int')
        # # print(np.shape(frame),np.shape(seg_b))
        # frame = cv2.resize(frame[crop[0]:crop[1],crop[2]:crop[3]],(self.width,self.height),interpolation=cv2.INTER_AREA)
        # seg_b = cv2.resize(seg_b[crop[0]:crop[1],crop[2]:crop[3]],(self.width,self.height),interpolation=cv2.INTER_AREA)
        # # seg_b = cv2.morphologyEx(seg_b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # line_thick_grow = 5
        # seg_b = cv2.morphologyEx(seg_b, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thick_grow, line_thick_grow)))
        # _, seg_b = cv2.threshold(seg_b, 0, 1, cv2.THRESH_BINARY)

        # for i, seg_i_ in enumerate(seg_i):
        #     seg_i_ = cv2.resize(seg_i_[crop[0]:crop[1],crop[2]:crop[3]],(self.width,self.height),interpolation=cv2.INTER_AREA)
        #     seg_i2[i] = seg_i_
        # if len(seg_i2) == 0:
        #     seg_i2 = np.zeros((1,self.height, self.width), dtype=np.uint8)
        # stage 1.5 random white balance shift
        #frame += Scalar(rng.uniform(-20, 20), self.rng.uniform(-10, 10), self.rng.uniform(-10, 10));C++API
        
        # stage 2 get GT
        # seg_b = cv2.morphologyEx(seg_b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1)))
        # line_thick_grow = 5
        # seg_b = cv2.morphologyEx(seg_b, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thick_grow, line_thick_grow)))
        # _, seg_b = cv2.threshold(seg_b, 0, 1, cv2.THRESH_BINARY)

        # seg_b = cv2.morphologyEx(seg_b, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thick_grow, line_thick_grow)))
        # _, seg_b = cv2.threshold(seg_b, 0, 1, cv2.THRESH_BINARY)


        # view
        # canvas = cv2.rectangle(canvas, (crop[2],crop[0]), (crop[3],crop[1]), (255,255,0), 2)
        # cv2.imshow("canvas",canvas)
        # cv2.imshow("src",frame)
        # cv2.imshow("seg",seg_view*255)
        # # cv2.waitKey(1)
        # seg_show = cv2.applyColorMap(seg_b*255,cv2.COLORMAP_OCEAN)
        # cv2.imshow("GT",cv2.addWeighted(seg_show,0.5,frame,0.5,0))
        # cv2.waitKey(1)
       
        return frame,seg_b
    def __getitem__(self, idx):
        
        self.path_from = self.path_list[idx][:-4]
        self.img, self.label_img= self.next(self.path_from)

        if self.flags['train']:
            #self.random_transform()
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            self.label_img = np.array(self.label_img, dtype=np.int64)
            #self.ins_img = np.array(self.ins_img, dtype=np.float32)

            #print(np.shape(self.img),np.shape(self.label_img),np.shape(self.ins_img),len(self.ins_img))
            return torch.Tensor(self.img), torch.LongTensor(self.label_img)
        else:
            self.img = np.array(np.transpose(self.img, (2,0,1)), dtype=np.float32)
            return torch.Tensor(self.img)
    
    def __len__(self):
        return self.len
