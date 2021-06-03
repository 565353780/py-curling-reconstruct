import numpy as np
import os
import argparse

import cv2
import torch
from LapNet import LAPNet

import mmap
import time



parser = argparse.ArgumentParser(description="InferenceBroker")
# Mmem config:
parser.add_argument('--batch',type=int, required=True)
parser.add_argument('--topic',type=str,required=True)
parser.add_argument('--h',type=int,required=True)
parser.add_argument('--w',type=int,required=True)
# Model config:
parser.add_argument('--model',type=str,required=True)
parser.add_argument('--ch',type=int,required=True)
# gpuid config:
parser.add_argument('--gpuidx',type=str,required=True)
# setup args vars
args = parser.parse_args()

# gpumask
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuidx

# setup vars
FILE_MAP_ALL_ACCESS  = 0xF001F
PAGE_READWRITE       = 0x04
INVALID_HANDLE_VALUE = 0xFFFFFFFF
TENSOR_INDX          = 1
TENSOR_SIZE          = args.batch * 3 * args.h * args.w * 4
MEMMAP_SIZE          = TENSOR_SIZE * 2
OUTPUT_INDX          = TENSOR_INDX + TENSOR_SIZE + 10240
OnMaster             = 0
OnSlave              = 1
Quit                 = 0x80

# indicator
class AnimateLoop:
    def __init__(self):
        self.string_animate_loop = list(["/","-","\\","|"])
        self.animate_idx = 0

    
    def animate_loop(self):
        self.animate_idx += 1
        if (self.animate_idx >= len(self.string_animate_loop)):
            self.animate_idx = 0
        return self.string_animate_loop[self.animate_idx]

# load model dict
model = LAPNet(input_ch=3,output_ch=2,internal_ch = args.ch,one_channel = True).cuda()
model.load_state_dict(torch.load(args.model)['net'],strict=False)
model.train(False)

###########################################################################################
def inference_pipeline(input_tensor):
    with torch.no_grad():
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        seg_maps = model(input_tensor)
        seg_maps = seg_maps.cpu().numpy()
    return seg_maps

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def postproc(seg_maps):
    ret = []
    ret.append(len(seg_maps))                                               # batch_sz
    centers = []
    
    for seg_map in seg_maps:
        _, seg_view = cv2.threshold(seg_map+100, 0, 0,cv2.THRESH_TOZERO)
        seg_view = cv2.normalize(seg_view,seg_view,0,1,cv2.NORM_MINMAX)
        seg_view = cv2.convertScaleAbs(seg_view,seg_view,255)
        seg_view = cv2.applyColorMap(seg_view,cv2.COLORMAP_PARULA)
        cv2.imshow("heatmap",seg_view)





        _,seg_ret = cv2.threshold(seg_map+2,0,255,cv2.THRESH_BINARY)
        seg_ret = cv2.convertScaleAbs(seg_ret)
        cv2.imshow(args.topic+"bin",seg_ret)
        cv2.waitKey(1)
        c,_ = cv2.findContours(seg_ret,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        ret.append(len(c))                                              # vec_sz
        # print("vecsz:",len(c))
        _ = []
        for cn in c:
            _.append(list(cv2.minAreaRect(cn))[0])        #  [center (x,y), (width, height), angle of rotation]
        centers.append(_)
    ret.append(list(flat(centers)))
    return np.array(list(flat(ret)),dtype = np.float32)

if __name__ == "__main__":

    # print("=GPU",args.gpuidx,"===================================================")

    # print("caption:",args.topic)
    ctx = mmap.mmap(-1,MEMMAP_SIZE,args.topic,access=(mmap.ACCESS_WRITE))
    print(len(ctx))

    a = AnimateLoop()

    
    while(ctx[0] != OnSlave):
        time.sleep(0.033)
        print("\r"+ a.animate_loop() +"Waiting for master proc sync...",end = "")
    print("Connected!")

    # print(">>MainLoop:")
    
    while(ctx[0] != Quit):
        while(ctx[0] != OnSlave):
            time.sleep(0.033)
            if(ctx[0] == Quit):
                exit(0)
        blob = np.array(np.frombuffer(ctx[1 : TENSOR_INDX + TENSOR_SIZE],dtype = np.float32),dtype = np.float32).reshape(args.batch,3 ,512,1024,)
        tensor = torch.from_numpy(blob).cuda()
        out = inference_pipeline(tensor)
        ret = postproc(out)
        
        # print(ret[:int(ret[0])+1].astype(np.uint64).tobytes())
        
        ret_sz = ret[1:int(ret[0])+1].astype(np.uint64).tobytes()
        ret_pts = ret[int(ret[0])+1:].astype(np.float32).tobytes()
        
        # print(ret[:int(ret[0])+1].astype(np.uint64))
        ret_data = ret_sz+ret_pts
        
        # print(len(ret_sz),len(ret_pts),len(ret_data))
        
        # print(OUTPUT_INDX)
        ctx[OUTPUT_INDX] = int(ret[0])
        ctx[OUTPUT_INDX+1:OUTPUT_INDX+len(ret_data)+1] = ret_data

        # seg_ret = out[0]#cv2.normalize(out[0],out[0],0,1,cv2.NORM_MINMAX)
        # _, seg_ret = cv2.threshold(seg_ret+10, 0, 0,cv2.THRESH_TOZERO)
        # seg_ret = cv2.convertScaleAbs(seg_ret,seg_ret,64)
        # cv2.imshow(args.topic,cv2.applyColorMap(seg_ret,cv2.COLORMAP_OCEAN))
        # cv2.waitKey(1)

        ctx[0] = OnMaster
        
        
        