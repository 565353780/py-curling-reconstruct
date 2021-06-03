import numpy as np
import os
import argparse

import cv2
import torch
import onnx
from torch.onnx import OperatorExportTypes
from torch.autograd import Variable
from LapNet import LAPNet

parser = argparse.ArgumentParser(description="Inference Server")
parser.add_argument('--model-path', required=True)
parser.add_argument('--export', required=True)
parser.add_argument('--ch',type=int, required=True)
#
args = parser.parse_args()


def inference_pipeline(img):
    with torch.no_grad():
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(np.array(np.reshape(img.swapaxes(1,2).swapaxes(0,1), (1, 3,512, 1024)), np.float32)).cuda()
        seg_map = model(input_tensor)
        seg_map =  torch.squeeze(seg_map,0).cpu().numpy()
        seg_ret = seg_map[1]
        seg_ret = cv2.pyrUp(seg_ret)
        _, seg_ret = cv2.threshold(seg_ret+2, 0, 0,cv2.THRESH_TOZERO)
        seg_ret = cv2.normalize(seg_ret,seg_ret,0,1,cv2.NORM_MINMAX)
        seg_ret = cv2.convertScaleAbs(seg_ret,seg_ret,255)

        return seg_map[1],cv2.applyColorMap(seg_ret,cv2.COLORMAP_MAGMA)

if __name__ == "__main__":
    model = LAPNet(input_ch=3, output_ch=2, internal_ch = args.ch).cuda()
    model.load_state_dict(torch.load(args.model_path)['net'],strict=False)
    # model.eval()
    model.train(False)
    model.eval()

    if(args.export == "1"):
        print("EXPORT MODEL:")
        dummy_tensor = torch.randn(1, 3, 512, 1024, device='cuda')
        torch.onnx.export(model, dummy_tensor, "./export.onnx", export_params=True)
        torch.save(model, "model.pth")
        print("EXIT")
    # , operator_export_type=OperatorExportTypes.RAW)
        exit()    	
    if(args.export == "2"):
        x = torch.randn((1, 3, 512, 1024), requires_grad=False).cuda()
        model(x)
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            model(x)
        print(prof) 
        exit()

    cap = cv2.VideoCapture("../../newfisheye.mp4")
    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, 1196444237)

    _, frame = cap.read()

    h,w,c = np.shape(frame)
    print(w,h)
    cx = w/2
    K = np.array([[1.5359092833382822e+03,-1.5621282801452412e+00,2.0481648289763698e+03],[0.,1.5355831846487040e+03,1.5192968328430725e+03],[0.,0.,1.]],dtype=np.float32)
    D = np.array([2.4927181325942505e-02,2.0546794839049018e-02,-1.0479756363460040e-02,3.1421725970784682e-03],dtype=np.float32)
    P = np.zeros((3,3))
    cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), None, P, 0.5);
    m1,m2  = cv2.fisheye.initUndistortRectifyMap(K, D,None, P,(w,h),cv2.CV_32FC1)

    print(cap.isOpened())
    offset = 11
    while(cv2.waitKey(1) != 27):
        _, frame = cap.read()
        # cv2.imshow("frame", frame)
        # cv2.putText(frame,str())
        frame[0:h-1,offset:w-1] = frame[0:h-1,0:w-offset-1]
        frame = cv2.remap(frame,m1,m2,cv2.INTER_AREA)
        # cv2.imshow("vin",frame)
        seg_raw, seg_view = inference_pipeline(cv2.resize(frame,(1024,512),interpolation=cv2.INTER_AREA))
        seg_view = cv2.resize(seg_view,(w,h))
        cv2.addWeighted(frame,0.5,seg_view,1,0,seg_view)
        cv2.imshow("canvas", seg_view)
        