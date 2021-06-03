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
args = parser.parse_args()


def inference_pipeline(img):
    frame_h, frame_w, _ = img.shape
    crop_factor_h = 0.32
    crop_factor_w = 0
    h = frame_h - frame_h*crop_factor_h
    w = frame_w
    x = 0
    y = int(frame_h - h)//2

    #rng.uniform(int(frame_h - h/2-1), int(frame_h - h/2+1))
    crop = np.array([y,y+h,x,x+w]).astype('int')
    img = img[crop[0]:crop[1],crop[2]:crop[3]]
    img = cv2.UMat.get(cv2.resize(cv2.UMat(img),(1024,512),interpolation=cv2.INTER_AREA))
    with torch.no_grad():
        if(args.export == 1):
            dummy_tensor = torch.from_numpy(np.array(np.reshape(img.swapaxes(1,2).swapaxes(0,1), (1, 3,512, 1024)), np.float32)).cuda()
            torch.onnx.export(model, dummy_tensor, "./export.onnx", export_params=True,
                verbose=True, operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
            exit()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(np.array(np.reshape(img.swapaxes(1,2).swapaxes(0,1), (1, 3,512, 1024)), np.float32)).cuda()
        seg_map = model(input_tensor)
        seg_map =  torch.squeeze(seg_map,0).cpu().numpy()
        seg_ret = seg_map[1]
        seg_ret = cv2.pyrUp(seg_ret)
        _, seg_ret = cv2.threshold(seg_ret+3, 0, 1,cv2.THRESH_TOZERO)
        seg_ret = cv2.normalize(seg_ret,seg_ret,0,1,cv2.NORM_MINMAX)
        seg_ret = cv2.convertScaleAbs(seg_ret,seg_ret,255)

        return seg_map[1],cv2.applyColorMap(seg_ret,cv2.COLORMAP_OCEAN),img

if __name__ == "__main__":
    model = LAPNet(input_ch=3, output_ch=2, internal_ch = 32).cuda()
    model.load_state_dict(torch.load(args.model_path)['net'],strict=False)
    model.eval()

    cap = cv2.VideoCapture("../newfisheye.avi")
    # cap.set(cv2.CAP_PROP_POS_FRAMES,17000)
    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, 1196444237)

    _, frame = cap.read()

    h,w,c = np.shape(frame)
    print(w,h)
    K = np.array([[1.5359092833382822e+03,-1.5621282801452412e+00,2.0481648289763698e+03],[0., 1.5355831846487040e+03, 1.5192968328430725e+03],[0., 0., 1.]],dtype=np.float32)
    D = np.array([ 2.4927181325942505e-02,2.0546794839049018e-02,-1.0479756363460040e-02,3.1421725970784682e-03],dtype=np.float32)
    P = np.zeros((3,3))
    cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), None, P, 1)
    m1,m2  = cv2.fisheye.initUndistortRectifyMap(K, D,None, P,(w,h),cv2.CV_32FC1)

    print(cap.isOpened())
    offset = 11
    w = 1024
    h = 512
    while(cv2.waitKey(1) != 27):
        n_fid = cap.get(cv2.CAP_PROP_POS_FRAMES)
        _, frame = cap.read()
        # cv2.imshow("frame", frame)
        # cv2.putText(frame,str())
        frame = cv2.UMat.get(cv2.remap(cv2.UMat(frame),m1,m2,cv2.INTER_AREA))
        seg_raw, seg_view,frame = inference_pipeline(frame)
        seg_view = cv2.resize(seg_view,(w,h))
        cv2.imwrite("../shougang_point_auto_annotation_0616/bin_"+str(int(n_fid))+".png",seg_view)
        frame = cv2.resize(frame,(w,h))
        cv2.addWeighted(frame,0.5,seg_view,1,0,seg_view)
        cv2.imshow("canvas", seg_view)
        print(n_fid/cap.get(cv2.CAP_PROP_FRAME_COUNT)*100,"%","    FrameID:",int(n_fid),end="                          \r")
        