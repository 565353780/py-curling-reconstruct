import numpy as np
import cv2
import os
path_name='output\\'
video_name='curling_5';
cap = cv2.VideoCapture(video_name+".mp4");
i=0
print(path_name+video_name+"\\"+video_name+"_frame_"+str(i)+".jpg")
#os.makedirs(path_name)
while(cap.isOpened()):
    ret, frame = cap.read()
    # 这里必须判断视频是否读取完毕,否则最后一帧播放出现问题

    if ret == True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame', gray)
        if i%10==7:
            print(path_name+video_name+"\\"+video_name+"_frame_"+str(i)+".jpg")
            cv2.imwrite(path_name+video_name+"_frame_"+str(i)+".jpg",frame)
        #cv2.imshow('frame',frame)
        i+=1
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
