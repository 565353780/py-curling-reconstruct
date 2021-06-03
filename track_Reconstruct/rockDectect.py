import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def hsvDist(color1,color2):
	pass

def isGray(color):
	avg=0
	variance=0
	for i in range(3):
		avg+=color[i]
	avg=avg/3
	for i  in range(3):
		variance+=pow((color[i]-avg),2)
	variance=math.sqrt(variance/3)
	if  avg>20 and avg<160 and variance < 30 :
		return True
	else:
		return False
	pass

# by #gray points > 10
def hasGrayNear(image,point):
	(h,w,_)=image.shape
	i0=point[1]
	j0=point[0]
	count=0
	for i in range(-6,6):
		for j in range(0,10):
			if i0+i<0 or j0+j<0 or i0+i>=w or j0+j>=h:
				continue
			if(isGray(image[j0+j,i0+i])):
				count+=1
	if count>10:
		return True
	else:
		return False

def rockPos(image):
	# yellow=(68,100,178)
	# red=(343,165,165)
	rock_list=[]
	yellow_list=[]
	# red_list=[]

	(h_im,w,_)=image.shape
	tmp_image=np.zeros((h_im,w,3),np.int)
	# print("yellow",yellow)
	yellow_count=0
	for i in range(w):
		for j in range(h_im):
			tmp_c=tmp_image[j,i]
			if(not tuple(tmp_c)==(0,0,0)):
				continue
			tmp_image[j,i][0]=1
			(b,g,r)=image[j,i]
			h,s,v= rgb2hsv(r,g,b)

			# find red
			# if abs(h-340)<10 and v<0.95  and s>0.3 and i-j<400:
			# 	tmp_image[j,i]=image[j,i]
			# 	red_list.append(np.array([j,i],np.int))

			# find yellow
			if abs(h-62)<15 and v<0.96  and v>0.5 and s>0.15:
				tmp_image[j,i]=image[j,i]
				yellow_count+=1
				yellow_list.append(np.array([j,i],np.int))
	
	yellow_dict={}
	for point in yellow_list:
		away=True
		for key in yellow_dict:
			x0=int(key/10000)
			y0=int(key)%10000
			if np.linalg.norm(np.array([x0,y0])-point)<30:
				for point_other in yellow_dict[key]:
					if (np.linalg.norm(point-point_other))>0 and(np.linalg.norm(point-point_other))<5:
						away=False
						yellow_dict[key].append(point)
						break
		if away:
			new_key=10000*int(point[0])+int(point[1])
			yellow_dict[new_key]=[point]
	# print(len(list(yellow_dict.keys())))

	for key in yellow_dict:
		avgpoint=yellow_dict[key][0]
		if len(yellow_dict[key])>6 and hasGrayNear(image,avgpoint):
			rock_list.append([[(avgpoint[1]+3)/w,(avgpoint[0]+2)/h_im],[6/w,4/h_im],1])
	
	# plt.subplot(1,2,2)
	# plt.imshow(tmp_image[:,:,::-1])
	
	return rock_list

def mark_rock(image,rock_list):
	(im_h,im_w,_) =image.shape
	
	colorList=[(0,0,255),(14,183,235)]
	for rock in rock_list:
		if 0 <= rock[2] < 2:
			[x,y]=rock[0]
			[w,h]=rock[1]
			color=colorList[rock[2]]
			#print("x,y,w,h,color",x,y,w,h,color)
			cv2.ellipse(image,(int(x*im_w),int(y*im_h)),(int(w*im_w),int(h*im_h)),0,300,580,color,2,cv2.LINE_AA)



# filename=("output/curling_4_frame_721.jpg")
# image=cv2.imread(filename)
# origin_image=image
# (h,w,_)=image.shape
# scale=min(600/w,600/h)
# w1=int(w*scale)
# h1=int(h*scale)
# print(image.shape)
# image=cv2.resize(image,(w1,h1))
# rock_list=rockPos(image)


# #rock_list=[[[188/w1,270/h1],[8/w1,6/h1],0]]
# mark_rock(origin_image,rock_list)
# plt.subplot(1,2,1)
# plt.imshow(origin_image[:,:,::-1])
# plt.show()