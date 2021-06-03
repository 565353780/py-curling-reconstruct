import cv2
import numpy as np
import os
import glob

class myEllipse():
	def __init__(self,pos,size):
		self.pos=pos
		self.size=size

def load_ellipse(filename):
	global ellipse_list
	ellipse_list=[]
	in_filename= filename+".txt"
	count=0
	with open(in_filename,"r") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()  
			if not len(line):  #
				continue  # 
			split_i=line.find(";")
			pos_str=line[5:split_i-1]
			size_str=line[split_i+6:-1]
			print(split_i,pos_str,size_str)
			pos_split=pos_str.find(",")
			size_split=size_str.find(",")
			pos=(int(pos_str[:pos_split]),int(pos_str[pos_split+1:]))
			size=(int(size_str[:size_split]),int(size_str[size_split+1:]))
			print(pos,size)
			ellipse_sample= myEllipse(pos,size)
			ellipse_list.append(ellipse_sample)
			count+=1
		f.close()
		print("finish load points,num:",count)
		return True
	return False

def generateAnnotationImage(filename):
	image=cv2.imread(filename)
	cv2.imshow("",image)
	load_ellipse(filename[:-4])
	tmp_image=np.zeros(image.shape,np.uint8)
	(h,w,_)=image.shape
	print("w,h",w,h)
	for ellipse in ellipse_list:
		pos=ellipse.pos
		size=ellipse.size
		a=int(size[0]/2)
		b=int(size[1]/2)
		for i in range(-a,a):
			for j in range(-b,b):
				if pos[0]+i<0 or pos[1]+j<0 or pos[0]+i>=w or pos[1]+j>=h:
					continue
				if (i*i/a/a+j*j/b/b)<1:
					tmp_image[pos[1]+j,pos[0]+i]=image[pos[1]+j,pos[0]+i]
					#print("color",image[pos[1]+j,pos[0]+i])

	tmp_image=cv2.cvtColor(tmp_image,cv2.COLOR_BGR2GRAY)
	cv2.imwrite(filename[:-4]+".png",tmp_image)

ellipse_list=[]
path_list=glob.glob("**/**.jpg")
for  path in path_list:
	generateAnnotationImage(path)
	pass