import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mp
import os
import glob

class myEllipse():
	def __init__(self,pos,size):
		self.pos=pos
		self.size=size

ellipse_sample=myEllipse((100,100),(6,4))

#e1 = mp.Ellipse((xcenter, ycenter), width, height,angle=angle, linewidth=2, fill=False, zorder=2)



def save_ellipse(filename):
	global ellipse_list
	out_filename = filename+".txt"
	count=0
	with open(out_filename,"w") as f:
		for ellipse in ellipse_list:
			print("x,y:"+str(ellipse.pos)+";w,h:"+str(ellipse.size),file=f)
			print("x,y:"+str(ellipse.pos)+";w,h:"+str(ellipse.size))
			count+=1
		f.close()
		print("finish write points,num:",count)
		return True
	return False

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
			#print(split_i,pos_str,size_str)
			pos_split=pos_str.find(",")
			size_split=size_str.find(",")
			pos=(int(pos_str[:pos_split]),int(pos_str[pos_split+1:]))
			size=(int(size_str[:size_split]),int(size_str[size_split+1:]))
			#print(pos,size)
			ellipse= myEllipse(pos,size)
			ellipse_list.append(ellipse)
			count+=1
		f.close()
		print("finish load points,num:",count)
		return True
	return False

def pop_ellipse():
	global ellipse_list
	global ax0
	if len(ellipse_list)>0:
		ellipse_list.pop()
		print(len(ax0.patches))
		ax0.patches.pop()
		ax0.patches.pop()
		e1=mp.Ellipse((0,0),0,0,angle=0,color='red',linewidth=1, fill=False)
		ax0.add_patch(e1)
		ax0.patches.pop()
		plt.draw()

def drawELlipse():
	global ax0
	global ellipse_list
	ax0.patches=[]
	for ellipse in ellipse_list:
		pos=ellipse.pos
		size=ellipse.size
		print("drawing:",ellipse.pos,ellipse.size[0],ellipse.size[1])
		rect=mp.Rectangle((pos[0]-size[0]/2,pos[1]-size[1]/2), ellipse.size[0],ellipse.size[1] ,color='black',linewidth=1, fill=False, linestyle='-.')
		ellp=mp.Ellipse(ellipse.pos,ellipse.size[0],ellipse.size[1],color='red',linewidth=1, fill=True, alpha=0.3)
		ax0.add_patch(rect)
		ax0.add_patch(ellp)
	plt.draw()


def mouse_press_event(event):
	global ax0
	global corner0
	global operate_mode
	global in_drag
	global operate_mode
	print("operate_mode:",operate_mode)
	if event.button==1 and operate_mode==1:
		if type(event.xdata)=='NoneType':
			return
		corner0=(int(event.xdata),int(event.ydata))
		rect=mp.Rectangle(corner0, 8,4, color='black',fill=False,linestyle='-.')
		ellp=mp.Ellipse(corner0, 8,4, color='red',linewidth=1, fill=True, alpha=0.3)
		ax0.add_patch(rect)
		ax0.add_patch(ellp)
		in_drag=True
	elif event.button==1 and operate_mode==2:
		if type(event.xdata)=='NoneType':
			return
		corner0=(int(event.xdata),int(event.ydata))
		rect=mp.Rectangle((int(event.xdata-4),int(event.ydata-2)), 8,4, color='black',fill=False,linestyle='-.')
		ellp=mp.Ellipse(corner0, 8,4, color='red',linewidth=1, fill=True, alpha=0.3)
		ax0.add_patch(rect)
		ax0.add_patch(ellp)
		plt.draw()
		in_drag=True
		
def mouse_move_event(event):
	global ax0
	global corner0
	global corner1
	global cur_ellipse_center
	global cur_ellipse_size
	global in_drag
	global operate_mode
	if event.button==1 and operate_mode==1 and in_drag:
		if type(event.xdata)=='NoneType':
			return
		corner1=(int(event.xdata),int(event.ydata))
		ax0.patches.pop()
		rect=ax0.patches[-1]
		center=(int(corner0[0]/2+corner1[0]/2),int(corner0[1]/2+corner1[1]/2))
		x=(min(corner0[0],corner1[0]))
		y=(min(corner0[1],corner1[1]))
		w=int(abs(corner0[0]-corner1[0]))
		h=int(abs(corner0[1]-corner1[1]))
		rect.set_xy((x,y))
		rect.set_width(w)
		rect.set_height(h)
		e1=mp.Ellipse(center, w,h,angle=0, color='red',linewidth=1, fill=True, alpha=0.3)
		ax0.add_patch(e1)
		plt.draw()

def mouse_release_event(event):
	global ax0
	global ellipse_list
	global in_drag
	global corner0
	global corner1
	global operate_mode
	if event.button==1 and operate_mode==1 and in_drag:
		center=(int(corner0[0]/2+corner1[0]/2),int(corner0[1]/2+corner1[1]/2))
		w=int(abs(corner0[0]-corner1[0]))
		h=int(abs(corner0[1]-corner1[1]))

		ellipse=myEllipse(center,(w,h))
		ellipse_list.append(ellipse)
		print("ellipse:",center,w,h)
		# e1=mp.Ellipse(center, w,h,angle=0, color='red',linewidth=1, fill=False)
		# ax0.add_patch(e1)
		# plt.draw()
		corner0=(0,0)
		corner1=(0,0)
		in_drag=False

def key_press_event(event):
	global file_index
	global path_list
	global operate_mode
	print("press key:",event.key)
	if event.key==" ":
		plt.draw()
	elif event.key=="d":
		readNext(1)
	elif event.key=="a":
		readNext(-1)
	elif event.key=="z":
		pop_ellipse()
	elif event.key=="w":
		save_ellipse(path_list[file_index][:-4])
	elif event.key=="r":
		load_ellipse(path_list[file_index][:-4])
		drawELlipse()
	elif event.key=="0":
		operate_mode=0
	elif event.key=="1":
		operate_mode=1
	elif event.key=="2":
		operate_mode=2

def readNext(num):
	global image_count
	global file_index
	global path_list
	global ellipse_list
	global ax0
	#save_ellipse(path_list[file_index][:-4])
	if file_index+num> len(path_list)-1 or file_index+num<0:
		print("first or last file!!!")
		return
	if len(ellipse_list)>0:
		save_ellipse(path_list[file_index][:-4])
	if image_count==10:
		restartPlt()
		image_count=0
	file_index+=num
	filename=path_list[file_index]
	print("reading file:",filename)
	image=cv2.imread(filename)
	load_ellipse(filename[:-4])
	ax0.imshow(image[:,:,::-1])
	plt.draw()
	drawELlipse()
	image_count=image_count+1
	if image_count==1:
		plt.show()
	del image


def restartPlt():
	global ax0
	plt.close()
	fig=plt.figure()
	ax0=plt.subplot(1,1,1)
	fig.canvas.mpl_connect('button_press_event', mouse_press_event)
	fig.canvas.mpl_connect('button_release_event', mouse_release_event)
	fig.canvas.mpl_connect("motion_notify_event",mouse_move_event)
	fig.canvas.mpl_connect("key_press_event",key_press_event)

	plt.ioff()
	print("finish restartPlt")


path_list=glob.glob("**/**.jpg")
file_index=-1
operate_mode=1 # 0,none; 1,draw ellipse 2, drawfixed ellipse
in_drag=False
cur_ellipse_center=(0,0)
cur_ellipse_size=(0,0)
corner0=(0,0)
corner1=(0,0)
ellipse_list=[]
image_count=0
restartPlt()
readNext(1)












