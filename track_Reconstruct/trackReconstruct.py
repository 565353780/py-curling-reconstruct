import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


track_width = 4.75
track_height = 44.5
back_height = 1.22
circle_height = 4.88
hog_height = 11.28
track_num = 6

def project(point, H):
    # print("H",H)
    actual_p = np.insert(np.array(point, np.float), len(point), 1).reshape(3, 1)
    # print("actual_p:",actual_p)
    projected_p = ((np.mat(H) * actual_p).flatten()).tolist()[0]
    if (projected_p[2] != 0):
        for i in range(3):
            projected_p[i] = projected_p[i] / projected_p[2]
    return projected_p[:-1]


def findPerspectiveMatrix(origin_quad, target_quad):
    src_pts = np.array(origin_quad, np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(target_quad, np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return np.mat(H)


# generate 2*num+1 rect
def generateTracks(center, width, height, interval, num, track_polygon_list):
    for i in range(-num, num):
        polygon = []
        x = center[0] + i * (width + interval)
        y = center[1]
        addRect(x, y, width, height, polygon)
        track_polygon_list.append(polygon)
    pass


def addRect(x, y, w, h, out):
    out.append([x - w / 2, y - h / 2])
    out.append([x + w / 2, y - h / 2])
    out.append([x + w / 2, y + h / 2])
    out.append([x - w / 2, y + h / 2])


def projectTracks(H, track_polygon_list):
    projected_polygon_list = []
    for polygon in track_polygon_list:
        tmp_list = []
        for point in polygon:
            projected_point = project(point, H)
            tmp_list.append(projected_point)
        projected_polygon_list.append(tmp_list)
    return projected_polygon_list


def drawProjectedPolygon():
    global projected_polygon_list
    global image
    for polygon in projected_polygon_list:
        print("polygon", polygon)
        cv2.polylines(image, [np.array(polygon, np.int)], True, (0, 0, 255))
        plt.imshow(image[:, :, ::-1])
        plt.draw()


def load_points(filename):
    global image
    global target_quad
    target_quad = []
    in_filename = filename + ".txt"
    with open(in_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not len(line):  #
                continue  # 
            data_list = line.split(",")
            print(data_list[0] + "," + data_list[1])
            point = [float(data_list[0]), float(data_list[1])]
            target_quad.append(point)
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)
        # cv2.polylines(image,[np.array(target_quad,np.int)],True,(0,255,255))
        plt.imshow(image[:, :, ::-1])
        plt.draw()
        f.close()


def save_points(filename):
    global target_quad
    out_filename = filename + ".txt"
    with open(out_filename, "w") as f:
        for point in target_quad:
            print(str(point[0]) + "," + str(point[1]), file=f)
        f.close()


def on_key_press(event):
    global target_quad
    global track_polygon_list
    global image
    global tempfilename
    global H
    if event.key == "enter":
        plt.draw()
    elif event.key == "c":
        print(track_polygon_list[0])
        H = findPerspectiveMatrix(track_polygon_list[3], target_quad)
        projectTracks()
        drawProjectedPolygon()
    elif event.key == "i":
        load_points(tempfilename)
    elif event.key == "o":
        save_points(tempfilename)


def on_press(event):
    print("pressed:")
    global target_quad
    global image
    global origin_img
    if event.button == 1:  #
        if event.xdata == None:
            return
        target_quad.append([event.xdata, event.ydata])
        cv2.circle(image, (int(event.xdata), int(event.ydata)), 10, (0, 255, 255), -1)
        plt.imshow(image[:, :, ::-1])
        plt.draw()
        print("add position:", event.button, event.xdata, event.ydata)
    elif event.button == 3:  #
        target_quad = []
        image = origin_img
        plt.imshow(img[:, :, ::-1])
        plt.draw()
        print("clear positions")


def drawTrack(image_container, h, rock_point_list):
    blue = (171, 130, 57)
    white = (255, 255, 255)
    red = (60, 20, 220)
    yellow = (14, 183, 235)

    scale = h / track_height
    actual_h, actual_w = int(scale * track_height), int(track_num * scale * track_width)
    tmp_image = np.zeros((actual_h, actual_w, 3), np.uint8)
    tmp_image[:, :, 0] = 255
    tmp_image[:, :, 1] = 255
    tmp_image[:, :, 2] = 255
    # draw vertical
    for i in range(1, track_num):
        x = int(scale * track_width * i)
        cv2.line(tmp_image, (x, 0), (x, actual_h), blue, 1)
    # draw horizonal
    height_list = [int(scale * hog_height), int(scale * (track_height - hog_height)), int(scale * back_height),
                   int(scale * (track_height - back_height))]
    color_list = [red, red, blue, blue]
    for i in range(2):
        cv2.line(tmp_image, (0, height_list[i]), (actual_w, height_list[i]), color_list[i], 1)
        pass
    circle_height_list = [int(scale * circle_height), int(scale * (track_height - circle_height))]
    for i in range(track_num):
        x = int(scale * track_width * (i + 0.5))
        for height_j in circle_height_list:
            drawTrackCircles(tmp_image, x, height_j, scale, blue, red, white)
    for point in rock_point_list:
        cv2.circle(tmp_image, (int(point[0] * actual_w), actual_h - int(point[1] * actual_h)), 3, red, 2, cv2.LINE_AA)

    image_container.append(tmp_image)


def drawTrackCircles(image, x, y, scale, blue, red, white):
    circle_r_list = [1.83, 1.22, 0.61, 0.15]
    circle_color_list = [blue, white, red, white]
    for i in range(4):
        cv2.circle(image, (int(x), int(y)), int(circle_r_list[i] * scale), circle_color_list[i], -1, cv2.LINE_AA)
    pass

## test sample
# H=np.mat(np.zeros((3,3))) #project matrix

# #polygon point order are conter clock wise
# track_polygon_list=[]
# projected_polygon_list=[]

# #4 point to be align 
# target_quad=[]
# center=[0.,0.]
# generateTracks(center,4.75,44.5,0,3)

# filename="curling_4.jpg"
# image=cv2.imread(filename)
# (tempfilename,extension) = os.path.splitext(filename)
# (h,w,_)=image.shape
# scale=min(512/w,512/h)
# w1=int(w*scale)
# h1=int(h*scale)
# image=cv2.resize(image,(w1,h1))
# origin_img=image
# tmp_image=[]
# drawTrack(tmp_image)

# fig = plt.figure()

# fig.canvas.mpl_connect('button_press_event', on_press)
# fig.canvas.mpl_connect('key_press_event', on_key_press)
# plt.imshow(tmp_image[0][:,:,::-1])
# plt.ioff()
# plt.show()
