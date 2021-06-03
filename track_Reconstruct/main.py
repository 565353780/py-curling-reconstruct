from lineDetect import LineDetect
import trackReconstruct
import rockDectect
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from useLapNet import LapNetResult
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy.linalg  as lin
from math import sqrt
from trackReconstruct import track_width, track_height, circle_height, hog_height, track_num

image_in = []
image = []
image_container = []


class TrackReconstruct:
    def __init__(self, CallerLineIdx=0, max_dist_scale=0.035, ShowTimeSpend=False):
        self.rockTracker = LapNetResult(ShowTimeSpend)
        self.lineDetect = LineDetect()
        self.H = None
        self.HT = None
        if 0 < CallerLineIdx <= track_num:
            # self.target_caller_line = [(2 * CallerLineIdx - 1) / track_num / 2, 1 - hog_height / track_height,
            #                            (2 * CallerLineIdx - 1) / track_num / 2, circle_height / track_height]
            self.target_caller_line = [(2 * CallerLineIdx - 1) / track_num / 2, 1 - hog_height / track_height]
        self.max_dist_scale = max_dist_scale
        self.InitCallerLine = False
        self.ShowTimeSpend = ShowTimeSpend

    def get_point_dist2(self, point_1, point_2):
        return (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (
                point_1[1] - point_2[1])

    def get_point_dist(self, point_1, point_2):
        return sqrt(self.get_point_dist2(point_1, point_2))

    def dir_dot(self, dir_1, dir_2):
        return dir_1[0] * dir_2[0] + dir_1[1] * dir_2[1]

    def reconstruct(self, image_path):
        global target_rock_track
        global projected_target_rock_track

        # if you open "draw_result_for_findLines", then the rock track will be bad due to the drawn color
        draw_result_for_findLines = False
        draw_intersections = False

        global image_in
        global image
        global image_container
        origin_image = image_in
        (h, w, _) = image_in.shape

        scale = min(600 / w, 600 / h)
        if scale < 1:
            w1 = int(w * scale)
            h1 = int(h * scale)
            image = cv2.resize(image_in, (w1, h1))
        else:
            image = image_in.copy()

        if self.ShowTimeSpend:
            print("=============================================================")
            print("Start lineDetect...")
            total_tick = time.time()
            tick = time.time()

        # find all short lines and merge them to several exact lines
        final_line_list = self.lineDetect.findLines(image)

        if self.ShowTimeSpend:
            print("----Spend time on findLines : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        # find the intersection points and merge them
        intersection_list, intersection_id_list, intersection_group_dict = self.lineDetect.findIntersectionsAndGroup(
            final_line_list)

        if self.ShowTimeSpend:
            print("----Spend time on findIntersectionsAndGroup : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        # find the vertical lines and base-line id and hog-line id in final-line-list
        vertical_line_id_list, base_line_id, hog_line_id = self.lineDetect.getVerticalAndHorizonalLineIdList(
            intersection_id_list, intersection_group_dict, intersection_list)

        if self.ShowTimeSpend:
            print("----Spend time on getVerticalAndHorizonalLineIdList : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        new_vertical_lines = self.lineDetect.updateVerticalLines(image, vertical_line_id_list, final_line_list, 5, 50)

        if self.ShowTimeSpend:
            print("----Spend time on updateVerticalLines : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        if draw_result_for_findLines:
            print('initial line num :', len(final_line_list))
            for line in final_line_list:
                [x1, y1, x2, y2] = line
                color = (0, 0, 255)
                cv2.line(image, (int(x1 * image.shape[1]), int(y1 * image.shape[0])),
                         (int(x2 * image.shape[1]), int(y2 * image.shape[0])), color, 1)
            cv2.imshow('test', image)
            cv2.waitKey()

        if draw_intersections:
            for group_id in intersection_group_dict:
                avg_point = np.zeros((1, 2))
                count = 0
                for point_id in intersection_group_dict[group_id]:
                    # print("point",intersection_list[point_id])
                    avg_point += np.array(intersection_list[point_id])
                    count += 1
                avg_point /= count
                print("avg_point", avg_point, count)
                if count > 5:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(image, (int(avg_point[0][0] * image.shape[1]), int(avg_point[0][1] * image.shape[0])), 3,
                           color, -1)
            cv2.imshow('test', image)
            cv2.waitKey()

        vertical_line_list = []

        for line in new_vertical_lines:
            [x1, y1, x2, y2] = line
            vertical_line_list.append([x1, y1, x2, y2])

        base_line_list = []

        [x1, y1, x2, y2] = self.lineDetect.updateBaseLine(image_in, final_line_list[base_line_id], 10, 5)
        base_line_list.append([x1, y1, x2, y2])

        if self.ShowTimeSpend:
            print("----Spend time on updateBaseLine : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        hog_line_list = []

        [x1, y1, x2, y2] = self.lineDetect.updateHogLine(image_in, final_line_list[hog_line_id], 10, 10)
        hog_line_list.append([x1, y1, x2, y2])

        if self.ShowTimeSpend:
            print("----Spend time on updateHogLine : ", int((time.time() - tick) * 1000), ' ms')
            print("lineDetect finished !")
            print("Spend time on lineDetect : ", int((time.time() - total_tick) * 1000), ' ms')

            total_tick = time.time()

        # create the relationship of corresponding points and get transformation from origin-points to target-points with cv
        target_point_list = []
        # origin_point_list = []
        normalized_origin_point_list = []
        start_vertical_line_idx = 2
        for i in range(start_vertical_line_idx, start_vertical_line_idx + 4):
            tmp_point = []
            self.lineDetect.intersection(final_line_list[vertical_line_id_list[-i]], final_line_list[base_line_id],
                                         tmp_point)
            target_point_list.append(tmp_point)
            # origin_point_list.append([(3.5 - i) * track_width, -track_height / 2])
            normalized_origin_point_list.append([(track_num + 1 - i) / track_num, 0])
            tmp_point = []
            self.lineDetect.intersection(final_line_list[vertical_line_id_list[-i]], final_line_list[hog_line_id],
                                         tmp_point)
            target_point_list.append(tmp_point)
            # origin_point_list.append([(3.5 - i) * track_width, -track_height / 2 + hog_height])
            normalized_origin_point_list.append([(track_num + 1 - i) / track_num, hog_height / track_height])

        self.H = trackReconstruct.findPerspectiveMatrix(normalized_origin_point_list[:8], target_point_list[:8])
        self.HT = trackReconstruct.findPerspectiveMatrix(target_point_list[:8], normalized_origin_point_list[:8])

        if self.ShowTimeSpend:
            print("Spend time on findHomography : ", int((time.time() - total_tick) * 1000), ' ms')

        # #draw result for  homography
        # track_polygon_list=[]
        # center=[0.,0.]
        # trackReconstruct.generateTracks(center,track_width,track_height,0,int((track_num+1)/2),track_polygon_list)
        # projected_polygon_list=trackReconstruct.projectTracks(H,track_polygon_list)
        # for polygon in projected_polygon_list:
        #     cv2.polylines(image,[np.array(polygon,np.int)],True,(0,0,255))

        if self.ShowTimeSpend:
            print("Start findRock...")
            total_tick = time.time()
            tick = time.time()

        if 0 < CallerLineIdx <= track_num:
            if not self.InitCallerLine:
                target_rock_track = [self.target_caller_line]

                caller_pos_ = self.load_line_caller_pos(
                    'line_caller_pos/' + image_path.split('/')[len(image_path.split('/')) - 1].split('.')[0])
                caller_pos_[0] /= image_in.shape[1]
                caller_pos_[1] /= image_in.shape[0]
                caller_pos_ = trackReconstruct.projectTracks(self.HT, [[caller_pos_]])[0][0]

                self.target_caller_line = [self.target_caller_line[0], self.target_caller_line[1], caller_pos_[0],
                                           caller_pos_[1]]

                self.InitCallerLine = True

        projected_target_caller_line = []
        projected_target_caller_line_point = \
            trackReconstruct.projectTracks(self.H, [[[self.target_caller_line[0], self.target_caller_line[1]]]])[0][0]
        projected_target_caller_line.append(projected_target_caller_line_point[0])
        projected_target_caller_line.append(projected_target_caller_line_point[1])
        projected_target_caller_line_point = \
            trackReconstruct.projectTracks(self.H, [[[self.target_caller_line[2], self.target_caller_line[3]]]])[0][0]
        projected_target_caller_line.append(projected_target_caller_line_point[0])
        projected_target_caller_line.append(projected_target_caller_line_point[1])

        if self.ShowTimeSpend:
            print("----Spend time on initCallerLine : ", int((time.time() - tick) * 1000), ' ms')
            # tick = time.time()

        rock_list = self.rockTracker.get_result(image_path, projected_target_caller_line, self.max_dist_scale)

        if self.ShowTimeSpend:
            # print("----Spend time on getRockfromLapNet : ", int((time.time() - tick) * 1000), ' ms')
            tick = time.time()

        rock_point_list = []
        for rock in rock_list:
            point = rock[0]
            rock_point_list.append([point[0], point[1]])
            rock[2] = 2

        unprojeced_rock_point_list = trackReconstruct.projectTracks(self.HT, [rock_point_list])
        target_rock = []
        if self.target_caller_line is not None:
            target_caller_line_point_1 = [self.target_caller_line[0], self.target_caller_line[1]]
            target_caller_line_point_2 = [self.target_caller_line[2], self.target_caller_line[3]]
            target_caller_line_dir = [target_caller_line_point_2[0] - target_caller_line_point_1[0],
                                      target_caller_line_point_2[1] - target_caller_line_point_1[1]]
            target_caller_line_dir_norm = self.get_point_dist(target_caller_line_dir, [0, 0])
            target_caller_line_dir = [target_caller_line_dir[0] / target_caller_line_dir_norm,
                                      target_caller_line_dir[1] / target_caller_line_dir_norm]
            num = len(unprojeced_rock_point_list[0])
            # count = 0
            # target_i = -1
            target_i = []
            for i in range(num):
                rock = unprojeced_rock_point_list[0][i]

                rock_dir = [rock[0] - target_caller_line_point_1[0], rock[1] - target_caller_line_point_1[1]]

                parallel_norm = self.dir_dot(rock_dir, target_caller_line_dir)
                if parallel_norm <= 0:
                    current_dist = self.get_point_dist(target_caller_line_point_1, rock)
                    if current_dist < self.max_dist_scale:
                        target_i.append([i, current_dist])
                elif parallel_norm >= 1:
                    current_dist = self.get_point_dist(target_caller_line_point_2, rock)
                    if current_dist < self.max_dist_scale:
                        target_i.append([i, current_dist])
                else:
                    vertical_dir = [rock_dir[0] - parallel_norm * target_caller_line_dir[0],
                                    rock_dir[1] - parallel_norm * target_caller_line_dir[1]]
                    current_dist = self.get_point_dist(vertical_dir, [0, 0])
                    if current_dist < self.max_dist_scale:
                        target_i.append([i, current_dist])

            last_point = target_rock_track[len(target_rock_track) - 1]
            if len(target_rock_track) == 1:
                min_dist = self.max_dist_scale
                min_idx = -1
                for idx in target_i:
                    if idx[1] < min_dist:
                        min_dist = idx[1]
                        min_idx = idx[0]
                if min_idx != -1:
                    temp_point = unprojeced_rock_point_list[0][min_idx]
                    if self.get_point_dist(temp_point, last_point) < 0.4:
                        target_rock_track.append(unprojeced_rock_point_list[0][min_idx])
                        rock_list[min_idx][2] = 0
            else:
                max_dot = 0.7
                temp_rock_track = []
                for idx in target_i:
                    current_point = unprojeced_rock_point_list[0][idx[0]]
                    current_dir = [current_point[0] - last_point[0], current_point[1] - last_point[1]]
                    current_dir_norm = self.get_point_dist(current_dir, [0, 0])
                    current_dir = [current_dir[0] / current_dir_norm, current_dir[1] / current_dir_norm]
                    current_dot = self.dir_dot(target_caller_line_dir, current_dir)
                    if current_dot > max_dot:
                        temp_rock_track.append([idx[0], current_dot])

                min_temp_dist = 1
                min_temp_idx = -1
                for idx in temp_rock_track:
                    temp_point = unprojeced_rock_point_list[0][idx[0]]
                    temp_dist = self.get_point_dist(temp_point, last_point) / idx[1]
                    if temp_dist < min_temp_dist:
                        min_temp_dist = temp_dist
                        min_temp_idx = idx[0]

                if min_temp_idx != -1:
                    if min_temp_dist < 0.4:
                        target_rock_track.append(unprojeced_rock_point_list[0][min_temp_idx])
                        rock_list[min_temp_idx][2] = 0
                        target_rock.append(unprojeced_rock_point_list[0][min_temp_idx])
                        # for i in range(11):
                        #     target_rock.append([4 / track_num, i / 10])
                        # target_rock = self.TestHTH(target_rock)
                        # print('------', unprojeced_rock_point_list[0][min_temp_idx])
                        # print('------', rock_point_list[min_temp_idx])

            projected_target_rock_track = trackReconstruct.projectTracks(self.H, [target_rock_track])[0]

        if self.ShowTimeSpend:
            print("----Spend time on findTargetRock : ", int((time.time() - tick) * 1000), ' ms')
            print("findRock finished !")
            print("Spend time on findRock : ", int((time.time() - total_tick) * 1000), ' ms')

        return vertical_line_list, hog_line_list, base_line_list, rock_list, target_rock

    def TestH(self, point_list):
        return trackReconstruct.projectTracks(self.H, [point_list])[0]

    def TestHT(self, point_list):
        return trackReconstruct.projectTracks(self.HT, [point_list])[0]

    def TestHTH(self, point_list):
        return self.TestHT(self.TestH(point_list))

    def TestHHT(self, point_list):
        return self.TestH(self.TestHT(point_list))

    def drawNomalizedLineList(self, image, lineList, color, thickness):
        (h, w, _) = image.shape
        for line in lineList:
            [x1, y1, x2, y2] = line
            cv2.line(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, thickness)

    # use matplotlib pathpatch to draw bezier path
    def drawTargetRockTrackPathPatch(self, shape, rock_track):
        track = []
        (h, w, _) = shape
        num = len(rock_track)
        if num < 2:
            return
        to_correct_id_list = []
        for i in range(num):
            point = rock_track[i]
            [x, y] = point
            if i > 0 and i < num - 1:
                p1 = rock_track[i - 1]
                p2 = rock_track[i + 1]
                dir1 = np.array(point) - np.array(p1)
                dir2 = np.array(p2) - np.array(point)
                if dir1[0] * dir2[1] - dir1[1] * dir2[0] > 0:
                    to_correct_id_list.append(i)
            track.append([(x * w), (y * h)])
        for id in to_correct_id_list:
            p1 = track[id - 1]
            p2 = track[id + 1]
            p = (np.array(p1) + np.array(p2)) * 0.5 + np.array([1., 0])
            track[id] = p.tolist()
        bezier_point_list = []
        Path = mpath.Path
        path_data = []
        for i in range(num - 1):
            p0 = np.array(track[i])
            p1 = np.array(track[i + 1])
            if i == 0:
                p_1 = 2 * p0 - p1 - np.array([1., 0])
            else:
                p_1 = np.array(track[i - 1])
            if i == num - 2:
                p2 = 2 * p1 - p0 - np.array([1., 0])
            else:
                p2 = np.array(track[i + 2])
            # print("bezier pointlist",p0,p1,p_1,p2)
            dir0 = ((p0 - p_1) / lin.norm(p0 - p_1) + (p1 - p0) / lin.norm(p1 - p0)) / 2
            dir1 = ((p1 - p0) / lin.norm(p1 - p0) + (p2 - p1) / lin.norm(p2 - p1)) / 2
            if i == 0:
                path_data.append((Path.MOVETO, track[i]))
            else:
                path_data.append((Path.CURVE4, track[i]))
            path_data.append((Path.CURVE4, (p0 + 3 * dir0).tolist()))
            path_data.append((Path.CURVE4, (p1 - 3 * dir1).tolist()))

        path_data.append((Path.CURVE4, track[num - 1]))
        codes, verts = zip(*path_data)

        my_path = mpath.Path(verts, codes, closed=False)
        ax0 = plt.gca()
        if len(ax0.patches) > 1:
            ax0.patches.pop()
        ax0.add_patch(mpatches.PathPatch(my_path, fill=False))
        pass

    def drawCallerLine(self, ax0, caller_begin, caller_pos):
        # print("caller line:",caller_begin, caller_pos)
        patch = mpatches.FancyArrow(caller_begin[0], caller_begin[1], caller_pos[0] - caller_begin[0],
                                    caller_pos[1] - caller_begin[1],
                                    width=3, head_width=10, length_includes_head=True, linewidth=1,
                                    facecolor='red', edgecolor='red', alpha=0.4,
                                    linestyle=":", rasterized=True)

        if len(ax0.patches) > 0:
            ax0.patches.clear()
        ax0.add_patch(patch)

    def load_line_caller_pos(self, filename):
        if '.txt' not in filename:
            in_filename = filename + ".txt"
        else:
            in_filename = filename
        count = 0
        with open(in_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not len(line):  #
                    continue  #
                split_i = line.find(";")
                pos_str = line[5:split_i - 1]
                pos_split = pos_str.find(",")
                pos = [float(pos_str[:pos_split]), float(pos_str[pos_split + 1:])]
                count += 1
            f.close()
            return pos
        return None


ShowlineDetectResult = False
ShowCallerLine = True
ShowRockResult = True
ShowTargetRockTrackPathPatch = True
ShowImageContainer = True
ShowTimeSpend = True

# Idx from left to right : 1, 2, 3, ..., track_num
CallerLineIdx = 5
WaitTimeForRecordVideo = 0

plt.figure(figsize=(13.33, 8.1), dpi=72)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.05)
ax0 = plt.gca()
plt.ioff()

trackreconstruct = TrackReconstruct(CallerLineIdx, 0.035, ShowTimeSpend)

projected_target_rock_track = []
for frame_count in range(117, 717, 40):
    filename = "output/curling_5_frame_" + str(frame_count) + ".jpg"
    image_in = cv2.imread(filename)
    image_container = []
    vertical_line_list, hog_line_list, base_line_list, rock_list, target_rock = trackreconstruct.reconstruct(
        os.getcwd() + '/' + filename)
    if ShowlineDetectResult:
        trackreconstruct.drawNomalizedLineList(image_in, vertical_line_list, (255, 0, 0), 2)
        trackreconstruct.drawNomalizedLineList(image_in, hog_line_list, (0, 0, 255), 1)
        trackreconstruct.drawNomalizedLineList(image_in, base_line_list, (0, 0, 0), 3)
    if ShowRockResult:
        rockDectect.mark_rock(image_in, rock_list)
        trackReconstruct.drawTrack(image_container, 300, target_rock)
    else:
        trackReconstruct.drawTrack(image_container, 300, [])

    (h0, w0, _) = image_in.shape
    (h, w, _) = image_container[0].shape
    # print("relative track:",target_rock_track)

    if ShowCallerLine:
        caller_begin = [projected_target_rock_track[0][0] * w0, projected_target_rock_track[0][1] * h0]
        caller_pos = trackReconstruct.projectTracks(trackreconstruct.H, [
            [[trackreconstruct.target_caller_line[2], trackreconstruct.target_caller_line[3]]]])[0][0]
        caller_pos[0] *= w0
        caller_pos[1] *= h0
        trackreconstruct.drawCallerLine(ax0, caller_begin, caller_pos)
    if ShowTargetRockTrackPathPatch:
        trackreconstruct.drawTargetRockTrackPathPatch(image_in.shape, projected_target_rock_track)

    if ShowImageContainer:
        for i in range(w):
            for j in range(h):
                image_in[j, i] = image_container[0][j, i]

    plt.imshow(image_in[:, :, ::-1])
    plt.draw()
    plt.pause(0.1)
    if WaitTimeForRecordVideo > 0:
        time.sleep(WaitTimeForRecordVideo)
        WaitTimeForRecordVideo = 0

while True:
    plt.pause(0.1)
