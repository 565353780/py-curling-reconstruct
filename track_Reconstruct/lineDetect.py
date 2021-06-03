import numpy as np
import cv2
import math
from time import time


class LineDetect:
    def __init__(self):
        pass

    # find the common point of lines
    def intersection(self, line1, line2, intersect):
        [x11, y11, x12, y12] = line1
        [x21, y21, x22, y22] = line2
        flag = True
        if (y11 - y12 == 0):
            if (y21 - y22 == 0):
                x3 = np.nan
                y3 = np.nan
                flag = False
            else:
                k2 = (x21 - x22) / (y21 - y22)
                y3 = y11
                x3 = k2 * (y3 - y21) + x21
            pass
        elif (y21 - y22 == 0):
            k1 = (x11 - x12) / (y11 - y12)
            y3 = y21
            x3 = k1 * (y3 - y11) + x11
            pass
        else:
            k1 = (x11 - x12) / (y11 - y12)
            k2 = (x21 - x22) / (y21 - y22)
            if abs(k2 - k1) > 0.001:
                y3 = ((x11 - x21) - (k1 * y11 - k2 * y21)) / (k2 - k1)
                x3 = k1 * (y3 - y11) + x11
                pass
            else:
                x3 = np.nan
                y3 = np.nan
                flag = False
        intersect.append(x3)
        intersect.append(y3)
        return flag

    # extent x to [0, 1]
    def extendLine(self, line):
        [x1, y1, x2, y2] = line

        x22 = 1
        y22 = y1 + (1 - x1) / (x2 - x1) * (y2 - y1)
        x11 = 0
        y11 = y1 + (-x1) / (x1 - x2) * (y1 - y2)
        return [x11, y11, x22, y22]

    # create new line from the farthest points of current lines
    def mergeLine(self, line1, line2):
        [x11, y11, x12, y12] = line1
        [x21, y21, x22, y22] = line2

        new_line_id_1 = 0
        new_line_id_2 = 0

        max_dist = (x21 - x11) * (x21 - x11) + (y21 - y11) * (y21 - y11)

        current_dist = (x22 - x11) * (x22 - x11) + (y22 - y11) * (y22 - y11)
        if current_dist > max_dist:
            max_dist = current_dist
            new_line_id_1 = 0
            new_line_id_2 = 2

        current_dist = (x21 - x12) * (x21 - x12) + (y21 - y12) * (y21 - y12)
        if current_dist > max_dist:
            max_dist = current_dist
            new_line_id_1 = 2
            new_line_id_2 = 0

        current_dist = (x22 - x12) * (x22 - x12) + (y22 - y12) * (y22 - y12)
        if current_dist > max_dist:
            max_dist = current_dist
            new_line_id_1 = 2
            new_line_id_2 = 2

        max_self_dist_1 = (x12 - x11) * (x12 - x11) + (y12 - y11) * (y12 - y11)
        max_self_dist_2 = (x22 - x21) * (x22 - x21) + (y22 - y21) * (y22 - y21)

        if max_self_dist_1 > max_self_dist_2:
            if max_self_dist_1 > max_dist:
                return line1
        else:
            if max_self_dist_2 > max_dist:
                return line2

        return [line1[new_line_id_1], line1[new_line_id_1 + 1], line2[new_line_id_2], line2[new_line_id_2 + 1]]

    def getAverageLineFromkb(self, lines):
        dir_sum = [0, 0]
        avg_point = [0, 0]
        for line in lines:
            if line[0] < line[2]:
                dir_sum[0] += line[2] - line[0]
                dir_sum[1] += line[3] - line[1]
            else:
                dir_sum[0] += line[0] - line[2]
                dir_sum[1] += line[1] - line[3]
            avg_point[0] += line[0] + line[2]
            avg_point[1] += line[1] + line[3]

        avg_point[0] /= 2 * len(lines)
        avg_point[1] /= 2 * len(lines)

        return [avg_point[0] - dir_sum[0] / 2, avg_point[1] - dir_sum[1] / 2, avg_point[0] + dir_sum[0] / 2,
                avg_point[1] + dir_sum[1] / 2]

    # if color r,g,self.b avg>150 and variance<20 color consider white
    def closeToWhite(self, color):
        avg = 0
        variance = 0
        for i in range(3):
            avg += int(color[i])
        avg = avg / 3
        for i in range(3):
            variance += pow((int(color[i]) - avg), 2)
        variance = math.sqrt(variance / 3)
        if avg > 150 and variance < 20:
            return True
        else:
            return False
        pass

    # if pixel color in radius*radius square centered at i0,j0 has more than 0.2 white points the pixel is near track
    def nearTrack(self, image, i0, j0):
        (w, h, _) = image.shape
        radius = 30
        interval = 5
        count = 0
        avg = 0
        num = 0
        avg_color = np.array([0., 0., 0.])
        for i in range(-radius, radius, interval):
            for j in range(-radius, radius, interval):
                if i + i0 < 0 or i + i0 >= h or j + j0 < 0 or j + j0 >= w:
                    continue
                num += 1
                c = image[(j + j0), (i + i0)].astype(np.int)
                avg_color = avg_color + c
                tmp_value = 0
                for k in range(3):
                    tmp_value += int(c[k])
                avg += tmp_value
                if self.closeToWhite(c):
                    count += 1
        if num > 0:
            avg_color = avg_color / num
            # print("i0,j0,count,avg_color:",i0,j0,count,avg_color)
        if count / num > 0.2:
            return True, avg_color
        else:
            return False, avg_color

    def areCloseLine(self, line1, line2):
        [x11, y11, x12, y12] = line1[0]
        [x21, y21, x22, y22] = line2[0]
        dir1 = np.array([x12 - x11, y12 - y11])
        dir1_norm = np.linalg.norm(dir1)
        dir1 = dir1 / dir1_norm

        dir2 = np.array([x22 - x21, y22 - y21])
        dir2_norm = np.linalg.norm(dir2)
        dir2 = dir2 / dir2_norm

        diff = np.array([x21 - x11, y21 - y11])
        diff_norm = np.linalg.norm(diff)
        diff = diff / diff_norm
        cos_alpha = abs(np.dot(dir1, dir2))
        # print(cos_alpha)
        if cos_alpha > 0.99:
            pass
        cos_beta = abs(np.dot(dir2, diff))
        if cos_beta > 0.95:
            pass

    def classOfLine(self, line):
        [x11, y11, x12, y12] = line[0]
        dir1 = np.array([x12 - x11, y12 - y11], np.double)
        dir1_norm = np.linalg.norm(dir1)
        dir1 = dir1 / dir1_norm
        theta = math.asin(dir1[1])
        divide = 100
        dir_class = int(theta / math.pi * divide / 2)
        # print("small dir",dir1,dir_class)
        if abs(dir_class) > divide - 10:
            # print("small dir--",dir1,line[0])
            t = -y11 / dir1[1]
            x_cut = x11 + t * dir1[0]
            y_cut = x_cut
        else:
            t = -x11 / dir1[0]
            y_cut = y11 + t * dir1[1]

        cut_class = int(y_cut / 10) * divide
        line_class = dir_class + cut_class
        return line_class

    def classDist(self, class1, class2, divide):
        dir_class1 = class1 % (divide / 2)
        cut_class1 = class1 / divide
        dir_class2 = class2 % (divide / 2)
        cut_class2 = class2 / divide
        if dir_class1 > divide / 2:
            dir_class1 -= divide / 2
        if dir_class2 > divide / 2:
            dir_class2 -= divide / 2
        return pow(dir_class1 - dir_class2, 2) / 4 + pow(cut_class1 - cut_class2, 2)

    def findLines(self, image, solve_mode=2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 10, minLineLength=60, maxLineGap=5)

        line_dict = {}
        lines_num = len(lines)
        for index in range(lines_num):
            line = lines[index]
            [x1, y1, x2, y2] = line[0]
            res, avg_color = self.nearTrack(image, int((x1 + x2) / 2), int((y1 + y2) / 2))
            if res:
                line_class = self.classOfLine(line)
                if (line_class in line_dict):
                    line_dict[line_class].append(index)
                else:
                    line_dict[line_class] = [index]
            else:
                pass

        key_list = list(line_dict.keys())
        new_key_list = []
        final_line_list = []

        if solve_mode == -1:
            for key in line_dict:
                for line_id in line_dict[key]:
                    final_line_list.append(lines[line_id][0])

        elif solve_mode == 0:
            for key in key_list:
                away = True
                for key_new in new_key_list:
                    if (self.classDist(key, key_new, 100) < 50):
                        away = False
                        break
                if away:
                    new_key_list.append(key)
                    final_line_list.append(self.extendLine(lines[line_dict[key][0]][0]))

        elif solve_mode == 1:
            for key in key_list:
                away = True
                key_new_idx = 0
                for key_new in new_key_list:
                    if (self.classDist(key, key_new, 100) < 50):
                        away = False

                        new_merge_line = lines[line_dict[key][0]][0]
                        if len(line_dict[key]) > 1:
                            for line_id in line_dict[key]:
                                new_merge_line = self.mergeLine(new_merge_line, lines[line_id][0])
                        final_line_list[key_new_idx] = self.mergeLine(final_line_list[key_new_idx], new_merge_line)

                        break
                    key_new_idx += 1
                if away:
                    new_key_list.append(key)

                    new_merge_line = lines[line_dict[key][0]][0]
                    if len(line_dict[key]) > 1:
                        for line_id in line_dict[key]:
                            new_merge_line = self.mergeLine(new_merge_line, lines[line_id][0])
                    final_line_list.append(new_merge_line)

            for i in range(len(final_line_list)):
                final_line_list[i] = self.extendLine(final_line_list[i])

        elif solve_mode == 2:
            lines_dict_of_key_new_list = {}
            key_selected = []
            for key in key_list:
                [x_1, y_1, x_2, y_2] = lines[line_dict[key][0]][0]
                if x_1 != x_2 and abs((y_2 - y_1) / (x_2 - x_1)) < 0.05:
                    key_selected.append(key)
            for key in key_selected:
                away = True
                key_new_idx = 0
                for key_new in new_key_list:
                    if self.classDist(key, key_new, 100) < 5:
                        away = False

                        for line_id in line_dict[key]:
                            lines_dict_of_key_new_list[key_new_idx].append(lines[line_id][0])

                        break
                    key_new_idx += 1
                if away:
                    new_key_list.append(key)

                    lines_dict_of_key_new_list[len(lines_dict_of_key_new_list)] = []
                    for line_id in line_dict[key]:
                        lines_dict_of_key_new_list[len(lines_dict_of_key_new_list) - 1].append(lines[line_id][0])

            for key in lines_dict_of_key_new_list:
                final_line_list.append(self.extendLine(self.getAverageLineFromkb(lines_dict_of_key_new_list[key])))

            for key in key_list:
                away = True
                key_new_idx = 0
                for key_new in new_key_list:
                    if (self.classDist(key, key_new, 100) < 50):
                        away = False

                        new_merge_line = lines[line_dict[key][0]][0]
                        if len(line_dict[key]) > 1:
                            for line_id in line_dict[key]:
                                new_merge_line = self.mergeLine(new_merge_line, lines[line_id][0])
                        final_line_list[key_new_idx] = self.mergeLine(final_line_list[key_new_idx], new_merge_line)

                        break
                    key_new_idx += 1
                if away:
                    new_key_list.append(key)

                    new_merge_line = lines[line_dict[key][0]][0]
                    if len(line_dict[key]) > 1:
                        for line_id in line_dict[key]:
                            new_merge_line = self.mergeLine(new_merge_line, lines[line_id][0])
                    final_line_list.append(new_merge_line)

            for i in range(len(final_line_list)):
                line = []
                for j in range(4):
                    line.append(final_line_list[i][j] / image.shape[(j + 1) % 2])
                final_line_list[i] = self.extendLine(line)

        return final_line_list

    # find the intersection points and merge them
    def findIntersectionsAndGroup(self, final_line_list):
        intersection_list = []
        intersection_id_list = []
        num = len(final_line_list)
        for i in range(num):
            extend_line = final_line_list[i]
            for j in range(i + 1, num):
                line_other = final_line_list[j]
                tmp_intersction = []
                if (not all(np.array([extend_line == line_other]).flatten())) and self.intersection(extend_line, line_other,
                                                                                               tmp_intersction):
                    intersection_list.append(tmp_intersction)
                    intersection_id_list.append((i, j))
                    pass
        intersection_num = len(intersection_list)
        intersection_group_dict = {}
        max_group_id = 0
        for i in range(intersection_num):
            point = intersection_list[i]
            away = True
            for group_id in intersection_group_dict:
                for point_id in intersection_group_dict[group_id]:
                    point_other = intersection_list[point_id]
                    if np.linalg.norm(np.array(point) - np.array(point_other)) < 0.02:
                        away = False
                        intersection_group_dict[group_id].append(i)
                        break
            if away:
                intersection_group_dict[max_group_id] = [i]
                max_group_id += 1
        # print("intersection_group_dict", intersection_group_dict)
        return intersection_list, intersection_id_list, intersection_group_dict

    # return the 2nd element
    def takeSecond(self, elm):
        return elm[1]

    def getVerticalAndHorizonalLineIdList(self, intersection_id_list, intersection_group_dict,
                                          intersection_list):
        vertical_line_id_list = []
        base_line_intersection_list = []
        hog_line_intersection_list = []
        for group_id in intersection_group_dict:
            count = len(intersection_group_dict[group_id])
            if count >= 5:
                for point_id in intersection_group_dict[group_id]:
                    (i, j) = intersection_id_list[point_id]
                    if i not in vertical_line_id_list:
                        # [x1,x2,y1,y2]=final_line_list[i]
                        # k=(x1-x2)/(y1-y2)
                        vertical_line_id_list.append(i)
                    if j not in vertical_line_id_list:
                        # [x1,x2,y1,y2]=final_line_list[j]
                        # k=(x1-x2)/(y1-y2)
                        vertical_line_id_list.append(j)
            else:
                for point_id in intersection_group_dict[group_id]:
                    if intersection_list[point_id][1] > 0.88:
                        base_line_intersection_list.append((point_id, intersection_list[point_id][0]))
                    elif 0.35 < intersection_list[point_id][1] < 0.47:
                        hog_line_intersection_list.append((point_id, intersection_list[point_id][0]))
        base_line_intersection_list.sort(key=self.takeSecond)
        final_vertical_line_id_list = []
        base_line_id = -1
        hog_line_id = -1
        last_k = -1

        final_base_line_id_list = []
        final_hog_line_id_list = []

        for k in range(0, len(base_line_intersection_list)):
            idm = base_line_intersection_list[k][0]
            (i, j) = intersection_id_list[idm]
            if (i in vertical_line_id_list):
                if last_k == -1 or base_line_intersection_list[k][1] - base_line_intersection_list[last_k][1] > 0.24:
                    final_vertical_line_id_list.append(i)
                    last_k = k
                base_line_id = j
                final_base_line_id_list.append(j)
            elif (j in vertical_line_id_list):
                if last_k == -1 or base_line_intersection_list[k][1] - base_line_intersection_list[last_k][1] > 0.24:
                    final_vertical_line_id_list.append(j)
                    last_k = k
                base_line_id = i
                final_base_line_id_list.append(i)
        for idm in hog_line_intersection_list:
            point_id = idm[0]
            (i, j) = intersection_id_list[point_id]
            if i in final_vertical_line_id_list:
                hog_line_id = j
                final_hog_line_id_list.append(j)
            elif j in final_vertical_line_id_list:
                hog_line_id = i
                final_hog_line_id_list.append(i)

        return final_vertical_line_id_list, base_line_id, hog_line_id

    def getTrueVerticalLine(self, image, lines, search_dist, divide):

        he, we, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        white_min = 100

        total_white_num = 0

        # time0 = 0
        # time1 = 0
        image_white = np.zeros((image.shape[0], image.shape[1]))
        for i in range(we):
            for j in range(he):
                color = image[j][i]
                # temp_t = time()
                if int(color[0]) + int(color[1]) + int(color[2]) > 3 * white_min:
                    # temp_t1 = time()
                    image_white[j][i] = 255
                    total_white_num += 1
                    # time1 += time() - temp_t1
                # time0 += time() - temp_t

        # print('--------spend time on if && write : ', time0 - time1, time1)

        average_white_scale = total_white_num / (he * we)

        h_up = 0
        h_down = 0

        find_line_delay = 6

        temp_find_line_delay = find_line_delay
        for h in range(0, he, int(he / divide)):
            white_num = 0
            for w in range(we):
                if image_white[h][w] == 255:
                    white_num += 1
            if white_num / we > average_white_scale * 0.6:
                if temp_find_line_delay > 0:
                    temp_find_line_delay -= 1
                else:
                    h_up = h / he
                    break

        temp_find_line_delay = find_line_delay
        for h in range(he - 1, 0, -int(he / divide)):
            white_num = 0
            for w in range(we):
                if image_white[h][w] == 255:
                    white_num += 1
            if white_num / we > average_white_scale:
                if temp_find_line_delay > 0:
                    temp_find_line_delay -= 1
                else:
                    h_down = h / he
                    break

        true_vertical_lines = []

        for line in lines:
            point_up = [0, 0]
            point_down = [0, 0]

            line_up = 0
            line_down = 2

            if line[1] > line[3]:
                line_up = 2
                line_down = 0

            new_h_up = h_up
            new_h_down = h_down

            if line[line_up + 1] > new_h_up:
                new_h_up = line[line_up + 1] + 0.05
            if line[line_down + 1] < new_h_down:
                new_h_down = line[line_down + 1] - 0.05

            inter_point_up = []
            inter_point_down = []

            self.intersection(line, [0, new_h_up, 1, new_h_up], inter_point_up)
            self.intersection(line, [0, new_h_down, 1, new_h_down], inter_point_down)

            bound_point_num = 0
            for i in range(int(inter_point_up[0]*we) - search_dist, int(inter_point_up[0]*we) + search_dist):
                if 0 <= i < we:
                    for j in range(int(inter_point_up[1]*he) - search_dist, int(inter_point_up[1]*he) + search_dist):
                        if 0 <= j < he:
                            if edges[j][i] == 255:
                                point_up[0] += i
                                point_up[1] += j
                                bound_point_num += 1

            point_up[0] /= bound_point_num * we
            point_up[1] /= bound_point_num * he

            bound_point_num = 0
            for i in range(int(inter_point_down[0] * we) - search_dist, int(inter_point_down[0] * we) + search_dist):
                if 0 <= i < we:
                    for j in range(int(inter_point_down[1] * he) - search_dist,
                                   int(inter_point_down[1] * he) + search_dist):
                        if 0 <= j < he:
                            if edges[j][i] == 255:
                                point_down[0] += i
                                point_down[1] += j
                                bound_point_num += 1

            point_down[0] /= bound_point_num * we
            point_down[1] /= bound_point_num * he

            true_vertical_lines.append(self.extendLine([point_up[0], point_up[1], point_down[0], point_down[1]]))

        return true_vertical_lines



        # for h in range(0, he, int(he/divide)):
        #     point = []
        #     self.intersection(line, [0, h/he, 1, h/he], point)
        #     true_bound_num = 0
        #     if 0 <= point[0] < 1 and 0 <= point[1] < 1:
        #         point[0] *= we
        #         point[1] *= he
        #         white_pixel_num = 0
        #         average_white_point = [0, 0]
        #         total_pixel_num = 0
        #         for i in range(int(point[0]) - search_dist, int(point[0]) + search_dist):
        #             if 0 <= i < we:
        #                 for j in range(int(point[1]) - search_dist, int(point[1]) + search_dist):
        #                     if 0 <= j < he:
        #                         total_pixel_num += 1
        #                         color = image[j][i]
        #                         if color[0] > white_min and color[1] > white_min and color[2] > white_min:
        #                             white_pixel_num += 1
        #                         if edges[j][i] == 255:
        #                             average_white_point[0] += i
        #                             average_white_point[1] += j
        #                             true_bound_num += 1
        #
        #         if white_pixel_num / total_pixel_num >= white_scale:
        #             point_up[0] = average_white_point[0] / true_bound_num / we
        #             point_up[1] = average_white_point[1] / true_bound_num / he
        #
        #             break
        #
        # for h in range(0, he, int(he/divide)):
        #     point = []
        #     self.intersection(line, [0, 1 - h/he, 1, 1 - h/he], point)
        #     true_bound_num = 0
        #     if 0 <= point[0] < 1 and 0 <= point[1] < 1:
        #         point[0] *= we
        #         point[1] *= he
        #         white_pixel_num = 0
        #         average_white_point = [0, 0]
        #         total_pixel_num = 0
        #         for i in range(int(point[0]) - search_dist, int(point[0]) + search_dist):
        #             if 0 <= i < we:
        #                 for j in range(int(point[1]) - search_dist, int(point[1]) + search_dist):
        #                     if 0 <= j < he:
        #                         total_pixel_num += 1
        #                         color = image[j][i]
        #                         if color[0] > white_min and color[1] > white_min and color[2] > white_min:
        #                             white_pixel_num += 1
        #                         if edges[j][i] == 255:
        #                             average_white_point[0] += i
        #                             average_white_point[1] += j
        #                             true_bound_num += 1
        #
        #         if white_pixel_num / total_pixel_num >= white_scale:
        #             point_down[0] = average_white_point[0] / true_bound_num / we
        #             point_down[1] = average_white_point[1] / true_bound_num / he
        #
        #             break
        #
        # print(point_up, point_down)
        # if point_up[0] + point_up[1] > 0 and point_down[0] + point_down[1] > 0:
        #     if 0.5 < (line[3] - line[1]) * (point_down[0] - point_up[0]) / (line[2] - line[0]) / (point_down[1] - point_up[1]) < 2:
        #         return self.extendLine([point_up[0], point_up[1], point_down[0], point_down[1]])
        # else:
        #     return line

    def updateVerticalLines(self, image, vertical_line_id_list, final_line_list, search_dist, divide):

        lines = []

        for line_id in vertical_line_id_list:
            line = final_line_list[line_id]
            y1 = line[1]
            y2 = line[3]

            if y1 < 0:
                line[0] = y1 / (y1 - y2)
                line[1] = 0
            elif y1 > 1:
                line[0] = (1 - y1) / (y2 - y1)
                line[1] = 1

            if y2 < 0:
                line[2] = y1 / (y1 - y2)
                line[3] = 0
            elif y2 > 1:
                line[2] = (1 - y1) / (y2 - y1)
                line[3] = 1

            lines.append(line)

        return self.getTrueVerticalLine(image, lines, search_dist, divide)

        return new_vertical_lines

    def updateHogLine(self, image, hog_line, search_dist, num_find_point):
        he, we, _ = image.shape

        average_h = (hog_line[1] + hog_line[3]) * he / 2

        image_red = np.zeros((image.shape[0], image.shape[1]))
        for i in range(we):
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 0 <= j < he:
                    color = image[j][i]
                    if int(color[2]) > 150 and int(color[0]) < 160 and int(color[1]) < 160:
                        image_red[j][i] = 255

        remove_points = []
        for i in range(1, we - 1):
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 0 <= j < he:
                    if image_red[j][i] == 255:
                        have_left_red = False
                        have_right_red = False
                        if image_red[j-1][i-1] == 255 or image_red[j][i-1] == 255 or image_red[j+1][i-1] == 255:
                            have_left_red = True
                        if image_red[j-1][i+1] == 255 or image_red[j][i+1] == 255 or image_red[j+1][i+1] == 255:
                            have_right_red = True

                        if not (have_left_red and have_right_red):
                            remove_points.append([i, j])

        for point in remove_points:
            image_red[point[1]][point[0]] = 0

        left_red_point = [0, 0]
        right_red_point = [0, 0]

        red_point_num = 0
        for i in range(1, we - 1):
            if red_point_num > num_find_point:
                break
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 1 <= j < he - 1:
                    if image_red[j][i] == 255:
                        red_point_num += 1
                        left_red_point[0] += i
                        left_red_point[1] += j
                    if red_point_num > num_find_point:
                        break
        left_red_point[0] /= red_point_num * we
        left_red_point[1] /= red_point_num * he

        red_point_num = 0
        for i in range(1, we - 1):
            if red_point_num > num_find_point:
                break
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 0 <= j < he:
                    if image_red[j][we - i] == 255:
                        red_point_num += 1
                        right_red_point[0] += we - i
                        right_red_point[1] += j
                    if red_point_num > num_find_point:
                        break
        right_red_point[0] /= red_point_num * we
        right_red_point[1] /= red_point_num * he

        return self.extendLine([left_red_point[0], left_red_point[1], right_red_point[0], right_red_point[1]])

    def updateBaseLine(self, image, base_line, search_dist, num_find_point):
        he, we, _ = image.shape

        average_h = (base_line[1] + base_line[3]) * he / 2

        image_black = np.zeros((image.shape[0], image.shape[1]))
        for i in range(we):
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 0 <= j < he:
                    color = image[j][i]
                    if int(color[2]) < 70 and int(color[0]) < 70 and int(color[1]) < 70:
                        image_black[j][i] = 255

        for i in range(we):
            find_black = False
            for j in range(int(average_h) + search_dist, int(average_h) - search_dist, -1):
                if 0 <= j < he:
                    if image_black[j][i] == 255:
                        if not find_black:
                            find_black = True
                        else:
                            image_black[j][i] = 0

        remove_points = []
        for i in range(1, we - 1):
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 1 <= j < he - 1:
                    if image_black[j][i] == 255:
                        have_left_red = False
                        have_right_red = False
                        if image_black[j - 1][i - 1] == 255 or image_black[j][i - 1] == 255 or image_black[j + 1][
                            i - 1] == 255:
                            have_left_red = True
                        if image_black[j - 1][i + 1] == 255 or image_black[j][i + 1] == 255 or image_black[j + 1][
                            i + 1] == 255:
                            have_right_red = True

                        if not (have_left_red and have_right_red):
                            remove_points.append([i, j])

        for point in remove_points:
            image_black[point[1]][point[0]] = 0

        left_black_point = [0, 0]
        right_black_point = [0, 0]

        red_point_num = 0
        for i in range(1, we - 1):
            if red_point_num > num_find_point:
                break
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 1 <= j < he - 1:
                    if image_black[j][i] == 255:
                        red_point_num += 1
                        left_black_point[0] += i
                        left_black_point[1] += j
                    if red_point_num > num_find_point:
                        break
        left_black_point[0] /= red_point_num * we
        left_black_point[1] /= red_point_num * he

        red_point_num = 0
        for i in range(1, we - 1):
            if red_point_num > num_find_point:
                break
            for j in range(int(average_h) - search_dist, int(average_h) + search_dist):
                if 0 <= j < he:
                    if image_black[j][we - i] == 255:
                        red_point_num += 1
                        right_black_point[0] += we - i
                        right_black_point[1] += j
                    if red_point_num > num_find_point:
                        break
        right_black_point[0] /= red_point_num * we
        right_black_point[1] /= red_point_num * he

        return self.extendLine([left_black_point[0], left_black_point[1], right_black_point[0], right_black_point[1]])