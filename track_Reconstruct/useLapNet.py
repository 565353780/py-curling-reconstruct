import torch
import numpy as np
import time
import os
import argparse

import cv2

from LapNet import LAPNet
from collections import OrderedDict
from torch.nn.parameter import Parameter
from create_dataset import createDataset


class LapNetResult:
    def __init__(self, ShowTimeSpend=False):
        self.model = None
        self.INPUT_CHANNELS = 3
        self.OUTPUT_CHANNELS = 2
        self.SIZE = [1024, 512]  # [224, 224]
        self.GPU_IDX = 0
        self.ShowTimeSpend = ShowTimeSpend

        torch.cuda.set_device(self.GPU_IDX)

        self.load_model()


    def state_dict(self, model, destination=None, prefix='', keep_vars=False):
        own_state = model.module if isinstance(model, torch.nn.DataParallel) \
            else model
        if destination is None:
            destination = OrderedDict()
        for name, param in own_state._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in own_state._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in own_state._modules.items():
            if module is not None:
                self.state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
        return destination


    def load_state_dict(self, model, state_dict, strict=True):
        own_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
            else model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


    def load_model(self):
        if self.ShowTimeSpend:
            tstart = time.time()

        # model = SegNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS).cuda()
        self.model = LAPNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS, internal_ch=8).cuda()

        current_file_list = os.listdir(os.getcwd() + '/trained_model')
        current_epoch_num = -1
        for file_name in current_file_list:
            if 'LapNet_chkpt_better_epoch' in file_name:
                temp_epoch_num = int(file_name.split('_')[3].split('h')[1])
                if temp_epoch_num > current_epoch_num:
                    current_epoch_num = temp_epoch_num
        chkpt_filename = os.getcwd() + '/trained_model/' + "LapNet_chkpt_better_epoch" + str(
            current_epoch_num) + "_GPU" + str(self.GPU_IDX) + ".pth"
        if not os.path.exists(os.getcwd() + '/trained_model'):
            os.mkdir(os.getcwd() + '/trained_model')
        if os.path.isfile(chkpt_filename):
            checkpoint = torch.load(chkpt_filename)
            start_epoch = checkpoint['epoch']
            print("Found Checkpoint file", chkpt_filename, ".")

            self.model.load_state_dict(checkpoint['net'])
            self.load_state_dict(self.model, self.state_dict(self.model))

        print("Found", torch.cuda.device_count(), "GPU(s).", "Using GPU(s) form idx:", self.GPU_IDX)

        # model = torch.nn.DataParallel(model)  # = model.cuda()
        # model.eval()

        if self.ShowTimeSpend:
            print('Spend time on load model : ', time.time() - tstart)


    def get_result(self, image_path, projected_target_caller_line, max_dist_scale):
        if self.ShowTimeSpend:
            tstart = time.time()

        train_dataset = createDataset(image_path, size=self.SIZE)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, pin_memory=True,
                                                       shuffle=False, num_workers=0)

        img = list(enumerate(train_dataloader))[0][1]

        img_tensor = torch.tensor(img).cuda()

        if self.ShowTimeSpend:
            print('----Spend time on loadImage : ', int((time.time() - tstart)*1000), ' ms')
            tstart = time.time()

        # Predictions
        sem_pred = self.model(img_tensor)

        if self.ShowTimeSpend:
            print('----Spend time on getPredTensor : ', int((time.time() - tstart) * 1000), ' ms')
            tstart = time.time()

        # sem_pred=torch.floor(sem_pred)
        seg_map = torch.squeeze(sem_pred, 0).cpu().detach().numpy()

        seg_show = seg_map[1]
        (h,w)=seg_show.shape

        if self.ShowTimeSpend:
            print('----Spend time on getSegMap : ', int((time.time() - tstart) * 1000), ' ms')
            tstart = time.time()

        # _, seg_show2 = cv2.threshold(seg_show + 1, 0, 0, cv2.THRESH_TOZERO)
        # seg_show2 = cv2.normalize(seg_show2, seg_show2, 0, 1, cv2.NORM_MINMAX)
        # seg_show2 = cv2.convertScaleAbs(seg_show2, seg_show2, 255)
        # result_img = cv2.applyColorMap(seg_show2, cv2.COLORMAP_MAGMA)
        # cv2.imshow('test', result_img)

        [x1, y1, x2, y2] = projected_target_caller_line

        if x1 < x2:
            x_min = x1
            x_max = x2
        else:
            x_min = x2
            x_max = x1

        if y1 < y2:
            y_min = y1
            y_max = y2
        else:
            y_min = y2
            y_max = y1

        # test_img1 = np.zeros((h, w))
        # for i in range(h):
        #     for j in range(w):
        #         if seg_show[i][j] > -2:
        #             test_img1[i][j] = 255
        # cv2.imshow('test1', test_img1)
        # cv2.waitKey()

        result_img = np.zeros((h, w))

        dx = projected_target_caller_line[2] - projected_target_caller_line[0]
        dy = projected_target_caller_line[3] - projected_target_caller_line[1]

        if y_max - y_min >= x_max - x_min:
            dx = dx * w / dy / h
            j_now = projected_target_caller_line[0]
            if dy < 0:
                j_now = projected_target_caller_line[2]
            j_now = j_now*w - dx*max_dist_scale*h

            for i in range(int((y_min - max_dist_scale)*h), int((y_max + max_dist_scale)*h)):
                if 0 <= i < h:
                    for j in range(int(j_now - max_dist_scale*w), int(j_now + max_dist_scale*w)):
                        if 0 <= j < w:
                            if seg_show[i][j] > -2:
                                result_img[i][j] = 255
                j_now += dx
        else:
            dy = dy * h / dx / w
            i_now = projected_target_caller_line[1]
            if dy < 0:
                i_now = projected_target_caller_line[3]
            i_now = i_now * h - dy * max_dist_scale * w

            for j in range(int((x_min - max_dist_scale) * w), int((x_max + max_dist_scale) * w)):
                if 0 <= j < w:
                    for i in range(int(i_now - max_dist_scale * h), int(i_now + max_dist_scale * h)):
                        if 0 <= i < h:
                            if seg_show[i][j] > -2:
                                result_img[i][j] = 255
                i_now += dy

        # cv2.imshow('result_image', result_img)
        # cv2.waitKey()

        if self.ShowTimeSpend:
            print('----Spend time on chooseResultOnTargetArea : ', int((time.time() - tstart) * 1000), ' ms')
            tstart = time.time()

        target_point_list = []
        target_point_dict={}
        for i in range(h):
            for j in range(w):
                #if result_img[i][j][0] + result_img[i][j][1] + result_img[i][j][2] > 100:
                if result_img[i][j] > 0:
                    target_point_list.append([i, j])
                    key=(int(i/20)*10000+int(j/20))
                    if key in target_point_dict:
                        target_point_dict[key].append([i,j])
                    else:
                        target_point_dict[key]=[[i,j]]
        # print("target_keys:",len(target_point_dict.keys()),target_point_dict.keys())
        new_dict={}
        keydiff=[1,-1,10000,-10000,10000+1,10000-1,-10000+1, -10000-1]
        for key in target_point_dict:
            away=True
            key_find=0
            for j in range(8):
                key_new=key+keydiff[j]
                if(not key_new in new_dict):
                    continue
                for point in target_point_dict[key]:
                    for point_new in new_dict[key_new]:
                        if (point_new[0]-point[0])*(point_new[0]-point[0])+(point_new[1]-point[1])*(point_new[1]-point[1])<100:
                            away=False
                            key_find=key_new
                            break
                    if away==False:
                        break
                if  away==False:
                    break
            if away:
                new_dict[key]=target_point_dict[key]
            else:
                new_dict[key_find].extend(target_point_dict[key])

        # merged_target_point_lists = []
        #
        # for point in target_point_list:
        #     saved = False
        #     for merged_point_list in merged_target_point_lists:
        #         merged_point = merged_point_list[0]
        #         if (merged_point[0] - point[0])*(merged_point[0] - point[0]) + (merged_point[1] - point[1])*(merged_point[1] - point[1]) < 1000:
        #             merged_point_list.append(point)
        #             saved = True
        #             break

        #     if not saved:
        #         merged_target_point_lists.append([point])

        final_merged_target_point_list = []

        for key in new_dict:
            avgpoint = [0, 0]
            for point in new_dict[key]:
                avgpoint[0] += point[1]
                avgpoint[1] += point[0]
            num=len(new_dict[key])
            avgpoint[0] /= num*w
            avgpoint[1] /= num*h

            final_merged_target_point_list.append([avgpoint, [8/w, 6/h], 1])

        if self.ShowTimeSpend:
            print('----Spend time on getAllRocks : ', int((time.time() - tstart) * 1000), ' ms')



        # for merged_point_list in merged_target_point_lists:
        #     avgpoint = [0, 0]

        #     for point in merged_point_list:
        #         avgpoint[0] += point[1]
        #         avgpoint[1] += point[0]

        #     avgpoint[0] /= len(merged_point_list)*w
        #     avgpoint[1] /= len(merged_point_list)*h

        #     final_merged_target_point_list.append([avgpoint, [10, 10], 0])

        return final_merged_target_point_list