import torch
import numpy as np
import time
import os
import argparse
import cv2

from LapNet import LAPNet
from loss import DiscriminativeLoss
from shougang_dataset import ShougangDataset
from logger import Logger
from torch.nn import DataParallel
from collections import OrderedDict
from torch.nn.parameter import Parameter
import platform

parser = argparse.ArgumentParser(description="Train model")

parser.add_argument('--dataset-path', default='ShougangDataset/')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--batch-size', type=int, default=24, help='batch size')
parser.add_argument('--img-size', type=int, nargs='+', default=[1024, 512], help='image resolution: [width height]')
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--gpu-idx',type = int,default= 0, help='using gpu(idx)')
parser.add_argument('--optimizer-reset', type=int, default=100)


args = parser.parse_args()

torch.cuda.set_device(args.gpu_idx)

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 2
LEARNING_RATE = args.lr #1e-5
BATCH_SIZE = args.batch_size #20
NUM_EPOCHS = args.epoch #100
LOG_INTERVAL = 20
INS_CH = 32
SIZE = [args.img_size[0], args.img_size[1]] #[224, 224]

def state_dict(model, destination=None, prefix='', keep_vars=False):
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
            state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
    return destination

def load_state_dict(model, state_dict, strict=True):
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
            

def train(model):
    # refer from : https://github.com/Sayan98/pytorch-segnet/blob/master/src/train.py
    is_better = True
    prev_loss = float('inf')

    print("Found",torch.cuda.device_count(),"GPU(s).","Using GPU(s) form idx:",args.gpu_idx)
    
    #model = DataParallel(cpu_model)
    # model = cpu_model.cuda()
    #device = torch.device("cuda")
    # model = torch.nn.DataParallel(model) #= model.cuda()
    model.train()

    last_better_epoch = start_epoch

    for epoch in range(start_epoch,NUM_EPOCHS):

        t_start = time.time()
        loss_f = [] 
        for batch_idx, (imgs, sem_labels) in enumerate(train_dataloader):
            #os.system("clear")

            loss = 0
            

            img_tensor = torch.tensor(imgs).cuda()
            sem_tensor = torch.tensor(sem_labels).cuda()
           # ins_tensor = torch.tensor(ins_labels).cuda()
            # Init gradients
            optimizer.zero_grad()
            img_inpt = np.array(np.transpose(torch.squeeze(img_tensor[0],0).cpu().detach().numpy(), (1,2,0)) ,dtype=np.uint8)
            # Predictions
            sem_pred = model(img_tensor)
            # sem_pred=torch.floor(sem_pred)
            seg_map = torch.squeeze(sem_pred,0).cpu().detach().numpy()
            # ins_map = torch.squeeze(ins_pred,0).cpu().detach().numpy()
            
            # Discriminative Loss
            # disc_loss = criterion_disc(ins_pred, ins_tensor, [INS_CH] * len(img_tensor))/6400.5

            # CrossEntropy Loss

            ce_loss = criterion_ce(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS),
                                   sem_tensor.view(-1))
            # print(
            # np.shape(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS)[:,1]),
            # np.shape(sem_tensor.view(-1).float())
            # )
            # mse = criterion_mse(sem_pred.permute(0,2,3,1).contiguous().view(-1,OUTPUT_CHANNELS)[:,1],sem_tensor.view(-1).float())/1000

            loss = ce_loss #+ disc_loss
                
            loss.backward()
            optimizer.step()

            loss_f.append(loss.cpu().data.numpy())
            print('    Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()),end = '\r')
            seg_show = seg_map[0][1]


            _, seg_show2 = cv2.threshold(seg_show+1, 0, 0,cv2.THRESH_TOZERO)
            seg_show = cv2.normalize(seg_show,seg_show,0,1,cv2.NORM_MINMAX)
            seg_show2 = cv2.normalize(seg_show2,seg_show2,0,1,cv2.NORM_MINMAX)
            seg_show = cv2.convertScaleAbs(seg_show,seg_show,255)
            seg_show2 = cv2.convertScaleAbs(seg_show2,seg_show2,255)
            # cv2.imshow("seg_pred",cv2.addWeighted(img_inpt,0.5,cv2.applyColorMap(seg_show,cv2.COLORMAP_JET),0.5,0))
            # cv2.imshow("colormap",cv2.applyColorMap(seg_show,cv2.COLORMAP_JET))
            # cv2.imshow("segthresh",cv2.applyColorMap(seg_show2,cv2.COLORMAP_MAGMA))
            # for i in range(32):
            #     ins_show = ins_map[0][i]
            #     ins_show = cv2.normalize(ins_show,ins_show,0,1,cv2.NORM_MINMAX)
            #     ins_show = cv2.convertScaleAbs(ins_show,ins_show,255)
            #     cv2.imshow("insmap"+str(i),cv2.applyColorMap(ins_show,cv2.COLORMAP_OCEAN))
            


            # cv2.imshow("img_inpt",img_inpt)
            if cv2.waitKey(1) == 27:
                print("Saving current chkpt...")
                state = {'net':state_dict(model), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, chkpt_filename)
                exit()
                
            #Tensorboard
            # if batch_idx % LOG_INTERVAL == 0:
            #     print("log at train idx:",batch_idx,end='\r')
            #     info = {'loss': loss.item(), 'ce_loss': ce_loss.item(), 'epoch': epoch}
            #     for tag, value in info.items():
            #         logger.scalar_summary(tag, value, batch_idx + 1)
            #     # 2. Log values and gradients of the parameters (histogram summary)
            #     for tag, value in model.named_parameters():
            #         tag = tag.replace('.', '/')
            #         logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx + 1)
            #           # logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx + 1)
                
            # if batch_idx % 100 == 0:
            #     torch.save(model.state_dict(), "Lap"+str(args.gpu_idx)+".pth")
            #     print("\t\tModel Saved.")
                # # 3. Log training images (image summary)
                # info = {'images': img_tensor.view(-1, 3, SIZE[1], SIZE[0])[:BATCH_SIZE].cpu().numpy(),
                #         'labels': sem_tensor.view(-1, SIZE[1], SIZE[0])[:BATCH_SIZE].cpu().numpy(),
                #         'sem_preds': sem_pred.view(-1, 2, SIZE[1], SIZE[0])[:BATCH_SIZE,1].data.cpu().numpy(),
                #         'ins_preds': ins_pred.view(-1, SIZE[1], SIZE[0])[:BATCH_SIZE*5].data.cpu().numpy()}

                # for tag, images in info.items():
                #     logger.image_summary(tag, images, batch_idx + 1)
            
        dt = time.time() - t_start
        is_better = np.mean(loss_f) < prev_loss
        
        state = {'net':state_dict(model), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        # torch.save(state, os.getcwd()+"/trained_model/"+"LapNet_chkpt_epoch"+str(epoch)+"_GPU"+str(args.gpu_idx)+".pth")

        if is_better:
            prev_loss = np.mean(loss_f)
            print("\t\tBest Model.")
            torch.save(state, os.getcwd()+"/trained_model/"+"LapNet_chkpt_better_epoch"+str(epoch)+"_GPU"+str(args.gpu_idx)+".pth")
            if last_better_epoch < epoch:
                os.remove(os.getcwd()+"/trained_model/"+"LapNet_chkpt_better_epoch"+str(last_better_epoch)+"_GPU"+str(args.gpu_idx)+".pth")
                last_better_epoch = epoch
            
        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s, Lr: {:2f}".format(epoch+1, np.mean(loss_f), dt, optimizer.param_groups[0]['lr']))


if __name__ == "__main__":
    logger = Logger('./logslite'+str(args.gpu_idx))

    dataset_path = args.dataset_path
    train_dataset = ShougangDataset(dataset_path, size=SIZE)

    workers_num = 0
    if platform.system() == 'Linux':
        workers_num = 20

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=workers_num)
    #model = SegNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS).cuda() 
    model = LAPNet(input_ch=INPUT_CHANNELS, output_ch=OUTPUT_CHANNELS,internal_ch = 8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), amsgrad=True)
    start_epoch = 0
    # chkpt_filename = "LapNet_chkpt_default"+"_GPU"+str(args.gpu_idx)+".pth"
    current_file_list = os.listdir(os.getcwd()+'/trained_model')
    current_epoch_num = -1
    for file_name in current_file_list:
        if 'LapNet_chkpt_better_epoch' in file_name:
            temp_epoch_num = int(file_name.split('_')[3].split('h')[1])
            if temp_epoch_num > current_epoch_num:
                current_epoch_num = temp_epoch_num
    chkpt_filename = os.getcwd()+'/trained_model/'+"LapNet_chkpt_better_epoch" + str(current_epoch_num) + "_GPU" + str(args.gpu_idx) + ".pth"
    if not os.path.exists(os.getcwd()+'/trained_model'):
        os.mkdir(os.getcwd()+'/trained_model')
    if os.path.isfile(chkpt_filename):
        checkpoint = torch.load(chkpt_filename)
        start_epoch = checkpoint['epoch']
        print("Found Checkpoint file",chkpt_filename,".")
        print("The checkpoint was saved at epoch",checkpoint['epoch'],".")
        print("Taining stats is reset form epoch",start_epoch)
        if(args.optimizer_reset != 1):
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Optimizer State Reset.")
        model.load_state_dict(checkpoint['net'])
        load_state_dict(model, state_dict(model))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,400,1000,6000,7000,8000], gamma=0.9)
    # 

    criterion_ce = torch.nn.CrossEntropyLoss()
    #    criterion_mse = torch.nn.MSELoss(reduce=True, size_average=True)
#    criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                        delta_dist=1.5,
#                                        norm=2,
#                                        usegpu=True)

    train(model)
