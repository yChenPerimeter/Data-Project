# -*- coding: utf-8 -*-
"""
 model_evaluation.py Originally Created on Fri Apr 21 09:49:56 2023
Modified on Thursday, Oct 05, 2023, from model_evaluation.py from MarkNguyen

@author: Youwei Chen
"""


import torch
torch.cuda.empty_cache()
from torchinfo import summary
import numpy as np
# from model import  ImgAssistCNN, U_Net, UNet , UDnCNN_old
D = 10
C = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

import subprocess

def get_gpu_usage():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    return float(result.strip())


#python3 cGAN_model_evaluation.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEpoch120Resnet9  --model pix2pix  --preprocess none --netG resnet_9blocks --direction BtoA --epoch 70


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    net_list = model.get_net()
    net = net_list[0]
    model_stats = summary(net, (1,1 , 420, 2400))
    summary_str = str(model_stats)
    print(summary_str)
    tensor = torch.rand((1,1 , 420, 2400)).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    gpu_usage = np.zeros((repetitions,1))
    if opt.eval:
        net.to(device)
        # model.eval()
        net.eval()
    for rep  in range(repetitions):
        print(f"processing rep{rep}")
        cur_usage = get_gpu_usage()
        # print(cur_usage, "percent")
        
        gpu_usage[rep] = cur_usage
        #model.set_input(rep) 
        starter.record()
        #model.test()  
        _ = net(tensor)
  
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    mean_gpu =  np.sum(gpu_usage) / repetitions
    print("time ",mean_syn, "ms")
    print("avg usage in percentage ",mean_gpu ,"percent" )
