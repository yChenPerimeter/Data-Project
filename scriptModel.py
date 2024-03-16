"""Code for script model, adapt from test.py"""

import torch
import torch.nn as nn
import time

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer




# Or torch.trace(), checkpoint version, .ptn script version.pt
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers


    model.eval()
    # model.print_networks(True)
    
    
    nets = model.get_net()
    # Check if nets is a DataParallel object
    if isinstance(nets, torch.nn.DataParallel):
        print("Warning:accessing wrong net type found.")
        print(type(nets))
  
    if nets:
        lns = len(nets)
        print(f'----------  Networks number:{lns} -------------')
        first_net = nets[0]
        
        # We can torch_tensort.compile(model,input = [],enabled_precisions = torch.half")
        if isinstance(first_net, torch.nn.DataParallel):
            scripted_modelD = torch.jit.script(first_net.module)  # Access the underlying module of DataParallel
        else:
            scripted_modelD = torch.jit.script(first_net)
    else:
        print("No networks found.")

    f_name = opt.name
    epo = str(opt.epoch)
    #convert to a scripted model, 


    #scripted_modelD.save(f'/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')
    # scripted_modelD.save(f'/home/david/Projects/de_noise/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')
    scripted_modelD.save(f'/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')