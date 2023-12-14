"""Code for script model, adapt from test.py"""

import torch
import torch.nn as nn
import time

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

# Or torch.trace(), checkpoint version, .ptn script version.pt
# python scriptModel.py --dataroot ./datasets/CNG --name CNG_pix2pix --model pix2pix --direction BtoA

# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pix200EResnet6 --model pix2pix --direction BtoA --epoch latest
# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pix200EResnet6 --model pix2pix --direction BtoA --epoch 55
# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pixEpoch120Resnet9 --model pix2pix --direction BtoA --epoch 70  --preprocess none --netG resnet_9blocks 
# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pixEpoch120Resnet9 --model pix2pix --direction BtoA --epoch 140  --preprocess none --netG resnet_9blocks
#python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pixEpoch200Unet128 --model pix2pix --direction BtoA --epoch latest --use_wandb False --netG unet_128
# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_pix2pixEp30_Resnet9_Layer2 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD n_layers --n_layers_D 2
# python scriptModel.py --dataroot ./datasets/CNG_Tomato_Air_GingerDiakon --name CNGTAGD_p2pEp30_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming
# python scriptModel.py --dataroot ./datasets/CNG_MoreTomato_Air --name CNG4TA_p2pEp30_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming

# python scriptModel.py --dataroot ./datasets/CNG_5000Tomato_Air --name CNG5000TA_p2pEp40_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming 

# python scriptModel.py --dataroot ./datasets/GingerWAD_CNG5Ktomato --name OrganicsAirWedges_p2pEp80_Resnet9_pixel_Kaiming --model pix2pix --direction BtoA --epoch 70  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming 

# python scriptModel.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name 7kOrganics_p2pEp30_Resnet9_pixel_KaimingLr15 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming 

"""
New loss, 
# python scriptModel.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming  --wandb_project_name newloss_cGAN
# python scriptModel.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming  --wandb_project_name newloss_cGAN

python scriptModel.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name mssim_l1_bExperiment_on7kData_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming  --wandb_project_name newloss_cGAN
"""
"""
float

python scriptModel.py --dataroot ./datasets/cGAN_input_float_2023114-Controled --name FloatTest_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming  --wandb_project_name Float_cGAN

python3 scriptModel.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch 8  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming  --wandb_project_name Float_cGAN
"""
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
    #convert to a scripted model


    #scripted_modelD.save(f'/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')
    # scripted_modelD.save(f'/home/david/Projects/de_noise/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')
    scripted_modelD.save(f'/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/{f_name}/{f_name}_checkpoints_scripted{epo}.pt')