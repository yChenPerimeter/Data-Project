# -*- coding: utf-8 -*-
"""
@Author: ychen
Date: 2024-07-25
Purpose: script the model weights for inference based on options

python3 scriptModel.py  --dataroot ./datasets/capstone --name capstoneDCIS_cyclegan_batch4 --model cycle_gan --epoch latest
python3 scriptModel.py --dataroot ./datasets/cGAN_input_uint8_O21_CV_PL00001_13_01_16 --name production_uint8_O21CVPL00001_13_01_16  --model pix2pix --direction AtoB --epoch 25 --preprocess none --netG resnet_9blocks --netD pixel  --data_bit 8 

python3 scriptModel.py --dataroot ./datasets/Adipose_IDC_20240813  --name Adipose_IDC_cyclegan --model cycle_gan 

"""

import torch
import torch.nn as nn
import os
import sys

# Add the root directory to sys.path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.deploy_options import DeployOptions  # Import TestOptions class
from models import create_model  # Import from models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def script_model(checkpoint):
                
    if isinstance(checkpoint, torch.nn.DataParallel):
        scripted_modelD = torch.jit.script(checkpoint.module)
    else:
        scripted_modelD = torch.jit.script(checkpoint)
    return scripted_modelD

def main():
    """Main function to optimize model weights for inference."""
    opt = DeployOptions().parse()  # get test options from command-line arguments

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    model.eval()
    
    nets = model.get_net()
    if isinstance(nets, torch.nn.DataParallel):
        print("Warning: accessing wrong net type found.")
        print(type(nets))
    

    f_name = opt.name
    epo = str(opt.epoch)


    if nets:
        lns = len(nets)
        print(f'----------  Networks number: {lns} -------------')

        if opt.model == "cycle_gan" :
            net_G_A = script_model(nets[0])
            net_G_B = script_model(nets[1])
            
            net_G_A.save(os.path.join(opt.scripted_checkpoints_dir, f"{f_name}_G_A_checkpoints_scripted{epo}.pt"))
            net_G_B.save(os.path.join(opt.scripted_checkpoints_dir, f"{f_name}_G_B_checkpoints_scripted{epo}.pt"))
        else:
            first_net = nets[0]
            
            # if isinstance(first_net, torch.nn.DataParallel):
            #     scripted_modelD = torch.jit.script(first_net.module)
            # else:
            #     scripted_modelD = torch.jit.script(first_net)
            scripted_modelD = script_model(first_net)
            optimized_model_path = os.path.join(opt.scripted_checkpoints_dir, f"{f_name}_checkpoints_scripted{epo}.pt")
            scripted_modelD.save(optimized_model_path)
    else:
        print("No networks found.")
        return



    if opt.quantize:
        quantize_precision = opt.precision
        if quantize_precision == 'fp16':
            net = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d}, dtype=torch.float16)
            torch.jit.save(net, optimized_model_path)
        elif quantize_precision == 'int8':
            net = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d}, dtype=torch.qint8)
            torch.jit.save(net, optimized_model_path)
        else:
            print('Invalid quantization precision selected.')

    # if opt.speedtest:
    #     mean_speed, st_dev, max_speed, min_speed = deploy_utils.speed_test(opt, net, torch_input, opt.batch_size)
    #     print(f'\nMean Speed: {mean_speed} ms')
    #     print(f'Standard Deviation: {st_dev} ms')
    #     print(f'Max Speed: {max_speed} ms')
    #     print(f'Min Speed: {min_speed} ms')
    # else:
    #     print('Model speed test not selected.')

if __name__ == '__main__':
    main()
