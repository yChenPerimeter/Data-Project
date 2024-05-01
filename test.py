"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""


##!./scripts/test_pix2pix.sh python test.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pix300E256Unet --model pix2pix --direction BtoA --epoch 250

"""
Non Eval mode
"""
#python test.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEpoch120Resnet9  --model pix2pix  --preprocess none --netG resnet_9blocks --direction BtoA --epoch 70


# python test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/synthesis_data--name CNG5000TA_p2pEp40_Resnet9_pixel_initType_Kaiming  --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming
# --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming


# python test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_synesis_input --name 7kOrganics_p2pEp30_Resnet9_pixel_KaimingLr15 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming 

#python test.py --dataroot /home/david/Projects/de_noise/pytorch-CycleGAN-and-pix2pix/datasets/syntheis_data --name 7kOrganics_p2pEp30_Resnet9_pixel_KaimingLr15 --model pix2pix --direction BtoA --epoch 23  --preprocess none --netG resnet_9blocks --netD pixel  


"""
non Eval mode
sudo chown david ./checkpoints/v4_FloatTest_lr10-4/test_opt.txt


python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-5 --model pix2pix --direction BtoA --epoch 60  --preprocess none --netG resnet_9blocks --netD pixel  
python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel 

python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-3 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel 

python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4 --model pix2pix --direction BtoA --epoch 50  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned 

python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch 31  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned  
python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch 40  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned  
python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch 8  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode alignedCustmoized   

# Generate denoised 8x images for student model
python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction AtoB --epoch 39  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode alignedCustmoized   
python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction AtoB --epoch 39  --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode alignedCustmoized  
"""

"""
20230317 - thesis

python test.py --dataroot ./datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch 52  --preprocess none --netG resnet_9blocks --netD pixel  
"""

#sys.exit(1)
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_FloatGrayImages
from util import html
import sys

import torch
import torch.nn as nn
from torchvision.models import inception_v3
# from ignite.metrics import FID
# from ignite.engine import Engine


from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg
import sys

from PIL import Image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')





# Define the InceptionV3 model for feature extraction
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, transform_input=False):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=transform_input)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# Define the evaluation function
# def eval_step(engine, batch):
#     return  batch

# Function to calculate FID
def calculate_fid(act1, act2):
    """ Calculate FID between two sets of activations """
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




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


    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        print("eval mode")
        model.eval()
    

        
    # Define the device for computation
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else "cpu"
    # Initialize the feature extractor
    feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()
    # Lists to store features
    real_features = []
    fake_features = []
    input_features = []
    
    
    
    #TODO net
    # print(f"Loading scripted model epoch {opt.epoch}")
    # net = torch.jit.load(f"/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/v4_FloatTest_lr10-4_batch1/v4_FloatTest_lr10-4_batch1_checkpoints_scripted{opt.epoch}.pt")
    # net.to(device)
    # net.eval()
    
    
    
    """Unit Tests"""
    #Check model differences vs Jit, same results
    # print(model.netG)
    # print(net)
    

    
    # Check parameter size diff: Same
    # for param_tensor in (model.netG).state_dict():
    #     print(param_tensor, "\t",(model.netG).state_dict()[param_tensor].size())

    # print("net")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        # print(data)
        model.set_input(data)  # unpack data from data loader, here Pix2Pix model swap defination of A and B, i.e To A is the 1x, B is the 8x
        model.test()           # run inference    
        visuals = model.get_current_visuals()  # get image results, in Dict 'real_A', 'fake_B', 'real_B' : tensor
        img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        
        
    
        
        #TODO comment/uncomment this line to use scripted model or unscripted model
        # y = net.forward(data["B"].to(device))
        # visuals["fake_B"] = y
        
        # print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        save_FloatGrayImages(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=False)

     
        #FID
        y_pred = visuals["fake_B"].to(device).repeat(1, 3, 1, 1)
        y_true = visuals["real_B"].to(device).repeat(1, 3, 1, 1)
        
        x_true = visuals["real_A"].to(device).repeat(1, 3, 1, 1)
        
        # Extract features
        with torch.no_grad():
            pred_features = feature_extractor(y_pred).cpu().numpy()
            true_features = feature_extractor(y_true).cpu().numpy()
            x_features = feature_extractor(x_true).cpu().numpy()
            

        real_features.append(true_features)
        input_features.append(x_features)
        fake_features.append(pred_features)
    
    webpage.save()  # save the HTML
    
    # Convert lists to numpy arrays
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    input_features = np.concatenate(input_features, axis=0)

    # Calculate FID
    fid_value = calculate_fid(real_features, fake_features)
    fid_input = calculate_fid(real_features, input_features)
    # TODO Need to test the input vs the GT 8x will affect work flow or not
    print(f"FID input and GT 8x: {fid_input}")
    # fid_value = calculate_fid(real_features, real_features)
    print(f"FID: {fid_value}")
