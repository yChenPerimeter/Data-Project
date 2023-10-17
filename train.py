"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb

"""
 It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}. During test time, you need to prepare a directory '/path/to/data/test'.
"""
# Make sure sett diff wanbd name each run
"""
Traing history log on cGAN architchure 

##python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pix300E256Unet --model pix2pix --direction BtoA 
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pix300E256Unet --model pix2pix --direction BtoA 
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEpoch140Resnet6 --model pix2pix --direction BtoA  --preprocess none
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEpoch120Resnet9 --model pix2pix --direction BtoA  --preprocess none --netG resnet_9blocks
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEpoch200Unet128 --model pix2pix --direction BtoA --netG unet_128
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEp30_Resnet9_Layer2 --model pix2pix --direction BtoA  --preprocess none --netG resnet_9blocks --netD n_layers --n_layers_D 2
#python train.py --dataroot ./datasets/CNG_Tomato_Air --name CNGTA_pix2pixEp30_Resnet9_Layer4 --model pix2pix --direction BtoA  --preprocess none --netG resnet_9blocks --netD n_layers --n_layers_D 4
# python train.py --dataroot ./datasets/CNG_Tomato_Air  --name CNGTA_p2pEp30_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming
# python train.py --dataroot ./datasets/CNG_Tomato_Air_GingerDiakon --name CNGTAGD_p2pEp30_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming

# training image 1570, 30 epoch , 
#python train.py --dataroot ./datasets/CNG_MoreTomato_Air --name CNG4TA_p2pEp30_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming

# train img 5200 images, 40 epoch
# python train.py --dataroot ./datasets/CNG_5000Tomato_Air --name CNG5000TA_p2pEp40_Resnet9_pixel_initType_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 20 --n_epochs_decay 20 

#train img 5500 images, 80 epoch
# python train.py --dataroot ./datasets/GingerWAD_CNG5Ktomato --name OrganicsAirWedges_p2pEp80_Resnet9_pixel_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 40 --n_epochs_decay 40 
# python train.py --dataroot ./datasets/GingerWAD_CNG5Ktomato --name OrganicsAirWedges_p2pEp80_Resnet9_pixel_Kaiming --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 40 --n_epochs_decay 40 

#Train Image 40% non tomato, 30 epoch, 6700 sum, 5k tomato
#python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name 7kOrganics_p2pEp30_Resnet9_pixel_KaimingLr15 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 15 --n_epochs_decay 15  --lr 0.00015

"""

"""
Traing history log on cGAN architchure new loss experiment
#Train Image 40% non tomato, 30 epoch, 6700 sum, 5k tomato
#python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 20 --n_epochs_decay 20  --lr 0.0002 --loss l2 --wandb_project_name newloss_cGAN
#python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 15 --n_epochs_decay 15  --lr 0.0001 --loss l2 --wandb_project_name newloss_cGAN

python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name SSIMlossExperiment_on7kData_lr20-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 15 --n_epochs_decay 15  --lr 0.0002 --loss ssim --wandb_project_name newloss_cGAN 
"""

"""
Traing history log on cGAN architchure , on diff batchsize experiment
#Train Image 40% non tomato, 30 epoch, 6700 sum, 5k tomato
#python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 20 --n_epochs_decay 20  --lr 0.0002 --loss l2 --wandb_project_name newloss_cGAN
#python train.py --dataroot ./datasets/2kGWAD_CNG5Ktomato --name L2lossExperiment_on7kData_lr10-4 --model pix2pix --direction BtoA --epoch latest  --preprocess none --netG resnet_9blocks --netD pixel  --init_type kaiming --n_epochs 15 --n_epochs_decay 15  --lr 0.0001 --loss l2 --wandb_project_name newloss_cGAN

"""
if __name__ == '__main__':
    # 🐝 initialize a wandb run
    # wandb.init(
    #     project="PixToPix_GAN",
    #     name= "PixToPix unet256 MiniMasterDS")
    
    opt = TrainOptions().parse()   # get training options
    #TODO
    print("generator of gan input: ", opt.netG )
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # TODO model contained in train loop, while wandb init in virtulizer
    # wandb.watch(model, log="all", log_freq=100)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
                
                # 🐝 log train_loss for each step to wandb， epoch_len = len(train_ds) // train_loader.batch_size， epoch_len * epoch + step
                # 🐝 log train_loss averaged over epoch to wandb
                #wandb.log({"train loss": losses})
            
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
