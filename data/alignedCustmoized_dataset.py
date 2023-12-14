import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import matplotlib.pyplot as plt
import sys

import torchvision as tv

def print_info(arr):
    print("dtype: ", arr.dtype)
    print("range: ", f'({arr.min(), arr.max()})')
    print("shape: ", arr.shape)
    # print("sample: ", arr[0, 0:5, 0:5])
    
class AlignedCustmoizedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        
        # self.dir_test = os.path.join("/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/20231128Float_A_B_v4/test")  # get the image directory
        # self.dir_A = os.path.join(self.dir_test, "1x")  # get the image directory
        # self.dir_B = os.path.join(self.dir_test, "8x")  # get the GT directory
        
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]

        #TODO chaning 
        # AB = Image.open(AB_path).convert('RGB')
        
        print("plt",AB_path)
        AB = plt.imread(AB_path)
        
        # AB = plt.imread ("/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_float_20231128_v4/test/grapeA/average_A-scanAvg_1.png")
        
        #take the first channel
        AB = AB[:,:,0]
        # print( AB.shape)
        # split AB image into A and B
        h, w = AB.shape[0], AB.shape[1]
        w2 = int(w / 2)
        
        A = AB[0:h,0:w2] # 8x
      
        B = AB[0:h,w2:w] # 1x 
        
        
        #Test A=B
        # B =  plt.imread("/home/david/workingDIR/datasets/Paired_FloatGWAD_CNGT_brain_v4/test/Grape/A-scanAvg/1x/average_A-scanAvg_1.png")[:,:,0]
        # A = B
        
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt,  (A.shape[0], A.shape[1]))
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        
        # print_info(B)
        
        # oriA = plt.imread("/home/david/workingDIR/datasets/Paired_FloatGWAD_CNGT_brain_v4/test/Grape/A-scanAvg/1x/average_A-scanAvg_1.png")
        # print_info(oriA)
        
        # plt.imsave("/home/david/workingDIR/ImgClear/Research/Image_test_data/review_data/1B.png", B, cmap = 'Greys')
        # sys.exit(1)
        
       
        
        
        A_transform = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((.5), (.5))])
        B_transform = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((.5), (.5))])

        A = A_transform(A)
        B = B_transform(B)
        
        

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
