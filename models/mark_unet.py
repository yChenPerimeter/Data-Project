"""
Code/original comment author : Mark
Reviewer and additional Comment : Youwei
"""

"""
Discussion Sept 01: 

Compared original setting, 
Unet paper and MONAI
https://docs.monai.io/en/stable/_modules/monai/networks/nets/unet.html#UNet
https://docs.monai.io/en/stable/_modules/monai/networks/blocks/convolutions.html



Current settings: 
1. to reduce parameters: not double convolution 
2. use max_unpool2d, instead of upscale after each conv block in encodr path, bilinear;
3. In decoder path, in convtranspose2d, use padding = 1, instead of 0,
    Kernel size 3 instead of 2. original paper up-conv 2*2 , 
    
    i.e. Current settings:
    kernel size 3, stride 1, padding 1, conv2d; 
    Benefit: Odd-sized kernels (3x3, 5x5, 7x7, etc.) have a center pixel
    
    Result in difference:
     Output size=(Input size - 1)xStride-2xPadding+Kernel size+Output padding
     for a example input size 2 (ex. tensor is (1, 1, 2, 2)
     
    Current settings:
     Given:
        Input size: 2
        Stride: 1
        Padding: 1
        Kernel size: 3
        Output padding: 0 
     Output size=(2-1)*1-2*1+3+0=3
     
    Original settings: MONAI
    Given:
        Input size: 2
        Stride: 2
        Padding: 0
        Kernel size: 3
        Output padding: 0 

    Output size=(2-1)*2-2*0+3+0=5
    
    Conclusion:
    1. Current settings and original settings have diff output size
    2. not only affect the spatial dimensions of the output tensor; it also affects the values 
    3. Changing the stride from 1 to 2 means the kernel will be applied with larger gaps
    4. pad = 1 affect:The convolutional kernel can operate on these padded values, which can influence the output, especially at the edges of the tensor.
    -> MONAI implementation wrong;
"""

import torch
from torch import nn
from torch.nn import functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, is_maxpooling = True):
        super(conv_block,self).__init__()
        self.is_maxpooling = is_maxpooling
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1, padding = 1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)#,
            #nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1, padding = 1,bias=True),
            #nn.BatchNorm2d(ch_out),
            #nn.ReLU(inplace=True)#,
            #nn.MaxPool2d(2, stride=2, return_indices=True)
        )
        self.maxpooling = nn.MaxPool2d(2, stride=2, return_indices=True)


    def forward(self,x):
        x = self.conv(x)
        if self.is_maxpooling: x = self.maxpooling(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, size = None):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            #nn.max_unpool2d(scale_factor=2),
            #nn.ConvTranspose2d(ch_in,ch_out,kernel_size=3,stride=2, padding = 1,output_padding = 1,bias=True),
		    #nn.BatchNorm2d(ch_out),
			#nn.ReLU(inplace=True)
   
            #TODO: check if this is correct
            # Output size=(Input size−1)×Stride−2×Padding+Kernel size+Output padding
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size= 3, stride=1, padding=1, output_padding=0), 
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            #nn.BatchNorm2d(ch_out),
            #nn.ConvTranspose2d(ch_out, ch_out, kernel_size = 3, stride=1, padding=1, output_padding=0), 
            #nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
    
class MarkUNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(MarkUNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)

        # maxpooling after each conv block in encoder path
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128, is_maxpooling=False)

        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64, is_maxpooling=False)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32, is_maxpooling=False)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16, is_maxpooling=False)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1,h1 = self.Conv1(x)

        x2, h2 = self.Conv2(x1)
        x3, h3 = self.Conv3(x2)
        x4, h4 = self.Conv4(x3)
        x5, h5 = self.Conv5(x4)
        
        
        
        # decoding + concat path
        # Upsample: not able to control to output size, ex. 26*2 = 52, 
        # Conv transpose2d : cannot control output size
        # max_unpool2d, new in pytorch 1.3, can control output size, upsampling_factor
        
        d5 = F.max_unpool2d(input=x5,indices= h5, kernel_size=(2, 2),output_size= x4.shape)
        d5 = self.Up5(d5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = F.max_unpool2d(input=d5,indices= h4, kernel_size=(2, 2),output_size= x3.shape)
        d4 = self.Up4(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = F.max_unpool2d(input=d4,indices= h3, kernel_size=(2, 2),output_size= x2.shape)
        d3 = self.Up3(d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = F.max_unpool2d(input=d3,indices= h2, kernel_size=(2, 2),output_size= x1.shape)
        d2 = self.Up2(d2)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = F.max_unpool2d(input=d2,indices= h1, kernel_size=(2, 2),output_size= x.shape)
        d1 = self.Conv_1x1(d1)
        return d1