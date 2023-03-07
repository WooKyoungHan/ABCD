import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
from datasets import register

from utils import to_pixel_samples
from utils import make_coord
from utils import quantization,quantization16,zeropadding





@register('A_B_Coefficients')
class A_B_Coefficients(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=None, scale_max=None,inpdepth=4,
                 augment=False, sample_q=None, arbit=False,gtdepth = 16,arbit_gt=False,scale_maxgt=None,scale_mingt=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.arbit = arbit
        self.gtdepth = gtdepth
        self.inpdepth = inpdepth
        self.arbit_gt = arbit_gt
        self.scale_maxgt=scale_maxgt
        self.scale_mingt=scale_mingt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        
        if self.arbit_gt is False:
            t = self.gtdepth
        else:
            t = random.randint(self.scale_mingt,self.scale_maxgt)
            
        if self.arbit is False:
            s=self.inpdepth
        else:
            s=random.randint(self.scale_min,self.scale_max)
        
        if self.inp_size is None:
            img_hbd = quantization16(img, t)
            img_lbd = quantization16(img, s)
            crop_lr, crop_hr = img_lbd, img_hbd
        else:
            w_hr = round(self.inp_size)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            
            crop_hr_ = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_hr = quantization16(crop_hr_, t)
            crop_lr = quantization16(crop_hr_, s)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        crop_lr = zeropadding(crop_lr,t,s)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            lr_rgb = lr_rgb[sample_lst]
            
        bit_query = (2**(t-s))/(2**t-1)
        cell = torch.ones_like(hr_coord[:,0]).unsqueeze(-1)
        cell = cell*bit_query
        
        coefficient = (hr_rgb-lr_rgb)/bit_query

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': coefficient,
            'valid_inp' : lr_rgb
        }    



@register('ABCD_test')
class A_B_Coefficients_test(Dataset):
    def __init__(self, dataset,inpdepth=4,gtdepth = 16):
        self.dataset = dataset
        self.gtdepth = gtdepth
        self.inpdepth = inpdepth

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        
        t = self.gtdepth            
        s=self.inpdepth
        
        crop_hr = quantization16(img, t)
        crop_lr = quantization16(img, s)        
        crop_lr = zeropadding(crop_lr,t,s)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())
            
        bit_query = (2**(t-s))/(2**t-1)
        cell = torch.ones_like(hr_coord[:,0]).unsqueeze(-1)
        cell = cell*bit_query
        
        coefficient = (hr_rgb-lr_rgb)/bit_query

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': coefficient,
            'valid_inp' : lr_rgb
        }    


    