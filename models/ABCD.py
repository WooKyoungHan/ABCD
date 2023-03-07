import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
import math

import numpy as np 

    
@register('ABCD')
class ABCD(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None , hidden_dim=256):
        super().__init__()        
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)  
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim+1})#hidden_dim//2})
        self.clipped = nn.Tanh()
        
    def gen_feat(self, inp):
        self.inp = inp       
        self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq
        coord_ = coord.clone()
        q_coef = F.grid_sample(
            coef, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_freq = F.grid_sample(
            freq, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        bs, q = coord.shape[:2]
        rel_cell = cell.clone()
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.cat((torch.cos(np.pi*q_freq[:,:,0,:]), torch.sin(np.pi*q_freq[:,:,1,:])), dim=-1)

        inp = torch.mul(q_coef, q_freq)                                 
        inp = torch.cat((inp,rel_cell),dim=-1)
        pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
        pred = (self.clipped(pred)+1)/2
        return pred 

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
