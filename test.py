import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

import datasets
import models
import utils
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torchvision.utils import save_image

from utils import createDirectory,foldername,filename




def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return val_loader


def batched_predict(model, inp, coord, cell, bsize):#30000
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]#bs*q
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, eval_type=None, eval_bsize=None, 
              window_size=0, lowbit = 4, highbit = 16, save= 0, save_path = '',verbose=False):
    model.eval()
    metric_fn = utils.calc_psnr
    val_res = utils.Averager()
    val_ssim = utils.Averager()
    basis = 2**(highbit-lowbit)/((2**highbit) -1)
    gtdepth =(2**highbit) -1
    pbar = tqdm(loader, leave=False, desc='val')
    i=0
    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = (batch['inp']) 
            if window_size is not 0:
                
                _, _, h_old, w_old = inp.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
                inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

                coord = utils.make_coord((h_old+h_pad,w_old+w_pad)).unsqueeze(0).cuda()
                cell = torch.ones_like(coord[:,:,0]).unsqueeze(-1).cuda()
                cell = cell*basis
            else:
                h_pad = 0
                w_pad = 0

                coord = batch['coord']
                cell = batch['cell']
                
            gt = batch['gt']
            h,w = inp.shape[-2:]
            pred = model(inp, coord, cell).view(-1,h, w, 3).permute(0,3, 1, 2)
            hdimage = pred*basis+inp
            hdimage = hdimage.clamp(0,1)
            if window_size is not 0:
                gt = gt.view(-1,h_old,w_old,3).permute(0,3,1,2)
                inp = inp[:,:,:h_old,:w_old]
                hdimage=hdimage[:,:,:h_old,:w_old]
            else:
                gt = gt.view(-1,h,w,3).permute(0,3,1,2)
            gt_ = gt*basis+inp
            
            val_ssim.add(SSIM(hdimage.squeeze().permute(1,2,0).cpu().numpy()*gtdepth,\
                              gt_.squeeze().permute(1,2,0).cpu().numpy()*gtdepth,\
                              channel_axis=0,data_range=gtdepth,multichannel=True),\
                               inp.shape[0])
            val_res.add(PSNR(hdimage.squeeze().cpu().numpy(),gt_.squeeze().cpu().numpy()),inp.shape[0])
            if save is 1:
                inp = inp.squeeze()
                hdimage = hdimage.squeeze()
                gt_ = gt_.squeeze()
                save_image(hdimage ,save_path +'/'+str(i)+'_'+str(lowbit)+'_'+str(highbit)+'_BDE_ABCD'+'.png')
                save_image(gt_ ,save_path +'/'+str(i)+'_'+str(highbit)+'_GT'+'.png')
                save_image(inp ,save_path +'/'+str(i)+'_'+str(lowbit)+'_input'+'.png')
            i = i+1
            if verbose:
                pbar.set_description('PSNR {:.4f}'.format(val_res.item()),'SSIM {:.4f}'.format(val_ssim.item()))               
    return val_res.item(), val_ssim.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--LBD', default='4')
    parser.add_argument('--HBD', default='16')
    parser.add_argument('--save', default='0')
    parser.add_argument('--foldertag', default='')
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    q =int(args.LBD)
    N =int(args.HBD)

    windowsize = int(args.window )
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'inpdepth': q,'gtdepth':N})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)
    #make_data_loader(config.get('val_dataset'), tag='val')

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
    
    save_ = int(args.save)
    tag = args.foldertag
    save_path = ''
    
    if save_ is 1:
        folder_name = foldername(args.config)##ex)kodak
        model_name = foldername(args.model)##ex)train_~~
        save_path='./result/'+model_name+'/'+str(q)+'-'+str(N)+'/'+folder_name + tag
        createDirectory(save_path)
        
    res,resssim = eval_psnr(loader, model,
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        lowbit = q,
        highbit = N,
        window_size= windowsize,
        save=save_,
        save_path = save_path,
        verbose=True)
    print('PSNR: {:.4f}'.format(res),'SSIM: {:.4f}'.format(resssim))
