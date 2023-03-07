import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
import cv2

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)


            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))
        
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x
        
@register('image-folder-cv2-16bit')
class ImageFolder16bit(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    #print(file)
                    with open(bin_file, 'wb') as f:
                        print(file)
                        pickle.dump(cv2.cvtColor(cv2.imread(file,-1),cv2.COLOR_BGR2RGB),f)
                    print('dump', bin_file)
                self.files.append(bin_file)
                self.bin_root = bin_root
            elif cache == 'in_memory':
                print(file)
                self.files.append(transforms.ToTensor()(
                    cv2.cvtColor(cv2.imread(file,-1),cv2.COLOR_BGR2RGB)/65535.).float())
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(cv2.cvtColor(cv2.imread(x,-1),cv2.COLOR_BGR2RGB)/65535.).float()

        elif self.cache == 'bin':
            #os.system('rm '+self.bin_root+'.pkl')
            with open(x, 'rb') as f:
                x = pickle.load(f)
            return transforms.ToTensor()(x/65535.).float()

        elif self.cache == 'in_memory':
            return x

        
@register('image-folder-cv2-8bit')
class ImageFolder8bit(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    #print(file)
                    with open(bin_file, 'wb') as f:
                        print(file)
                        pickle.dump(cv2.cvtColor(cv2.imread(file,-1),cv2.COLOR_BGR2RGB),f)
                    print('dump', bin_file)
                self.files.append(bin_file)
                self.bin_root = bin_root
            elif cache == 'in_memory':
                print(file)
                self.files.append(transforms.ToTensor()(
                    cv2.cvtColor(cv2.imread(file,-1),cv2.COLOR_BGR2RGB)/255.).float())
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(cv2.cvtColor(cv2.imread(x,-1),cv2.COLOR_BGR2RGB)/255.).float()

        elif self.cache == 'bin':
            #os.system('rm '+self.bin_root+'.pkl')
            with open(x, 'rb') as f:
                x = pickle.load(f)
            return transforms.ToTensor()(x/255.).float()

        elif self.cache == 'in_memory':
            return x
