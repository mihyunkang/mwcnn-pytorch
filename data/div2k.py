import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio


import torch
import torch.utils.data as data
import h5py
import torchvision

from skimage.external.tifffile import imsave as imsave_tiff
from skimage.external.tifffile import imread as imread_tiff


class DIV2K(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        #mat = h5py.File('../MWCNN/imdb_gray.mat')
        self.dir_A = '../../data/sidd1/HIGH/'
        self.dir_B = '../../data/sidd1/LOW/'
        self.args.ext = 'tiff'
        self.A_paths = sorted(os.listdir(self.dir_A))
        self.B_paths = sorted(os.listdir(self.dir_B))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths) 
        if self.A_size != self.B_size:
            print("unpaired : length of file_A and length of file_B are different")
            #self.hr_data = mat['images']['labels'][:,:,:,:]
            #self.num = self.hr_data.shape[0]
            #print(self.hr_data.shape)

        if self.split == 'test':
            self._set_filesystem(args.dir_data)

        #self.images_hr = self._scan()



    #def _scan(self):
    #    raise NotImplementedError
    #
    #def _set_filesystem(self, dir_data):
    #    raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        #hr, filename = self._load_file(idx)
        if self.train:
            A_filename = self.A_paths[idx % self.A_size]
            B_filename = self.B_paths[idx % self.A_size]
            A_path = os.path.join(self.dir_A, A_filename)
            B_path = os.path.join(self.dir_B, B_filename)

            A_img = imread_tiff(A_path)
            B_img = imread_tiff(B_path)

            B_img, A_img = self.get_patch(B_img, A_img)

            if A_img.ndim == 2:
                A_tensor = torchvision.transforms.functional.to_tensor(A_img)
                B_tensor = torchvision.transforms.functional.to_tensor(B_img)

            else:
                B_tensor, A_tensor = common.np2Tensor([B_img, A_img], self.args.rgb_range)

            ##normalize 
            return B_tensor, A_tensor, A_filename
        else:
            A_filename = self.A_paths[idx % self.A_size]
            B_filename = self.B_paths[idx % self.A_size]
            A_path = os.path.join(self.dir_A, A_filename)
            B_path = os.path.join(self.dir_B, B_filename)

            A_img = imread_tiff(A_path)
            B_img = imread_tiff(B_path)

            B_img, A_img = self.get_patch(B_img, A_img)

            if A_img.ndim == 2:
                A_tensor = torchvision.transforms.functional.to_tensor(A_img)
                B_tensor = torchvision.transforms.functional.to_tensor(B_img)
               
            else:
                B_tensor, A_tensor = common.np2Tensor([B_img, A_img], self.args.rgb_range)
            #print("b tensor size {}".format(B_tensor.shape))
            #print(2)
            ##normalize 
            return B_tensor, A_tensor, A_filename


    def __len__(self):
        return self.A_size

    def _get_index(self, idx):
        return idx


    def get_patch(self, lr, hr):
        if self.args.patch_size > 1000:
            return lr, hr
        if self.train:
            scale = self.scale[0]
            lr, hr = common.get_patch(
                    lr, hr, patch_size= self.args.patch_size, scale=1
                )
            lr, hr= common.augment(lr, hr)
            #print(self.args.patch_size)
            #print("11111111111111111111 {}".format(lr.shape))
            return lr, hr
        else:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = common.get_patch(
                    lr, hr, patch_size= self.args.patch_size, scale=1
                )
            lr, hr= common.augment(lr, hr)
            return lr, hr
            # lr = common.add_noise(lr, self.args.noise)





    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

