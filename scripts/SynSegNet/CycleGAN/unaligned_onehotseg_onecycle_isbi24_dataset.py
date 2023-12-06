import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import cv2
import numpy as np

class UnalignedOneHotsegOnecycleISBI24Dataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_grayscale = get_transform(self.opt, grayscale=True)
        self.transform_label = get_transform(self.opt, grayscale=False, curImgIsLabel=True)
        
        self.patch_size = self.opt.patch_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)#.convert('RGB')
        B_img = Image.open(B_path)#.convert('RGB')
        
        # apply image transformation
        #A = self.transform_A(A_img)
        seed_A = np.random.randint(2147483647)
        seed_B = np.random.randint(2147483647)

        # transform_B_fix_seed = get_transform(self.opt, grayscale=False, curImgIsHE = True) # get custom
        # self.reset_random_seed(seed_B)
        # B = transform_B_fix_seed(B_img)
        # B = self.transform_B(B_img)
       
        # transform_A_fix_seed = get_transform(self.opt, grayscale=True) # get custom

        X = []
        total_marker = 1
        contain_list = [0]#,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        for i in range(0, total_marker):
            # select only 11 markers
            if i in contain_list:
                A_tmp = A_img.crop((i * self.patch_size, 0, i * self.patch_size + self.patch_size, self.patch_size))
                self.reset_random_seed(seed_A) # for consistency
                A_tmp = self.transform_grayscale(A_tmp)
                X.append(A_tmp)
        # tmp for create label
        self.reset_random_seed(seed_A)

        #####i = 1 # get binary segmentation mask
        i = 1 # get one hot segmentation mask 
        A_tmp = A_img.crop((i * self.patch_size, 0, i * self.patch_size + self.patch_size, self.patch_size))
        Atruth = self.transform_label(A_tmp)
        Atruth = Atruth * 255

        A_tensor = X[0]

        Y = []
        total_marker = 1
        contain_list = [0]#,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        for i in range(0, total_marker):
            # select only 11 markers
            if i in contain_list:
                B_tmp = B_img.crop((i * self.patch_size, 0, i * self.patch_size + self.patch_size, self.patch_size))
                self.reset_random_seed(seed_B) # for consistency
                B_tmp = self.transform_grayscale(B_tmp)
                Y.append(B_tmp)
        # tmp for create label
        self.reset_random_seed(seed_B)

        #####i = 1 # get binary segmentation mask
        i = 1 # get one hot segmentation mask 
        B_tmp = B_img.crop((i * self.patch_size, 0, i * self.patch_size + self.patch_size, self.patch_size))
        Btruth = self.transform_label(B_tmp)
        Btruth = Btruth * 255

        B_tensor = Y[0]
        # for i in range(29):
 
        # A1_tensor = None
        # for i in range(0, 24):
        #     if A1_tensor is None:
        #         A1_tensor = X1[i]
        #     else:
        #         A1_tensor = torch.cat((A1_tensor, X1[i]), 0)

        # return {'A': A_tensor, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A1': A1_tensor, 'Atruth': Atruth}
        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path, 'Atruth': Atruth, 'Btruth': Btruth}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def reset_random_seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)

