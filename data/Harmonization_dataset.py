
from lib2to3.pytree import convert
import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class HDay2nightDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, dataset_root, split):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, dataset_root)
        self.image_paths = []
        self.isTrain = (split == 'train')
        if self.isTrain==True:
            print('loading training file: ')
            self.trainfile =dataset_root+'Hday2night_train.txt' #修改点1，替换HCOCO_train.txt
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(dataset_root, 'composite_noisy25_images', line.rstrip())) #修改点2，增加composite_images，如果是带噪声的训练，将这里修改为composite_noisy25_images
        elif self.isTrain==False:
            print('loading test file')
            self.trainfile = dataset_root+'Hday2night_test.txt' #修改点3， 替换HCOCO_test
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(dataset_root, 'composite_noisy25_images', line.rstrip())) #修改点4，如果是带噪声的训练，将这里修改为composite_noisy25_images
        self.transform = get_transform()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        name_parts=path.split('_')
        mask_path = self.image_paths[index].replace('composite_noisy25_images','masks') # #修改点5，如果是带噪声的训练，将这里修改为composite_noisy25_images
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_noisy25_images','real_images') # #修改点6，如果是带噪声的训练，将这里修改为composite_noisy25_images
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        comp = tf.resize(comp, [256, 256])
        mask = tf.resize(mask, [256, 256])
        #mask = tf.resize(mask, [224, 224]) #对MAE训练，需要将这里修改为224,224
        real = tf.resize(real,[256,256])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        #mask = self.mask_transform(mask)
        mask = tf.to_tensor(mask)

        real = self.transform(real)
        #concate the composite and mask as the input of generator
        inputs=torch.cat([comp,mask],0)

        return {'SR': comp, 'HR': real, 'Index': index}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
