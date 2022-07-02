import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image

from .utils import cvtColor, preprocess_input, resize_image


class TrainDataset(data.Dataset):
    def __init__(self, input_shape, lines, random):
        self.input_shape    = input_shape
        self.lines          = lines
        self.random         = random

    def __len__(self):
        return len(self.lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        annotation_path = self.lines[index].split(';')[1].strip()
        y               = int(self.lines[index].split(';')[0])

        image = cvtColor(Image.open(annotation_path))
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        if self.rand()<.5 and self.random: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)

        image = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, y

def dataset_collate(batch):
    images  = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets


class ValDataset(data.Dataset):

    def __init__(self, input_shape, lines, random):
        super(ValDataset, self).__init__()
        self.input_shape    = input_shape
        self.lines          = lines
        self.random         = random
        self.pair_list = self.gen_pairs()

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def gen_pairs(self):
        pair_list = []
        issame_list = []

        for idx, line in enumerate(self.lines):
            annotation_path_cur = line.split(';')[1].strip()
            target_cur = int(line.split(';')[0])
            rand_another_idx = random.randint(0, len(self.lines)-1)

            annotation_path_another = self.lines[rand_another_idx].split(';')[1].strip()
            target_another = int(self.lines[rand_another_idx].split(';')[0])

            if target_cur == target_another:
                issame = True

            else:
                issame = False

            issame_list.append(issame)
            pair_list.append((annotation_path_cur, annotation_path_another, issame))

        return pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]

        image1 = cvtColor(Image.open(pair[0]))
        image2 = cvtColor(Image.open(pair[1]))
        issame = pair[2]
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        if self.rand() < .5 and self.random:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

        image1 = resize_image(image1, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        image1 = np.transpose(preprocess_input(np.array(image1, dtype='float32')), (2, 0, 1))

        image2 = resize_image(image2, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        image2 = np.transpose(preprocess_input(np.array(image2, dtype='float32')), (2, 0, 1))
        return image1, image2, issame


class TestDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(TestDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs, dtype=object)

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[1])
                path1 = os.path.join(lfw_dir, pair[0], pair[2])
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[1])
                path1 = os.path.join(lfw_dir, pair[2], pair[3])
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame)    = self.validation_images[index]
        image1, image2              = Image.open(path_1), Image.open(path_2)

        image1 = resize_image(image1, [self.image_size[1], self.image_size[0]], letterbox_image = True)
        image2 = resize_image(image2, [self.image_size[1], self.image_size[0]], letterbox_image = True)
        
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)),[2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)),[2, 0, 1])

        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)
