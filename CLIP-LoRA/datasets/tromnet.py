import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.datasets as datasets


imagenet_classes = ["injury", "no_injury", "injury_and_amputation"]
imagenet_templates = ["For the highlighted limb {} is present."]

class trom_net():

    dataset_dir = 'DATASET'

    def __init__(self, root, train_preprocess=None,test_preprocess=None):

        # self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, 'images')
        
        
        preprocess = transforms.Compose([ transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
        
        # self.test=DirectImageDataset(test_imgs,transform=preprocess)
        self.test = datasets.ImageFolder((root), transform=preprocess)
        
        self.template = imagenet_templates
        self.classnames = imagenet_classes