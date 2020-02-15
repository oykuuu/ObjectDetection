
import os
import cv2
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


class LoLDataset(Dataset):
    """ LoL dataset. """
    
    def __init__(self, data_root, test=False, transform=None) -> None:
        """
        Parameters
        ----------
        data_root
            string, dataset folder
        test
            bool, true when test set is used
        transform
            callable, transformation to be applied
        """
        self.data_root = data_root
        self.test = test
        self.transform = transform
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.image_paths = []
        
        self._init_dataset()
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.test:
            path = self.image_paths[idx]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            # TODO: pass screenshot and crop transform here
            return image

        path, label = self.image_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = self.transform(image)
        one_hot_label = self.encoder.transform(label).toarray()
        
        return (image, one_hot_label)
    
    def _init_dataset(self):
        champions = set()
        
        for name in os.listdir(self.data_root):
            label = name.split('.')[0]
            champions.add(label)
            if self.test:
                self.image_paths += [os.path.join(self.data_root, name)]
            else:
                self.image_paths += [(os.path.join(self.data_root, name), [[label]])]
        
        self.encoder = self.encoder.fit(np.array(list(champions)).reshape(-1, 1))