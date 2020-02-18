
import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pdb

class LoLTrainDataset(Dataset):
    """ LoL training dataset. """
    
    def __init__(self, data_root, transform=None) -> None:
        """
        Initializes parameters.

        Parameters
        ----------
        data_root
            string, dataset folder
        test
            bool, true when test set is used
        transform
            callable, transformation to be applied

        Returns
        -------
        None
        """
        self.data_root = data_root
        self.transform = transform
        #self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder = LabelEncoder()
        self.image_paths = []
        
        self._init_dataset()
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns item at index idx.

        Parameters
        ----------
        idx
            int, sample index number

        Returns
        -------
        image
            tensor, image at idx
        encode_label
            tensor, integer label of image (between 0 and 147)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transform_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        path, label = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image, box = self.transform(image)
        image = transform_tensor(image)
        area = (box[3] - box[1]) * (box[2] - box[0])
        encoded_label = self.encoder.transform(label)#.toarray()
        
        return (image, encoded_label)
    
    def _init_dataset(self):
        """
        Dataset initalizer. Looks into data_root folder and collects
        unique champion names for encoding.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        champions = set()

        for name in os.listdir(self.data_root):
            label = name.split('.')[0]
            champions.add(label)
            self.image_paths += [(os.path.join(self.data_root, name), [[label]])]
        
        #self.encoder = self.encoder.fit(np.array(list(champions)).reshape(-1, 1))
        self.encoder = self.encoder.fit(list(champions))

    def get_encoder(self):
        return self.encoder

class LoLValidDataset(Dataset):
    """ LoL validation dataset. """
    
    def __init__(self, encoder, data_path='../screenshot.png', team='left') -> None:
        """
        Initializes parameters.

        Parameters
        ----------
        data_root
            string, dataset folder
        test
            bool, true when test set is used
        transform
            callable, transformation to be applied

        Returns
        -------
        None
        """
        self.encoder = encoder
        self.team = team
        
        image = cv2.imread(data_path, cv2.IMREAD_COLOR)

        team_left, team_right = self.split_game_image(image, adjustment=5)
        if self.team == 'left':
            self.labels = ['Ornn', 'RekSai', 'Syndra', 'Aphelios', 'Thresh']
            self.images = self.split_into_players(team_left, num_players=5)
        elif self.team == 'right':
            self.images = self.split_into_players(team_right, num_players=5)
            self.labels = ['Jax', 'LeeSin', 'Yasuo', 'Cassiopeia', 'Blitzcrank']

        else:
            raise ValueError('Team values must be either left or right.')
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns item at index idx.

        Parameters
        ----------
        idx
            int, sample index number

        Returns
        -------
        image
            tensor, image at idx
        encode_label
            tensor, integer label of image (between 0 and 147)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        transform_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        image = self.images[idx]
        image = transform_tensor(image)
        encoded_label = self.encoder.transform([self.labels[idx]]).ravel()
        
        return (image, encoded_label)

    def split_game_image(self, game, adjustment=5):
        """
        Find the leaderboard within the screenshot and
        split it in half to isolate two teams.

        Parameters
        ----------
        game
            np.array, game screenshot
        adjustment
            int, adjustment pixels as image leans towards right

        Returns
        -------
        team_left
            np.array, left half of the screenshot
        team_right
            np.array, right half of the screenshot
        """
        mid = game.shape[1]//2 + adjustment
        width = 50
        team_left = game[850:, mid-width:mid, :].copy()
        team_right = game[850:, mid:mid+width+1, :].copy()

        return team_left, team_right

    def split_into_players(self, team, num_players=5):
        """
        Isolate players from team image.

        Parameters
        ----------
        team
            np.array, split team image
        num_players
            int, number of players

        Returns
        -------
        players
            list(np.array), images of individual players
        """
        height = team.shape[0] // num_players
        players = []
        
        for h in range(num_players):
            player = team[h * height:(h + 1) * height, :, :].copy()
            players.append(self.convert_to_pil_image(player))
        
        return players

    def convert_to_pil_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image

    def get_true_labels(self):
        return self.labels

'''
class LoLDataset_ObjectDetection(Dataset):
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

        transform_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
        if self.test:
            path = self.image_paths[idx]
            image = Image.open(path).convert("RGB")
            # TODO: pass screenshot and crop transform here
            return image

        path, label = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image, box = self.transform(image)
        image = transform_tensor(image)
        area = (box[3] - box[1]) * (box[2] - box[0])
        one_hot_label = self.encoder.transform(label).toarray()
        
        target = {}
        target['boxes'] = torch.as_tensor([box], dtype=torch.float32)
        target['labels'] = torch.as_tensor([one_hot_label], dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = [area]

        return (image, target)
    
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
'''
