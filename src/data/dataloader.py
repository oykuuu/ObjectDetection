import random
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import torchvision

from src.data.dataset import LoLTrainDataset, LoLValidDataset
import pdb

class ChampionTransform(object):
    """ Resize and pad champion images to increase training size.
    """
    
    def __init__(self, background, output_size=45, object_size=(30, 40)):
        """ Establishes background image and checks if output size
        is an integer.
        
        Parameters
        ----------
        output_size
            int or tuple, size of the image after transformation
        object_size
            tuple, lower and upper bound on champion size
        background
            np.array, background reference image
        """
        self.output_size = output_size
        self.background = background
        self.min_size = object_size[0]
        self.max_size = object_size[1]
        assert isinstance(output_size, int)
        
    def __call__(self, image):
        """ Return reshaped and padded image.
        
        Parameters
        ----------
        image
            np.array, input image
            
        Returns
        -------
        transformed
            np.array, transformed image
        """
        transformed, box = self.transform_image(image)
        return transformed, box

    def transform_image(self, image):
        """ Randomly add background noise to image by padding and resize
        champion image.
        
        Parameters
        ----------
        image
            np.array, input image
        """
        y = random.randint(0, self.background.shape[0] - self.output_size - 1)
        x = random.randint(0, self.background.shape[1] - self.output_size - 1)
        base = self.background[y:y + self.output_size, x:x + self.output_size, :].copy()
        
        new_size = random.randint(self.min_size, self.max_size)
        scaled_image = image.resize((new_size, new_size))
        
        pad = (self.output_size - new_size) // 2
        base[pad:pad + new_size, pad:pad + new_size, :] = scaled_image
        box = [pad, pad, pad+new_size, pad+new_size]
            
        return base, box


def get_train_dataloader(data_root, background, output_size=65, object_size=(30, 40), batch_size=4, shuffle=True):
    
    transform_train = ChampionTransform(background, output_size=output_size, object_size=object_size)

    train_dataset = LoLTrainDataset(data_root, transform=transform_train)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    trained_encoder = train_dataset.get_encoder()
    
    return train_dataloader, trained_encoder

def get_valid_dataloader(encoder, data_path='../screenshot.png', team='left', batch_size=4, shuffle=True):

    valid_dataset = LoLValidDataset(encoder, data_path=data_path, team=team)

    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return valid_dataloader
    
if __name__ == "__main__":
    data_root = '~/Documents/falconai/Assets/Assets/FalconAIChallenge/champions/'
    game_img_path = '~/Documents/falconai/Assets/Assets/FalconAIChallenge/screenshot.png'

    game = Image.open(game_img_path).convert("RGB")
    background = game[90:-300, 80:-90, :]

    train_dataloader = get_train_dataloader(data_root, background)