import random
from torch.utils.data import DataLoader, random_split
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

from dataset import LoLDataset

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
        transformed = self.transform_image(image)
        return transformed

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
        if random.random() > 0.7:
            base = cv2.flip(base, 1)
        
        new_size = random.randint(self.min_size, self.max_size)
        scaled_image = cv2.resize(image,(new_size, new_size))
        
        pad = (self.output_size - new_size) // 2
        base[pad:pad + new_size, pad:pad + new_size, :] = scaled_image
            
        return base


def get_train_dataloader(data_root, transform, train_val_split=0.8, batch_size=4, shuffle=True):
    
    dataset = LoLDataset(data_root, test=False, transform=transform)
    
    # train, validation split
    train_size = int(len(dataset) * train_val_split)
    train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
    
    # assign dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return (train_dataloader, valid_dataloader)


if __name__ == "__main__":
    data_root = '~/Documents/falconai/Assets/Assets/FalconAIChallenge/champions/'
    
    game = cv2.imread(game_img_path, cv2.IMREAD_COLOR)
    background = game[90:-300, 80:-90, :]

    transform_train = ChampionTransform(background, output_size=145, object_size=(30, 40))

    train_dataloader, valid_dataloader = get_train_dataloader(data_root, transform_train)