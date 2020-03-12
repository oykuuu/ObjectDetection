""" Custom dataloaders for Deep Learning approach.
"""
import random
from torch.utils.data import DataLoader
from PIL import Image
import torchvision

from data.dataset import LoLTrainDataset, LoLValidDataset


class ChampionTransform(object):
    """ Resize and pad champion images to increase training size.
    """

    def __init__(self, background, output_size=45, object_size=(30, 40)):
        """ Establishes background image and checks if output size
        is an integer.

        Parameters
        ----------
        background
            np.array, background reference image
        output_size
            int or tuple, size of the image after transformation
        object_size
            tuple, lower and upper bound on champion size
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
        base = self.background[
            y : y + self.output_size, x : x + self.output_size, :
        ].copy()

        new_size = random.randint(self.min_size, self.max_size)
        scaled_image = image.resize((new_size, new_size))

        pad = (self.output_size - new_size) // 2
        base[pad : pad + new_size, pad : pad + new_size, :] = scaled_image
        box = [pad, pad, pad + new_size, pad + new_size]

        return base, box


def get_train_dataloader(
        data_root,
        background,
        output_size=65,
        object_size=(30, 40),
        batch_size=4,
        shuffle=True,
):
    """
    Prepares the training dataloader and outputs the LabelEncoder used in
    mapping champion names into integers.

    Parameters
    ----------
    data_root
        string, path of the folder containing training images (champions)
    background
        np.array, image selected as background
    output_size
        int or tuple, size of the image after transformation
    object_size
        tuple, lower and upper bound on champion size
    batch_size
        int, size of the batch
    shuffle
        boolean, set to True if data is to be shuffled during training

    Returns
    -------
    train_dataloader
        DataLoader, dataloader containing transformed images for training
    trained_encoder
        LabelEncoder, encoder used to turn champion names into integers
    """

    transform_train = ChampionTransform(
        background, output_size=output_size, object_size=object_size
    )

    train_dataset = LoLTrainDataset(data_root, transform=transform_train)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    trained_encoder = train_dataset.get_encoder()

    return train_dataloader, trained_encoder


def get_valid_dataloader(
    encoder, data_path="../screenshot.png", team="left", batch_size=4, shuffle=True
):
    """
    Prepares the training dataloader and outputs the LabelEncoder used in
    mapping champion names into integers.

    Parameters
    ----------
    encoder
        LabelEncoder, encoder used to turn champion names into integers
    data_path
        string, filepath of the image to be used in validation (screenshot)
    team
        string, can take values 'left' or 'right' to refer to either team
    batch_size
        int, size of the batch
    shuffle
        boolean, set to True if data is to be shuffled during training

    Returns
    -------
    valid_dataloader
        DataLoader, dataloader containing transformed images of one
        team to be used for validation
    true_labels,
        list, list of the true champion names of the team
    """

    valid_dataset = LoLValidDataset(encoder, data_path=data_path, team=team)

    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle
    )

    true_labels = valid_dataset.get_true_labels()

    return valid_dataloader, true_labels


if __name__ == "__main__":

    data_root = "~/Documents/champions/"
    game_img_path = (
        "~/Documents/screenshot.png"
    )

    game = Image.open(game_img_path).convert("RGB")
    background = game[90:-300, 80:-90, :]

    train_dataloader = get_train_dataloader(data_root, background)
