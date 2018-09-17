import numpy as np
import torch
from PIL import Image
from torch import utils


class COWCDataset(utils.data.dataset.Dataset):
    """
    Loading images from COWC-N dataset, see description at
    https://github.com/LLNL/cowc/tree/master/COWC-M
    """

    def __init__(self, file_list, transform= None):
        super(COWCDataset, self).__init__()
        self.file_list = file_list
        self.labels = np.full((len(file_list),3), 0.0)
        self.transform = transform

    def __getitem__(self, index):
        file = self.file_list[index]

        img = np.array(Image.open(Image.open(file)))

        label = self.labels[index]
        label[2] = 1 if file.split('.')[0][-3:] == 'car' else 0

        # as long as COWC-M dataset has only class labels,
        # contained in file names, we randomly generate
        # regression labels

        x = label[0] = np.random.randint(-10, 10)
        y = label[1] = np.random.randint(-10, 10)

        label[0] = label[0] / 10
        label[1] = label[1] / 10

        shift_x = 128 - 28 - x
        shift_y = 128 - 28 - y

        img = img[shift_y:shift_y + 57, shift_x:shift_x + 57]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.file_list)

