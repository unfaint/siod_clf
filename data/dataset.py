import numpy as np
import torch
import torch.utils.data
from PIL import Image
import os

class COWCDataset(torch.utils.data.Dataset):
    """
    Loading images from COWC-N dataset, see description at
    https://github.com/LLNL/cowc/tree/master/COWC-M
    """

    def __init__(self, file_list, transform= None, mini= True):
        self.file_list = file_list
        self.labels = np.full((len(file_list),3), 0.0)
        self.transform = transform
        self.mini = mini

    def __getitem__(self, index):
        file = self.file_list[index]

        img = np.array(Image.open(file))

        label = self.labels[index]
        label[2] = 1 if file.split('.')[0][-3:] == 'car' else 0

        # as long as COWC-M dataset has only class labels,
        # contained in file names, we randomly generate
        # regression labels

        img_H = 256 if self.mini else 256
        crop_h = 56 if self.mini else 94

        H = int(img_H / 2)
        h = int(crop_h / 2)

        x = label[0] = np.random.randint(-10, 10)
        y = label[1] = np.random.randint(-10, 10)

        label[0] = label[0] / 10
        label[1] = label[1] / 10

        shift_x = H - h - x
        shift_y = H - h - y

        img = img[shift_y:shift_y + crop_h + 1, shift_x:shift_x + crop_h + 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.file_list)


class OIRDSDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, file_list, labels, transform=None):
        self.data_path = data_path
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        file = self.file_list[index]
        img = Image.open(os.path.join(self.data_path, file))
        labels = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.LongTensor(labels)

    def __len__(self):
        return len(self.file_list)