import numpy as np
import torch
import torch.utils.data
from PIL import Image
import os
from osgeo import gdal
from math import ceil
import struct

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

        print(len(img.shape))
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


class CITDataset(torch.utils.data.Dataset):

    def __init__(self, band, paths, transform= None, n= 0, size= (318, 318), jitter= 0):
        # parameters
        self.width, self.height = size
        self.jitter = jitter

        self.band = band

        self.pos = self.get_coords(paths['pos'])
        print('Positive examples: ', self.pos.shape[1])
        self.neg = self.get_coords(paths['neg'])
        print('Negative examples: ', self.neg.shape[1])

        if n > 0:
            self.coords = np.concatenate((
                self.pos[:, np.random.choice(self.pos.shape[1], n)],
                self.neg[:, np.random.choice(self.neg.shape[1], n)]), axis= 1)
            self.labels = np.concatenate((np.full(n, 1), np.full(n, 0)))
        else:
            self.coords = np.concatenate((
                self.pos,
                self.neg), axis=1)
            self.labels = np.concatenate((np.full(self.pos.shape[1], 1),
                                          np.full(self.neg.shape[1], 0)))

        self.transform = transform

    def __getitem__(self, index):
        labels = self.labels[index]

        coords = self.coords[:, index]
        width = self.width
        height = self.height

        xoff = int(coords[1] - ceil(width / 2))
        yoff = int(coords[0] - ceil(height / 2))

        if self.jitter > 0:
            xoff += np.random.randint(-self.jitter, self.jitter)
            yoff += np.random.randint(-self.jitter, self.jitter)

        qx, qy = xoff, yoff
        sx, sy = xoff + width, yoff + height

        if qx < 0:
            width = width - abs(qx) - 1

        if qy < 0:
            height = height - abs(qy) - 1

        if sx > self.band.XSize:
            width = width - (sx - self.band.XSize) - 1

        if sy > self.band.YSize:
            height = height - (sy - self.band.YSize) - 1

        scanline = self.band.ReadRaster(xoff, yoff,
                                        width, height,
                                        width, height, gdal.GDT_Float32)

        tuple_of_floats = struct.unpack('f' * width * height, scanline)
        img = np.array(list(tuple_of_floats)).reshape(height, width)
        # img = np.expand_dims(img, 2)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.LongTensor([labels])

    def __len__(self):
        return self.coords.shape[1]

    def get_coords(self, path):
        img = np.array(Image.open(path))
        mask = img[:, :, 1] == 255
        grid = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        coords = grid[:, mask]

        return coords