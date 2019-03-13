import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, TIFFDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
from PIL import Image, ImageTk
from math import ceil, floor
import struct
import skimage
from osgeo import gdal
from model_siod.model import VGGRegCls
from tqdm import tqdm, trange
from time import time
from viewer.patch_loader import TIFFPatchLoader


class TIFFPatchDataset(Dataset):
    """TIFF Patch dataset."""

    def __init__(self, tiff, patches, transform=None):
        self.tiff = tiff
        self.patches = patches
        self.transform = transform
        self.band_num = tiff.RasterCount

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        if self.transform:
            img = self.transform(img)

        return img

    def load_image(self, idx):
        xoff, yoff, width, height = self.patches[idx]
        xoff, yoff, width, height = int(xoff), int(yoff), int(width), int(height)

        img_list = []

        for b in range(self.band_num):
            band = self.tiff.GetRasterBand(b + 1)
            scanline = band.ReadRaster(xoff, yoff,
                                       width, height,
                                       width, height, gdal.GDT_Float32)
            tuple_of_floats = struct.unpack('f' * width * height, scanline)

            img = np.array(list(tuple_of_floats)).reshape(height, width)
            img_list.append(np.expand_dims(img, 2))

        if len(img_list) == 1:
            res_img = skimage.color.gray2rgb(img)
        else:
            res_img = np.concatenate(img_list, axis=2)

        save_img = Image.fromarray(res_img.astype(np.uint8))
        save_img.save('test_patch_1.bmp')

        res_img = res_img.astype(np.float32) / 255.

        save_img = Image.fromarray(res_img.astype(np.uint8))
        save_img.save('test_patch_2.bmp')

        return res_img


class TIFFPreciseDataset(Dataset):
    def __init__(self, tiff, df, transform=None):
        self.tiff = tiff
        self.df = df
        self.transform = transform
        self.band_num = tiff.RasterCount

    def __getitem__(self, index):
        path, xoff, yoff, width, height, x1, y1, x2, y2, _ = self.df.values[index]

        buffer_y, buffer_x = width, height

        x0 = x1 + floor((x2 - x1) / 2)
        y0 = y1 + floor((y2 - y1) / 2)

        xoff += x0 - width // 2
        yoff += y0 - height // 2

        img_list = []

        for b in range(self.band_num):
            band = self.tiff.GetRasterBand(b + 1)
            scanline = band.ReadRaster(xoff, yoff, width, height, buffer_x, buffer_y,
                                       gdal.GDT_Float32)
            tuple_of_floats = struct.unpack('f' * buffer_x * buffer_y, scanline)

            img = np.array(list(tuple_of_floats)).reshape(buffer_y, buffer_x)
            img_list.append(np.expand_dims(img, 2))

        if len(img_list) == 1:
            res_img = skimage.color.gray2rgb(img)
        else:
            res_img = np.concatenate(img_list, axis=2)

        res_img = np.uint8(res_img)

        if self.transform is not None:
            res_img = self.transform(res_img)

        return res_img

    def __len__(self):
        return self.df.shape[0]


class RetinaNetPredictor:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([Normalizer(), Resizer()])
        self.unnormalize = UnNormalizer()
        self.overlap_threshold = 0.6
        self.score_threshold = 0.5
        self.distance_threshold = 1.

        self.bboxes = None

    def model_load(self, filepath):
        print('Loading model...')
        try:
            self.model = torch.load(filepath).cuda()
            self.model.eval()
            print('Model loaded successfully!')
            return True
        except:
            return False

    def get_bboxes(self, img):
        bbox_list = []

        # print(img.shape)

        # save_img = Image.fromarray(img.astype(np.uint8))
        # save_img.save('test1.bmp')

        img = img.astype(np.float32) / 255.0
        sample = {
            'img': img,
            'annot': np.ones((1, 4))
        }
        sample = self.transform(sample)
        img = sample['img']
        scale = sample['scale']
        # print(img.shape)

        # save_img = Image.fromarray(img.cpu().data.numpy().astype(np.uint8))
        # save_img.save('test2.bmp')

        img = img.permute(2, 0, 1)
        # print(img.shape)

        # save_img = Image.fromarray(img.cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8))
        # save_img.save('test3.bmp')
        time1 = time()
        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img.cuda().float().unsqueeze(0))[0]
            idxs = np.where(scores > self.score_threshold)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :] / scale
                # x1 = int(bbox[0])
                # y1 = int(bbox[1])
                # x2 = int(bbox[2])
                # y2 = int(bbox[3])
                bbox = bbox.cpu().data.numpy()
                bbox_list.append(bbox)
        time2 = time()
        # print('Detection time: {} sec.'.format(time2 - time1))

        bbox_list = np.array(bbox_list)
        self.bboxes = bbox_list

        # bbox_list = non_max_suppression(bbox_list, self.overlap_threshold)
        return bbox_list

    def apply_distance_filter(self, path, xoff, yoff, width, height, model_path):
        if self.bboxes is not None:
            print('Starting distance filter...')
            # get df from bboxes

            # pass bboxes to dataset, dataloader
            df = pd.DataFrame(columns=['path', 'xoff', 'yoff', 'width',
                                       'height', 'x1', 'y1', 'x2', 'y2', 'class'])
            df['x1'] = self.bboxes[:, 0]
            df['y1'] = self.bboxes[:, 1]
            df['x2'] = self.bboxes[:, 2]
            df['y2'] = self.bboxes[:, 3]
            df['path'] = path
            df['xoff'] = xoff
            df['yoff'] = yoff
            df['width'] = width
            df['height'] = height

            transform = transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
            tiff = gdal.Open(path, gdal.GA_ReadOnly)
            dataset = TIFFPreciseDataset(tiff=tiff, df=df, transform=transform)
            batch_size = 1
            dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)

            model = torch.load(model_path).cuda()
            model.eval()

            dataiter = iter(dataloader)
            distances = []
            scores = []

            # get predictions
            with torch.no_grad():
                for inputs in tqdm(dataiter):
                    # print(inputs.shape)
                    outputs_l, outputs_c = model(Variable(inputs).cuda())
                    distances.append(outputs_l.squeeze().cpu().data.numpy())
                    scores.append(outputs_c.squeeze().cpu().data.numpy())
                    # print(outputs_l, outputs_c)

            # delete bboxes with scores lower than threshold
            distances = np.array(distances)
            scores = np.array(scores)

            # return bboxes
            print('Distance filter applied!')

            self.bboxes = self.bboxes[scores[:, 1] > scores[:, 0]]
            # self.bboxes = non_max_suppression(self.bboxes, self.overlap_threshold)
            return self.bboxes
        else:
            print('Bounding boxes list is empty!')

    def get_bboxes_for_area(self, tiff, a_x1, a_y1, a_x2, a_y2, f_w=800, f_h=800, overlap=40):
        pl = TIFFPatchLoader(tiff=tiff)

        a_w = a_x2 - a_x1
        a_h = a_y2 - a_y1

        w_N = ceil((a_w - f_w) / (f_w - overlap)) + 1
        h_N = ceil((a_h - f_h) / (f_h - overlap)) + 1

        transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        time1 = time()
        time_pl = 0.
        time_predictor = 0.

        patches = []

        for w_n in range(w_N):
            for h_n in range(h_N):
                x1 = a_x1 + w_n * (f_w - overlap)
                y1 = a_y1 + h_n * (f_h - overlap)

                # x2 = x1 + f_w
                # y2 = y1 + f_h

                patches.append([
                    x1, y1, f_w, f_h
                ])

                # time_pl1 = time()
                # img = pl.get_patch(x1=x1, y1=y1, x2=x2, y2=y2)
                # time_pl += time() - time_pl1

                # time_predictor1 = time()
                # bbox_list = self.get_bboxes(img=img)
                # time_predictor += time() - time_predictor1

        patches = np.array(patches)
        # print('Patches shape: ', patches.shape)
        batch_size = 16

        dataset = TIFFPatchDataset(tiff=tiff, patches=patches, transform=transform)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        dataiter = iter(dataloader)

        bbox_list = []
        time_pl1 = time()

        with torch.no_grad():
            current_iter = 1
            current_patch = 0
            iterations = ceil(len(dataset) / batch_size)

            for inputs in dataiter:
                # print('Inputs:', inputs.shape)
                time_pl += time() - time_pl1

                # save_img = Image.fromarray(inputs[0].cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8))
                # save_img.save('test_patch.bmp')

                time_predictor1 = time()
                result = self.model(inputs.cuda().float())
                time_predictor += time() - time_predictor1

                for img_i in range(len(result)):
                    if result[img_i] is not None:
                        scores, classification, transformed_anchors = result[img_i]
                        # print('Scores:', scores, scores.shape)
                        # print('Classification:', classification.shape)
                        # print('transformed_anchors:', transformed_anchors.shape)

                        idxs = np.where(scores > self.score_threshold)
                        # print('Idxs', len(idxs), idxs)

                        for j in idxs[0]:
                            bbox = transformed_anchors[j, :]
                            bbox = bbox.cpu().data.numpy()
                            bbox = list(patches[current_patch]) + list(bbox)
                            bbox_list.append(bbox)

                    current_patch += 1

                time_pl1 = time()
                elapsed = time() - time1
                print('[{}/{}] Elapsed: {} sec, estimated: {} sec.'.format(
                    current_iter, iterations, elapsed,
                    (iterations - current_iter) * (elapsed / current_iter)
                ))

                current_iter += 1

        print('Patches:', current_patch)
        print("Total time spent: \n * on image loading: {},\n * on prediction: {}.".format(time_pl, time_predictor))
        bbox_list = np.array(bbox_list)
        # print('Bbox list shape: ', bbox_list.shape)
        return bbox_list


def non_max_suppression(bbox_list, overlap_threshold=0.6):
    pick = []

    x1 = bbox_list[:, 0]
    y1 = bbox_list[:, 1]
    x2 = bbox_list[:, 2]
    y2 = bbox_list[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1  # last - index of last index (% in sorted list
        i = idxs[last]  # i - the last bbox index in sorted list
        pick.append(i)  # add the last bbox index in list of picked bbox indices
        suppress = [last]  # suppress - list of suppressed (deleted) indices of indices

        for pos in range(last):
            j = idxs[pos]  # index of index (from first to next-to-last)

            xx1 = max(x1[j], x1[i])  # find biggest x1 and y1
            yy1 = max(y1[j], y1[i])

            xx2 = min(x2[j], x2[j])  # find smallest x2 and y2
            yy2 = min(y2[j], y2[i])

            w = max(0, xx2 - xx1 + 1)  # find width an height of computed "perfect" bbox
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]  # how much "perfect" bbox overlap with current bbox

            if overlap > overlap_threshold:
                suppress.append(pos)  # if too much overlap, suppress current bbox

        idxs = np.delete(idxs, suppress)

    return bbox_list[pick]
