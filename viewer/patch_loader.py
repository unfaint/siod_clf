from osgeo import gdal
import struct
import numpy as np
import skimage
from PIL import Image, ImageTk
from time import time


class TIFFPatchLoader:
    def __init__(self, tiff):
        self.tiff = tiff
        self.band_num = tiff.RasterCount

    def get_patch(self, x1, y1, x2, y2, buffer_x=None, buffer_y=None):

        # print(x1, y1, x2, y2)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        img_list = []

        assert self.tiff.RasterXSize >= x2 > x1 >= 0
        assert self.tiff.RasterYSize >= y2 > y1 >= 0

        width = x2 - x1
        height = y2 - y1

        if buffer_x is None:
            buffer_x = width

        if buffer_y is None:
            buffer_y = height

        time1 = time()
        for b in range(self.band_num):
            band = self.tiff.GetRasterBand(b + 1)
            # print(x1, y1, width, height, buffer_x, buffer_y)
            scanline = band.ReadRaster(x1, y1, width, height, buffer_x, buffer_y, gdal.GDT_Float32)
            tuple_of_floats = struct.unpack('f' * buffer_x * buffer_y, scanline)

            img = np.array(list(tuple_of_floats)).reshape(buffer_y, buffer_x)
            img_list.append(np.expand_dims(img, 2))

        if len(img_list) == 1:
            res_img = skimage.color.gray2rgb(img)
        else:
            res_img = np.concatenate(img_list, axis=2)

        res_img = np.uint8(res_img)

        # print('Patch loaded in {} sec.'.format(time() - time1))

        return res_img

    def get_overview(self, buffer_x, buffer_y):
        x1 = y1 = 0
        x2 = self.tiff.RasterXSize
        y2 = self.tiff.RasterYSize

        ratio = x2 / y2
        b_ratio = buffer_x / buffer_y

        if b_ratio > ratio:  # height matters
            scale = buffer_y / y2
            # print('height')
        else:  # width matters
            scale = buffer_x / x2
            # print('width')

        buffer_x = int(x2 * scale)
        buffer_y = int(y2 * scale)

        # print(x1, y1, x2, y2, buffer_x, buffer_y)

        return self.get_patch(x1, y1, x2, y2, buffer_x, buffer_y), buffer_x, buffer_y, scale
