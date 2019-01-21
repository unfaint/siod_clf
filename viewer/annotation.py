from typing import List, Any, Union

import pandas as pd
import numpy as np


class BBoxList:
    def __init__(self, img_path):
        self.img_path = img_path
        self.frame = None
        self.columns = ['img_path', 'xoff', 'yoff', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 'class_name']
        self.dtypes = [str, int, int]
        self.bb_df = pd.DataFrame(columns=self.columns)

    def set_frame(self, xoff, yoff, width, height):
        self.frame = [
            xoff, yoff, width, height
        ]
        print(self.frame)

    def add_bbox(self, x1, y1, x2, y2, class_name):
        assert self.frame is not None
        x1, x2 = sorted((x2, x1))
        y1, y2 = sorted((y2, y1))

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, self.frame[2])
        y2 = min(y2, self.frame[3])

        bbox_entry = [self.img_path] + \
                     self.frame + \
                     [x1, y1, x2, y2, class_name]  # type: Union[List[Any], Any]
        print(bbox_entry)
        add_df = pd.DataFrame(columns=self.columns, data=[bbox_entry])
        self.bb_df = pd.concat([self.bb_df, add_df], axis=0, ignore_index=True)

    def get_frames_in_patch(self, patch_x1, patch_y1, patch_x2, patch_y2, relative= True):
        bb_df = self.bb_df.copy()
        bb_df['xoff_x2'] = bb_df['xoff'] + bb_df['width']
        bb_df['yoff_y2'] = bb_df['yoff'] + bb_df['height']

        result_df = bb_df[
            bb_df['xoff'].apply(lambda x: x > patch_x1)
            & bb_df['yoff'].apply(lambda x: x > patch_y1)
            & bb_df['xoff_x2'].apply(lambda x: x < patch_x2)
            & bb_df['yoff_y2'].apply(lambda x: x < patch_y2)
        ]

        if relative:
            result_df['xoff'] = result_df['xoff'].apply(lambda x: x - patch_x1)
            result_df['yoff'] = result_df['yoff'].apply(lambda x: x - patch_y1)
            result_df['xoff_x2'] = result_df['xoff_x2'].apply(lambda x: x - patch_x1)
            result_df['yoff_y2'] = result_df['yoff_y2'].apply(lambda x: x - patch_y1)

        return result_df[['xoff', 'yoff', 'xoff_x2', 'yoff_y2']].values

    def save_csv(self, filepath):
        self.bb_df.to_csv(path_or_buf= filepath, sep= ',', header= None, index= None)

    def read_csv(self, filepath):
        self.bb_df = pd.read_csv(filepath_or_buffer= filepath, sep= ',', header= None,
                                 index_col= None, names= self.columns)