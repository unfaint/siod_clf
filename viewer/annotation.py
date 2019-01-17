from typing import List, Any, Union

import pandas as pd
import numpy as np


class BBoxList:
    def __init__(self, img_path):
        self.img_path = img_path
        self.frame = None
        self.columns = ['img_path', 'xoff', 'yoff', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 'class_name']
        self.bb_df = pd.DataFrame(columns=self.columns)

    def set_frame(self, xoff, yoff, width, height):
        self.frame = [
            xoff, yoff, width, height
        ]
        print(self.frame)

    def add_bbox(self, x1, y1, x2, y2, class_name):
        assert self.frame is not None
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, self.frame[2])
        y2 = min(y2, self.frame[3])

        bbox_entry = [self.img_path] + \
                     self.frame + \
                     [x1, y1, x2, y2, class_name]  # type: Union[List[Any], Any]
        print(bbox_entry)
        add_df = pd.DataFrame(columns=self.columns, data=np.array(bbox_entry).reshape(1, len(bbox_entry)))
        self.bb_df = pd.concat([self.bb_df, add_df], axis=0, ignore_index=True)


class BFrame:
    def __init__(self, xoff, yoff, width, height):
        self.xoff = xoff
        self.yoff = yoff
        self.width = width
        self.height = height

        self.bbox_list = []

    def add_bbox(self, bbox):
        self.bbox_list.append(bbox)

    def get_bbox_table(self):
        bbox_table = []
        for bb in self.bbox_list:
            bbox_table.append([
                bb.x1,
                bb.y1,
                bb.x2,
                bb.y2,
                bb.class_name
            ])

        return np.array(bbox_table)


class BBox:
    def __init__(self, x1, y1, x2, y2, class_name):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.class_name = class_name
