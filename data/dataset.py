from PIL import Image
from torch import utils


class COWCDataset(utils.data.dataset.Dataset):
    def __init__(self, file_list, transform= None):
        super(COWCDataset, self).__init__()
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.file_list)