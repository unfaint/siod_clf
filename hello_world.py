import torch
from data.dataset import COWCDataset

print(torch.cuda.is_available())
dataset = COWCDataset(file_list=[])


def main():
    pass


if __name__ == '__main__':
    main()
