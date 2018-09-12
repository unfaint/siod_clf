import torch
from data.dataset import COWCDataset

print(torch.cuda.is_available())
dataset = COWCDataset(file_list= [])