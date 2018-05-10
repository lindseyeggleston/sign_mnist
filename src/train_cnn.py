import matplotlib.pyplot as plt
import torch
import torchvision
from preprocess import preprocess
from torchvision import transforms


if __name__ == '__main__':
    trainset = preprocess(filepath)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
