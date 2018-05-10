import torch
import argparse
from torch import nn
from torch.nn import functional as F
from rnn import BiLSTM
from utils import is_prcnt

# input_ = torch.randn(250, 100, 50)
# rnn = BiLSTM(3, 512, 128)


def run(params):

    # Train model
    for epoch in range(params['epochs']):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BiLSTM')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='input batch size for training; default=64')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='number of recurrent layers in NN; default=3')
    parser.add_argument('--r-hidden', type=int, default=256,
                        help='number of hidden units in recurrent layer; default=256')
    parser.add_argument('--d-hidden', type=int, default=128,
                        help='number of hidden units in dense layer; default=128')
    parser.add_argument('--dropout', type=is_prcnt, default=.2,
                        help='dropout percentage for recurrent layer; default=0.2')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for training; default=50')

    args = vars(parser.parse_args())
