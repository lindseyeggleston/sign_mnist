from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __inti__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels for RGB
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # reshapes Tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BiLSTM(nn.Module):
    '''Simple BiLSTM'''
    def __init__(self, n_layers, r_hidden, d_hidden, dropout=.2):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(hidden_size=r_hidden, num_layers=n_layers,
                           dropout=dropout, bidirectional=True)
        self.pool = nn.MaxPool2d()  # TODO: add shape
        self.fc1 = nn.Linear()  # TODO: add shape
        self.fc2 = nn.Linear()  # TODO: add shape
        self.fc3 = nn.Linear()  # TODO: add shape

    def forward(self, x):
        x = self.rnn(x)
        x = self.pool(F.relu(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
