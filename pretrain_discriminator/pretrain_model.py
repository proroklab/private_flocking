import torch.nn as nn

class Network(nn.Module):

    def __init__(self, input_shape, output_size):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size

        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear((input_shape[1]) * (input_shape[2]) * 16, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x