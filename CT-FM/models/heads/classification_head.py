import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes, pre_func=None):
        super(ClassificationHead, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.pre_func = pre_func

    def forward(self, x):
        if self.pre_func is not None:
            x = self.pre_func(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        return x
