import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(120, 84),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(84, 62)
        )
        # self.fc1 = nn.Linear(400, 120)
        # self.activation3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.classifier = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.activation3(x)
        # x = self.fc2(x)
        x = self.classifier(x)
        return x

    def get_param_vector(self):
        param_vector = []
        for name, param in self.named_parameters():
            if 'classifier' in name:
                param_vector.append(param.data.view(-1))
        m_param_vector = torch.cat(param_vector)
        m_param_vector = m_param_vector.cpu()
        return m_param_vector


if __name__ == "__main__":
    model = Model()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape)
    param_vector = model.get_param_vector()
    print(param_vector)