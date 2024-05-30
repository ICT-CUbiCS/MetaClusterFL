import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = make_layers(cfg['D'], batch_norm=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def get_param_vector(self):
        param_vector = []
        for name, param in self.named_parameters():
            if 'classifier' in name:
                param_vector.append(param.data.view(-1))
        m_param_vector = torch.cat(param_vector)
        m_param_vector = m_param_vector.cpu()
        return m_param_vector


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU()]
        input_channel = l

    return nn.Sequential(*layers)

if __name__ == "__main__":
    model = Model()
    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape)