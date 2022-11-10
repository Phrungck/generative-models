import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, n_classes=1, h_layers=[128, 256, 512, 1024]):
        super().__init__()

        self.n_classes = n_classes
        self.h_layers = h_layers

        conv_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    self.n_classes, self.h_layers[0], kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2)
            )
        )

        for i in range(len(h_layers)-1):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(self.h_layers[i], self.h_layers[i+1],
                              kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.h_layers[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.main = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Conv2d(self.h_layers[-1], self.n_classes,
                      kernel_size=4, stride=2, padding=0, bias=True),
            nn.Sigmoid()
        )

#         self.weights_init()

    def forward(self, x):

        out = self.main(x)
        out = self.fc(out)
        return torch.flatten(out, 1)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, n_classes=1, n_features=100, h_layers=[1024, 512, 256, 128]):
        super().__init__()

        self.n_classes = n_classes
        self.h_layers = h_layers
        self.n_features = n_features

        conv_layers = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.n_features, self.h_layers[0], kernel_size=4, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(self.h_layers[0]),
                nn.ReLU()
            )
        )

        for i in range(len(self.h_layers)-1):
            conv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.h_layers[i], self.h_layers[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.h_layers[i+1]),
                    nn.ReLU()
                )
            )

        self.main = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.ConvTranspose2d(
                self.h_layers[-1], self.n_classes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

#         self.weights_init()

    def forward(self, x):

        out = self.main(x)
        return self.fc(out)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
