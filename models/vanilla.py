from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_classes, n_features, h_layers=[1024, 512, 256]):
        super().__init__()

        # note that h_layers should be in decending order
        self.n_classes = n_classes
        self.n_features = int(n_features**2)
        self.h_layers = sorted(h_layers, reverse=True)

        linear_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.n_features, self.h_layers[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
        )

        for i in range(len(self.h_layers)-1):
            linear_layers.append(
                nn.Sequential(
                    nn.Linear(self.h_layers[i], self.h_layers[i+1]),
                )
            )

        self.main = nn.Sequential(*linear_layers)

        # final layer with sigmoid
        self.fc = nn.Sequential(
            nn.Linear(self.h_layers[-1], self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return self.fc(out)


class Generator(nn.Module):
    def __init__(self, n_classes, n_features=100, h_layers=[256, 512, 1024]):
        super().__init__()

        self.n_classes = int(n_classes**2)
        self.n_features = n_features
        self.h_layers = sorted(h_layers)

        linear_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.n_features, self.h_layers[0]),
                nn.ReLU()
            )
        )

        for i in range(len(h_layers)-1):
            linear_layers.append(
                nn.Sequential(
                    nn.Linear(self.h_layers[i], self.h_layers[i+1]),
                    nn.ReLU()
                )
            )

        self.main = nn.Sequential(*linear_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.h_layers[-1], self.n_classes),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return self.fc(out)
