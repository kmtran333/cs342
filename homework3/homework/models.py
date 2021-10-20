import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                    torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity

    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        """
        Your code here
        """
        super().__init__()
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        z = self.network(x)
        z = self.classifier(z.mean(dim=[2, 3]))
        return z


class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

            self.downsample = None

            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                    torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity

    class BlockUp(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size//2,
                                         stride=stride, output_padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=5):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l

        # # Convert linear layer to convolution
        # L.append(torch.nn.Conv2d(c, n_output_channels, kernel_size=3, padding=1, stride=2))
        # L.append(torch.nn.BatchNorm2d(n_output_channels))
        # L.append(torch.nn.ReLU())

        # Add Up-Convolutions
        up_layers = layers[::-1]
        c = 128
        for l in up_layers:
            L.append(self.BlockUp(c, l, stride=2))
            c=l
        L.append(self.BlockUp(c, 32, kernel_size=7, stride=2))  # Initial conv

        # Final convolution to desired output channels
        L.append(torch.nn.Conv2d(32, n_output_channels, kernel_size=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))
        L.append(torch.nn.ReLU())

        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        h = x.shape[2]
        w = x.shape[3]

        z = self.network(x)

        return z[:, :, :h, :w]


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
