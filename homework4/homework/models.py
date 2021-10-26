import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_pool = torch.nn.MaxPool2d(kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)
    heatmap_max = max_pool(heatmap[None, None].float())

    peaks = torch.eq(heatmap, heatmap_max)
    max_peaks = peaks.sum()
    peaks = peaks * heatmap_max

    return_size = min([max_peaks, max_det])

    v, i = torch.topk(peaks.flatten(), return_size)

    output = []
    for idx in range(v.size()[0]):
        if v[idx] > min_score:
            output.append((v[idx].item(), i[idx].item() % heatmap.size()[1], i[idx].item() // heatmap.size()[1]))

    return output


class Detector(torch.nn.Module):
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

    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=3):
        """
           Your code here.
           Setup your detection network
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

        # Add Up-Convolutions
        up_layers = layers[::-1]
        c = 128
        for l in up_layers:
            L.append(self.BlockUp(c, l, stride=2))
            c = l
        L.append(self.BlockUp(c, 32, kernel_size=7, stride=2))  # Initial conv

        # Final convolution to desired output channels
        L.append(torch.nn.Conv2d(32, n_output_channels, kernel_size=1))

        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        h = x.shape[2]
        w = x.shape[3]

        z = self.network(x)

        return z[:, :, :h, :w]

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        w = image.size(2)
        h = image.size(1)
        all_detections = []
        for i in range(image.size(0)):
            current_class = image[i]
            peaks = extract_peak(current_class, max_det=30)
            for j in peaks:
                peaks[j] = peaks[j] + (w/2, h/2)
            all_detections.append(peaks)

        return all_detections[0], all_detections[1], all_detections[2]


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
