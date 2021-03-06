import torch
from torchvision.models import vgg16_bn

class VGGRegCls(torch.nn.Module):
    def __init__(self, mini= True, grayscale= False, last_kernel=13, last_padding=3):
        super(VGGRegCls, self).__init__()
        if mini:
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1 if grayscale else 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )

            kernel_size = last_kernel

            self.last_conv = torch.nn.Sequential(
                torch.nn.Dropout(p=0.5),
                torch.nn.Conv2d(256, 500, kernel_size=kernel_size, stride=1, padding=last_padding),  # PADDING!
                torch.nn.BatchNorm2d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(),
            )

            self.regression = torch.nn.Sequential(
                torch.nn.Conv2d(500, 2, kernel_size=1, stride=1),
            )

            self.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(500, 2, kernel_size=1, stride=1),
                torch.nn.Softmax2d()
            )

        else:
            self.features = vgg16_bn(pretrained= True).features

            kernel_size = 3

            self.last_conv = torch.nn.Sequential(
                torch.nn.Dropout(p=0.5),
                torch.nn.Conv2d(512, 64, kernel_size=kernel_size, stride=1, padding=1), # IN_DIM, PADDING!
                torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(),
            )

            self.regression = torch.nn.Sequential(
                torch.nn.Conv2d(64, 2, kernel_size=1, stride=1),
            )

            self.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(64, 2, kernel_size=1, stride=1),
                torch.nn.Softmax2d()
            )

        if mini:
            for m in self.features.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                    m.bias.data.zero_()

        for m in self.regression.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)

        for m in self.classifier.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = self.last_conv(x)
        l = self.regression(x)
        c = self.classifier(x)
        return l, c