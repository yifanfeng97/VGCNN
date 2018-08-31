import config
import torchvision
from models import *


class _vgg11bn(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base_model = torchvision.models.vgg11_bn(pretrained=pretrained)
        self.features = base_model.features
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(4096, config.view_net.num_classes)

    def forward(self, x):
        ft1 = self.features[:]

