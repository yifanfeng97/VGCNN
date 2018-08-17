import torch
import config
import torchvision
import torch.nn as nn
import models


class BaseFeatureNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, pretrained=True):
        super(BaseFeatureNet, self).__init__()
        base_model_name = base_model_name.upper()
        self.fc_features = None

        if base_model_name == models.VGG13:
            base_model = torchvision.models.vgg13(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.VGG13BN:
            base_model = torchvision.models.vgg13_bn(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.ALEXNET:
            base_model = torchvision.models.alexnet(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.RESNET50:
            base_model = torchvision.models.resnet50(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif base_model_name == models.RESNET101:
            base_model = torchvision.models.resnet101(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])

        elif base_model_name == models.INCEPTION_V3:
            base_model = torchvision.models.inception_v3(pretrained=pretrained)
            base_model_list = list(base_model.children())[0:13]
            base_model_list.extend(list(base_model.children())[14:17])
            self.features = nn.Sequential(*base_model_list)
            self.feature_len = 2048

        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

    def forward(self, x):
        # fetch view num
        view_num = x.size(1)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))

        # forward
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x) if self.fc_features is not None else x

        # reshape to a normal shape
        x = x.view(-1, view_num, x.size(1))

        return x


class BaseClassifierNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, num_classes=40, pretrained=True):
        super(BaseClassifierNet, self).__init__()
        base_model_name = base_model_name.upper()
        if base_model_name in (models.VGG13, models.VGG13BN, models.ALEXNET):
            self.feature_len = 4096
        elif base_model_name in (models.RESNET50, models.RESNET101, models.INCEPTION_V3):
            self.feature_len = 2048
        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

        self.classifier = nn.Linear(self.feature_len, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGCNN, self).__init__()
        base_model_name = config.base_model_name
        num_classes = config.view_net.num_classes
        print(f'\ninit {base_model_name} model...\n')
        self.features = BaseFeatureNet(base_model_name, pretrained)
        self.classifier = BaseClassifierNet(base_model_name, num_classes, pretrained)
        print(f'init {base_model_name} model... Down!\n')
        if config.aggrategor == models.MaxPooling:
            self.aggregator = models.Max_Viewpooling()
        elif config.aggrategor == models.FeatureBuildGraph:
            self.aggregator = models.Feature_Viewpooling()
        elif config.aggrategor == models.AngleBuildGraph:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = self.aggregator(x)
        x = self.classifier(x)
        return x


