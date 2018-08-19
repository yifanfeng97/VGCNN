import torch
import config
import models
import torch.nn as nn

# Basemodel
ALEXNET = "ALEXNET"
DENSENET121 = "DENSENET121"
VGG13 = "VGG13"
VGG13BN = "VGG13BN"
RESNET50 = "RESNET50"
RESNET101 = "RESNET101"
INCEPTION_V3 = 'INVEPTION_V3'

# Aggregator
MaxPooling = "MAX_POOLING"
FeatureBuildGraph = 'FEATURE_BUILD_GRAPH'
AngleBuildGraph = 'ANGLE_BUILD_GRAPH'


# MVCNN functions
class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, relu=True):
        super(fc_layer, self).__init__()
        self.fc = [nn.Linear(in_ch, out_ch)]
        if bn:
            self.fc.append(nn.BatchNorm1d(out_ch))
        if relu:
            self.fc.append(nn.ReLU(inplace=True))

        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1024, 1))
        self.fc1 = fc_layer(1024, 512, bn=True)
        self.fc2 = fc_layer(512, 256, bn=True)
        self.fc3 = nn.Linear(256, K*K)


    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x, _ = torch.max(x, dim=-1, keepdim=True)
        x = self.conv2d3(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(3).view(1,9).repeat(x.size(0),1)
        iden = iden.to(device=config.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


def pairwise_distance(x):
    batch_size = x.size(0)
    point_cloud = torch.squeeze(x)
    if batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    point_cloud_transpose = torch.transpose(point_cloud, dim0=1, dim1=2)
    point_cloud_inner = torch.matmul(point_cloud_transpose, point_cloud)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    point_cloud_square_transpose = torch.transpose(point_cloud_square, dim0=1, dim1=2)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def gather_neighbor(x, nn_idx, n_neighbor):
    x = torch.squeeze(x, 3)
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x.unsqueeze(2).expand(batch_size, num_dim, num_point, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, num_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_feature_based_edge_feature(x, n_neighbor):
    x = torch.transpose(x, dim0=1, dim1=2)
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors - point_cloud_center), dim=1)

    return edge_feature


def get_angle_based_edge_feature(x, n_neighbor):
    x = torch.transpose(x, dim0=1, dim1=2)
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    nn_idx = torch.Tensor([])
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors - point_cloud_center), dim=1)

    return edge_feature


class Max_Viewpooling(nn.Module):
    def __init__(self):
        super(Max_Viewpooling, self).__init__()

    def forward(self, x):
        x, _ = torch.max(x, 1)
        x = x.view(x.size(0), -1)
        return x


class Feature_Viewpooling(nn.Module):
    def __init__(self):
        super(Feature_Viewpooling, self).__init__()
        self.n_neig = config.view_net.n_neighbor
        if config.base_model_name in (models.VGG13, models.VGG13BN, models.ALEXNET):
            self.feature_len = 4096
        elif config.base_model_name in (models.RESNET50, models.RESNET101, models.INCEPTION_V3):
            self.feature_len = 2048
        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')
        self.conv2d1 = models.conv_2d(self.feature_len*2, self.feature_len, 1)

    def forward(self, x):
        x = get_feature_based_edge_feature(x, self.n_neig)
        x = self.conv2d1(x)
        x, _ = torch.max(x, dim=-1, keepdim=True)
        x, _ = torch.max(x, dim=2, keepdim=True)
        return x


class Angle_Viewpooling(nn.Module):
    def __init__(self):
        super(Angle_Viewpooling, self).__init__()
        self.n_neig = config.view_net.n_neighbor
        if config.base_model_name in (models.VGG13, models.VGG13BN, models.ALEXNET):
            self.feature_len = 4096
        elif config.base_model_name in (models.RESNET50, models.RESNET101, models.INCEPTION_V3):
            self.feature_len = 2048
        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')
        self.conv2d1 = models.conv_2d(self.feature_len*2, self.feature_len, 1)

    def forward(self, x):
        x = get_angle_based_edge_feature(x, self.n_neig)
        x = self.conv2d1(x)
        x, _ = torch.max(x, dim=-1, keepdim=True)
        x, _ = torch.max(x, dim=2, keepdim=True)
        return x
