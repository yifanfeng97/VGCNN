import os.path as osp
import models

# configuration file
description  = 'baseline'
# description = 'fc_no_relu'
version_string = '0.1'

# device can be "cuda" or "gpu"
device = 'cuda'
num_workers = 4
# available_gpus = '4,5'
available_gpus = '2,3'
print_freq = 15
num_views = 20

result_root = '/home/fengyifan/result/vgcnn'
result_sub_folder = osp.join(result_root, f'{description}_{version_string}')
ckpt_folder = osp.join(result_sub_folder, 'ckpt')

# base_model_name = models.ALEXNET
# base_model_name = models.VGG13
base_model_name = models.VGG13BN
# base_model_name = models.INCEPTION_V3
# base_model_name = models.RESNET50
# base_model_name = models.RESNET101

aggrategor = models.MaxPooling
# aggrategor = models.FeatureBuildGraph

class view_net:
    num_classes = 40
    n_neighbor = 2

    # multi-view cnn
    # data_root = '/repository/12_ModelNet40'
    if num_views == 8:
        data_root = '/repository/8_ModelNet40'
    elif num_views == 12:
        data_root = '/repository/12_ModelNet40'
    elif num_views == 20:
        data_root = '/repository/20_ModelNet40'
    else:
        raise NotImplementedError

    pre_trained_model = None
    if aggrategor == models.MaxPooling:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-{num_views}VIEWS-{aggrategor}-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'{base_model_name}-{num_views}VIEWS-{aggrategor}-record')
    else:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-{num_views}VIEWS-{aggrategor}-neig-{n_neighbor}-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'{base_model_name}-{num_views}VIEWS-{aggrategor}-neig-{n_neighbor}-record')

    class train:
        if base_model_name == models.ALEXNET:
            if aggrategor == models.MaxPooling:
                if num_views == 12:
                    batch_sz = 128 # AlexNet 2 gpus
                elif num_views == 20:
                    batch_sz = 40
            else:
                if num_views == 12:
                    batch_sz = 120
                elif num_views == 20:
                    batch_sz = 60
        elif base_model_name == models.INCEPTION_V3:
            batch_sz = 2
        elif base_model_name in (models.VGG13BN, models.VGG13):
            if num_views == 12:
                batch_sz = 10
            elif num_views == 20:
                batch_sz = 6
        else:
            batch_sz = 32
        resume = False
        resume_epoch = None

        lr = 0.001
        momentum = 0.9
        weight_decay = 1e-4
        max_epoch = 40
        data_aug = True

    class validation:
        batch_sz = 256

    class test:
        batch_sz = 32
