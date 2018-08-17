import os.path as osp
import models

# configuration file
description  = 'baseline'
version_string  = '0.1'

# device can be "cuda" or "gpu"
device = 'cuda'
num_workers = 4
available_gpus = '1,6'
print_freq = 15

result_root = '/home/fengyifan/result/vgcnn'
result_sub_folder = osp.join(result_root, f'{description}_{version_string}')
ckpt_folder = osp.join(result_sub_folder, 'ckpt')

# base_model_name = models.ALEXNET
base_model_name = models.VGG13BN
# base_model_name = models.INCEPTION_V3
# base_model_name = models.VGG13
# base_model_name = models.RESNET50
# base_model_name = models.RESNET101

# aggrategor = models.MaxPooling
aggrategor = models.FeatureBuildGraph

class view_net:
    num_classes = 40
    n_neighbor = 2

    # multi-view cnn
    # data_root = '/repository/12_ModelNet40'
    data_root = '/home/fengyifan/data/12_ModelNet40'

    pre_trained_model = None
    if aggrategor == models.MaxPooling:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-{aggrategor}-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'{base_model_name}-{aggrategor}-record')
    else:
        ckpt_file = osp.join(ckpt_folder, f'{base_model_name}-{aggrategor}-neig-{n_neighbor}-ckpt.pth')
        ckpt_record_folder = osp.join(ckpt_folder, f'{base_model_name}-{aggrategor}-neig-{n_neighbor}-record')

    class train:
        if base_model_name == models.ALEXNET:
            if aggrategor == models.MaxPooling:
                batch_sz = 128 # AlexNet 2 gpus
            else:
                batch_sz = 120
        elif base_model_name == models.INCEPTION_V3:
            batch_sz = 2
        elif base_model_name in (models.VGG13BN, models.VGG13):
            batch_sz = 10
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
