import yaml
from easydict import EasyDict as edict

config = edict()

# basic configuration
config.MXNET_VERSION = ''
config.gpus = ''
config.plot = True
config.cudnn_tune = 'off'
config.workspace = 512
config.batch_per_device = 4

# input and output
config.output = ''
config.df_test = ''
config.df_train = ''
config.pretrained = ''

# network related params
config.model_depth = 18
config.pooled_size = 10
config.final_spatial_kernel = 5
config.spatial_scale = 8
config.temporal_scale = 4
config.n_frame = 16
config.num_class = 10
config.scale_w = 180
config.scale_h = 144
config.n_bbox = 3
config.use_large_bbox = True

# relation related params
config.nongt_dim = 2
config.geometric_dim = 32

# training related params
config.num_epoch = 90
config.optimizer = 'sgd'
config.lr = 1e-4
config.lr_step = 0
config.lr_factor = 0.1
config.warmup = False
config.warmup_lr = 0
config.warmup_step = 0
config.momentum = 0.9
config.bn_mom = 0.9
config.wd = 1e-4
config.begin_epoch = 0
config.model_prefix = 'test'


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            # if k in config:
            if isinstance(v, dict):
                for vk, vv in v.items():
                    config[k][vk] = vv
            else:
                    config[k] = v
            # else:
            #     pass
                # raise ValueError("key must exist in config.py")
    return config