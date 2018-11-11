# Yikang Liao <yikang.liao@tusimple.ai>
# Training Module For R2Plus1D Network

import logging
import argparse
import os
import sys
import mxnet as mx
from utils import load_from_caffe2_pkl
from symbols.r3d import create_r3d
from data_loader import ClipBatchIter
import pandas as pd
from metric import RCNNAccMetric
from config import config, update_config

def train(config):
    gpus = [int(i) for i in config.gpus.split(',')]
    num_gpus = len(gpus)

    logging.info("number of gpu %d" % num_gpus)

    if len(gpus) == 0:
        kv = None
    else:
        kv = mx.kvstore.create('local')
    logging.info("Running on GPUs: {}".format(gpus))

    # Modify to make it consistent with the distributed trainer
    total_batch_size = config.batch_per_device * num_gpus

    # Create symbol, arg and aux
    if config.begin_epoch>0:
        sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(config.output, 'test'), config.begin_epoch)
    else:
        # Create Network
        sym = create_r3d(
            num_class=config.num_class,
            no_bias=True,
            model_depth=config.model_depth,
            final_spatial_kernel=config.final_spatial_kernel,
            final_temporal_kernel=int(config.n_frame / 8),
            bn_mom=config.bn_mom,
            cudnn_tune=config.cudnn_tune,
            workspace=config.workspace,
            spatial_scale=720.0 / config.scale_w * config.spatial_scale,
            pooled_size=config.pooled_size,
            n_frame=config.n_frame,
            n_bbox=config.n_bbox,
        )
        # Load pretrained params
        arg_params, aux_params = {}, {}
        if config.pretrained:
            arg_params, aux_params = load_from_caffe2_pkl(config.pretrained, sym)
        logging.info("load pretrained okay, num of arg_p %d, num of aux_p %d" % (len(arg_params), len(aux_params)))

    # Create Module
    # We can set fixed params here if needed
    m = mx.module.Module(sym, context=[mx.gpu(i) for i in gpus], data_names=['data', 'rois'],
                         label_names=['softmax_label'])

    if config.plot:
        v = mx.viz.plot_network(sym, title='R2Plus1D-train',
                                shape={'data': (total_batch_size, 3, config.n_frame, config.scale_h, config.scale_w),
                                       'rois': (total_batch_size, config.n_frame // config.temporal_scale, config.n_bbox, 5),
                                       'softmax_label': (total_batch_size, config.n_bbox)})
        v.render(filename=os.path.join(config.output, 'vis'), cleanup=True)

    df_train = pd.read_pickle(config.df_train)
    df_test = pd.read_pickle(config.df_test)
    train_data = mx.io.PrefetchingIter(ClipBatchIter(df=df_train, batch_size=total_batch_size,
                                                     n_frame=config.n_frame, train=True, n_bbox=config.n_bbox,
                                                     scale_w=config.scale_w, scale_h=config.scale_h,
                                                     batch_per_device=config.batch_per_device,
                                                     temporal_scale=config.temporal_scale,
                                                     use_large_bbox=config.use_large_bbox))
    test_data = mx.io.PrefetchingIter(ClipBatchIter(df=df_test, batch_size=total_batch_size,
                                                    n_frame=config.n_frame, train=False, n_bbox=config.n_bbox,
                                                    scale_w=config.scale_w, scale_h=config.scale_h,
                                                    batch_per_device=config.batch_per_device,
                                                    temporal_scale=config.temporal_scale,
                                                    use_large_bbox=config.use_large_bbox))

    # Set optimizer
    optimizer = config.optimizer
    optimizer_params = {}
    optimizer_params['learning_rate'] = config.lr
    optimizer_params['momentum'] = config.momentum
    optimizer_params['wd'] = config.wd

    print(config.lr)
    print(config.lr_step)

    if config.lr_step:
        optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=config.lr_step,
                                                                           factor=config.lr_factor)
    metric = RCNNAccMetric()

    m.fit(
        train_data=train_data,
        eval_data=test_data,
        eval_metric=metric,
        epoch_end_callback=mx.callback.do_checkpoint(config.output + '/test', 1),
        batch_end_callback=mx.callback.Speedometer(total_batch_size, 20),
        kvstore=kv,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        begin_epoch=config.begin_epoch,
        num_epoch=config.num_epoch,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train R2Plus1D Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    # update config
    config = update_config(args.cfg)

    # Create Output Dir
    if not os.path.exists(config.output):
        os.makedirs(config.output)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(config.output, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(config)

    train(config)