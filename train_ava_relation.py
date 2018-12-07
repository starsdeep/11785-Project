import logging
import argparse
import os
import sys
import mxnet as mx
import numpy as np
from utils import load_from_caffe2_pkl
from symbols.r3d_multiclass_relation import create_r3d
from data_loader_ava_relation import ClipBatchIter
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
    config.total_batch_size = total_batch_size

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
            spatial_scale=config.spatial_scale,
            pooled_size=config.pooled_size,
            n_frame=config.n_frame,
            n_bbox=config.n_bbox,
            batch_per_device=config.batch_per_device,
            geometric_dim=config.geometric_dim,
            nongt_dim=config.nongt_dim,
        )
        # Load pretrained params
        arg_params, aux_params = {}, {}
        if config.pretrained:
            arg_params, aux_params = load_from_caffe2_pkl(config.pretrained, sym)
        logging.info("load pretrained okay, num of arg_p %d, num of aux_p %d" % (len(arg_params), len(aux_params)))

    # Create Module
    # We can set fixed params here if needed
    m = mx.module.Module(sym, context=[mx.gpu(i) for i in gpus], data_names=['data', 'rois', 'mask'],
                         label_names=['softmax_label'])

    if config.plot:
        v = mx.viz.plot_network(sym, title='R2Plus1D-train',
                                shape={'data': (total_batch_size, 3, config.n_frame, config.scale_h, config.scale_w),
                                       'rois': (total_batch_size, config.n_frame // config.temporal_scale, config.n_bbox, 5),
                                       'mask': (total_batch_size, config.n_bbox),
                                       'softmax_label': (total_batch_size, config.n_bbox, config.num_class)})
        v.render(filename=os.path.join(config.output, 'vis'), cleanup=True)

    train_data = mx.io.PrefetchingIter(ClipBatchIter(config=config, train=True))
    test_data = mx.io.PrefetchingIter(ClipBatchIter(config=config, train=False))

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
    # metric = RCNNAccMetric()


    def acc(label, pred):
        label = label.reshape((-1, config.num_class))
        # print('in acc, pred.size', pred.size, 'pred.shape', pred.shape, 'label.shape', label.shape, 'numerator', (label == np.round(pred)).sum(), 'res', float((label == np.round(pred)).sum()) / pred.size)
        return (label == np.round(pred)).astype(np.float32).mean()

    def all_correct_acc(label, pred):
        label = label.reshape((-1, config.num_class))
        # print('in acc, pred.size', pred.size, 'pred.shape', pred.shape, 'label.shape', label.shape, 'numerator', (label == np.round(pred)).sum(), 'res', float((label == np.round(pred)).sum()) / pred.size)
        equal = (label == np.round(pred)).astype(np.int32)
        equal_sum = equal.sum(axis=-1)
        return (equal_sum == label.shape[-1]).astype(np.float32).mean()


    def loss(label, pred):
        label = label.reshape((-1, config.num_class))
        loss_all = 0
        for i in range(len(pred)):
            loss = 0
            loss -= label[i] * np.log(pred[i] + 1e-6) + (1.- label[i]) * np.log(1. + 1e-6 - pred[i])
            loss_all += np.sum(loss)
        loss_all = float(loss_all)/float(len(pred) + 0.000001)
        return loss_all

    eval_metric = list()
    eval_metric.append(mx.metric.np(acc))
    eval_metric.append(mx.metric.np(all_correct_acc))
    eval_metric.append(mx.metric.np(loss))

    m.fit(
        train_data=train_data,
        eval_data=test_data,
        eval_metric=eval_metric,
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