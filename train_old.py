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


TSCALE = 4

def train(args):
    gpus = [int(i) for i in args.gpus.split(',')]
    num_gpus = len(gpus)

    logging.info("number of gpu %d" % num_gpus)

    if len(gpus) == 0:
        kv = None
    else:
        kv = mx.kvstore.create('local')
    logging.info("Running on GPUs: {}".format(gpus))

    # Modify to make it consistent with the distributed trainer
    total_batch_size = args.batch_per_device * num_gpus

    # Create symbol, arg and aux
    if args.begin_epoch>0:
        sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(args.output, 'test'), args.begin_epoch)
    else:
        # Create Network
        sym = create_r3d(
            num_class=args.num_class,
            no_bias=True,
            model_depth=args.model_depth,
            final_spatial_kernel=args.final_spatial_kernel,
            final_temporal_kernel=int(args.n_frame / 8),
            bn_mom=args.bn_mom,
            cudnn_tune=args.cudnn_tune,
            workspace=args.workspace,
            spatial_scale=720.0 / args.scale_w * args.spatial_scale_factor,
            pooled_size=args.pooled_size,
            n_frame=args.n_frame,
        )
        # Load pretrained params
        arg_params, aux_params = {}, {}
        if args.pretrained:
            arg_params, aux_params = load_from_caffe2_pkl(args.pretrained, sym)
        logging.info("load pretrained okay, num of arg_p %d, num of aux_p %d" % (len(arg_params), len(aux_params)))

    # Create Module
    # We can set fixed params here if needed
    m = mx.module.Module(sym, context=[mx.gpu(i) for i in gpus], data_names=['data', 'rois'],
                         label_names=['softmax_label'])

    if args.plot:
        v = mx.viz.plot_network(sym, title='R2Plus1D-train',
                                shape={'data': (total_batch_size, 3, args.n_frame, args.scale_h, args.scale_w),
                                       'rois': (total_batch_size, args.n_frame // TSCALE, args.n_bbox, 5),
                                       'softmax_label': (total_batch_size, args.n_bbox)})
        v.render(filename=os.path.join(args.output, 'vis'), cleanup=True)

    df_train = pd.read_pickle(args.df_train)
    df_test = pd.read_pickle(args.df_test)
    train_data = mx.io.PrefetchingIter(ClipBatchIter(df=df_train, batch_size=total_batch_size,
                                                     n_frame=args.n_frame, train=True, n_bbox=args.n_bbox,
                                                     scale_w=args.scale_w, scale_h=args.scale_h,
                                                     batch_per_device=args.batch_per_device, tscale=args.tscale,
                                                     large_box=args.large_box))
    test_data = mx.io.PrefetchingIter(ClipBatchIter(df=df_test, batch_size=total_batch_size,
                                                    n_frame=args.n_frame, train=False, n_bbox=args.n_bbox,
                                                    scale_w=args.scale_w, scale_h=args.scale_h,
                                                    batch_per_device=args.batch_per_device, tscale=args.tscale,
                                                    large_box=args.large_box))

    # Set optimizer
    optimizer = args.optimizer
    optimizer_params = {}
    optimizer_params['learning_rate'] = args.lr
    optimizer_params['momentum'] = args.momentum
    optimizer_params['wd'] = args.wd

    if args.lr_scheduler_step:
        optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=args.lr_scheduler_step,
                                                                           factor=args.lr_scheduler_factor)
    metric = RCNNAccMetric()

    m.fit(
        train_data=train_data,
        eval_data=test_data,
        eval_metric=metric,
        epoch_end_callback=mx.callback.do_checkpoint(args.output + '/test', 1),
        batch_end_callback=mx.callback.Speedometer(total_batch_size, 20),
        kvstore=kv,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        begin_epoch=args.begin_epoch,
        num_epoch=args.num_epoch,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training p3d network")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--plot', type=int, default=1, help='plot the network architecture')
    parser.add_argument('--pretrained', type=str, default='/mnt/truenas/scratch/yikangliao/pretrained_models/r2.5d_d18_l16.pkl')
    parser.add_argument('--df_train', type=str, default='/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/train-df.pickle')
    parser.add_argument('--df_test', type=str, default='/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/valid-df.pickle')
    parser.add_argument('--output', type=str, default='/mnt/truenas/scratch/yikangliao/output_liris_test')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--cudnn_tune', type=str, default='off', help='optimizer')
    parser.add_argument('--workspace', type=int, default=512, help='workspace for GPU')
    parser.add_argument('--lr_scheduler_step', type=int, default=0, help='reduce lr after n step')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr scheduler factor')
    parser.add_argument('--lr', type=float, default=5e-5, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=5e-3, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--num_class', type=int, default=10, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=18, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=1000, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=16, help='the number of frame to sample from a video')
    parser.add_argument('--scale_w', type=int, default=180, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=144, help='the rescaled height of image')
    parser.add_argument('--n_bbox', type=int, default=3)
    parser.add_argument('--pooled_size', type=int, default=10)
    parser.add_argument('--final_spatial_kernel', type=int, default=5)
    parser.add_argument('--spatial_scale_factor', type=int, default=8)
    parser.add_argument('--tscale', type=int, default=4)
    parser.add_argument('--large_box', type=int, default=1)

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.output, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)

    train(args)