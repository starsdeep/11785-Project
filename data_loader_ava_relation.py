from __future__ import absolute_import
from __future__ import division

import pandas as pd
import numpy as np
import mxnet as mx
from video_reader_ava import sample_clips
import logging
import random
import os
import os.path as osp
import cv2
import pickle
import sys

logger = logging.getLogger(__name__)

# TODO: multiple labels! single bbox! -- provide data / label

DEBUG_NUM = 20
DATA_ROOT = '/mnt/truenas/scratch/yikangliao/dataset/AVA/train-validation'
video_id_to_rel_path = {}
for item in os.listdir(DATA_ROOT):
    video_id_to_rel_path[osp.splitext(item)[0]] = item


class ClipBatchIter(mx.io.DataIter):
    def __init__(self, config, train):
        super(ClipBatchIter, self).__init__(config.total_batch_size)

        df_path = config.df_train if train else config.df_test
        self.df = pd.read_pickle(df_path)
        self.debug = config.debug
        self.debug_dataloader = config.debug_dataloader
        if self.debug:
          self.df = self.df[:DEBUG_NUM]

        self.grouped = list(self.df.groupby(['video_id', 'timestamp', 'person_id']))

        self.batch_size = config.total_batch_size
        self.n_frame = config.n_frame
        self.min_overlap = self.n_frame // 2
        self.scale_w = config.scale_w
        self.scale_h = config.scale_h
        self.train = train
        self.n_bbox = config.n_bbox
        self.n_sample = len(self.grouped)#len(self.df)
        self.i_sample = 0
        self.batch_per_device = config.batch_per_device
        self.temporal_scale = config.temporal_scale

        self.sample_half_time = config.sample_half_time
        self.num_class = config.num_class
        self.mask = self.create_mask()
        self.reset()
        logger.info("%d sample for %s" % (self.n_sample, "train" if train else "test"))

    @property
    def provide_data(self):
        return [mx.io.DataDesc(name="data", shape=(self.batch_size, 3, self.n_frame, self.scale_h, self.scale_w),
                               dtype=np.float32, layout='NCDHW'),
                mx.io.DataDesc(name="rois", shape=(self.batch_size, self.n_frame//self.temporal_scale, self.n_bbox, 5),
                               dtype=np.float32,),
                mx.io.DataDesc(name='mask', shape=(self.batch_size, self.n_bbox), dtype=np.float32)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(name="softmax_label", shape=(self.batch_size, self.n_bbox, self.num_class), dtype=np.float32)]

    def create_mask(self):
        mask = np.zeros((self.batch_size, self.n_bbox), dtype=np.float32)
        mask[:, 0] = 1
        return mask

    def reset(self):
        self.i_sample = 0
        if self.train:
            # random shuffle
            # verified, good!
            # https://stackoverflow.com/questions/45585860/shuffle-a-pandas-dataframe-by-groups
            random.shuffle(self.grouped)

    def get_bbox_and_label(self, grouped_sample, flip, p, scale_w, scale_h):
        #print("test")
        bboxes = np.zeros((self.n_frame // self.temporal_scale, self.n_bbox, 5), dtype=np.float32)
        labels = np.zeros((self.n_bbox, self.num_class), dtype=np.float32)

        df = grouped_sample[1]


        # first bbox
        cnt = 0
        # BBOX
        x1, y1, x2, y2 = df['xmin'].values[0], df['ymin'].values[0], df['xmax'].values[0], df['ymax'].values[0]
        if self.train and flip:
            x1, x2 = 1. - x2, 1. - x1

        x1 *= scale_w
        x2 *= scale_w
        y1 *= scale_h
        y2 *= scale_h

        bboxes[:, cnt, 0] = np.arange(
            self.n_frame // self.temporal_scale) + int(
            p % self.batch_per_device) * \
                            (self.n_frame // self.temporal_scale)
        bboxes[:, cnt, 1] = x1
        bboxes[:, cnt, 2] = y1
        bboxes[:, cnt, 3] = x2
        bboxes[:, cnt, 4] = y2

        # LABEL
        labels[cnt, np.array(df['action_id'].values[0]) - 1] = 1.

        # second bbox
        cnt = 1
        # BBOX
        x1, y1, x2, y2 = df['nn_xmin'].values[0], df['nn_ymin'].values[0], df['nn_xmax'].values[0], df['nn_ymax'].values[0]
        if self.train and flip:
            x1, x2 = 1. - x2, 1. - x1

        x1 *= scale_w
        x2 *= scale_w
        y1 *= scale_h
        y2 *= scale_h

        bboxes[:, cnt, 0] = np.arange(
            self.n_frame // self.temporal_scale) + int(
            p % self.batch_per_device) * \
                            (self.n_frame // self.temporal_scale)
        bboxes[:, cnt, 1] = x1
        bboxes[:, cnt, 2] = y1
        bboxes[:, cnt, 3] = x2
        bboxes[:, cnt, 4] = y2
        # do nothing for label
        return bboxes, labels

    def next(self):
        """Get next data batch from iterator.
        Returns
        -------
        DataBatch
            The data of next batch.
        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """

        if self.i_sample < self.n_sample:
            df_batch = self.grouped[self.i_sample:min(self.n_sample, self.i_sample + self.batch_size)]
            # at end of epoch, number of sample remains may be smaller than batch size
            if len(df_batch) < self.batch_size:
                df_sample = random.sample(self.grouped, self.batch_size-len(df_batch))
                df_batch = pd.concat([df_batch, df_sample])
            assert len(df_batch) == self.batch_size

            # get random frame_idxs
            if self.train:
                flips = np.random.choice(a=[False, True], size=(self.batch_size,), p=[0.5, 0.5])
            else:
                flips = np.zeros(self.batch_size, dtype=bool)


            video = sample_clips(df_batch, flips, self.batch_size, self.n_frame,
                                 self.scale_w, self.scale_h, self.sample_half_time, self.train)

            bboxes = np.zeros((self.batch_size, self.n_frame // self.temporal_scale, self.n_bbox, 5))
            labels = np.zeros((self.batch_size, self.n_bbox, self.num_class))
            for i in range(len(df_batch)):
                tmp_bbox, tmp_label = self.get_bbox_and_label(df_batch[i], flips[i], i, self.scale_w, self.scale_h)
                bboxes[i] = tmp_bbox
                labels[i] = tmp_label

            #print(video.shape, bboxes.shape, labels.shape)
            ret = mx.io.DataBatch(data=[mx.nd.array(video), mx.nd.array(bboxes), mx.nd.array(self.mask)],
                                  label=[mx.nd.array(labels),],
                                  provide_data=self.provide_data,
                                  provide_label=self.provide_label)

            self.i_sample += self.batch_size
            return ret
        else:
            raise StopIteration
