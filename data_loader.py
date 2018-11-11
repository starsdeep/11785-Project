from __future__ import absolute_import
from __future__ import division

import pandas as pd
import numpy as np
import mxnet as mx
from video_reader import sample_clips
import logging
import random

logger = logging.getLogger(__name__)


class ClipBatchIter(mx.io.DataIter):
    def __init__(self, df, batch_size=8, n_frame=16, scale_w=512, scale_h=288, train=True, n_bbox=3, batch_per_device=4,
                 temporal_scale=4, use_large_bbox=1):
        super(ClipBatchIter, self).__init__(batch_size)
        self.df = df
        self.batch_size = batch_size
        self.n_frame = n_frame
        self.min_overlap = self.n_frame // 2
        self.scale_w = scale_w
        self.scale_h = scale_h
        self.train = train
        self.n_bbox = n_bbox
        self.n_sample = len(self.df)
        self.i_sample = 0
        self.batch_per_device = batch_per_device
        self.temporal_scale = temporal_scale
        self.use_large_bbox = use_large_bbox
        self.reset()
        logger.info("%d sample for %s" % (len(self.df), "train" if train else "test"))

    @property
    def provide_data(self):
        return [mx.io.DataDesc(name="data", shape=(self.batch_size, 3, self.n_frame, self.scale_h, self.scale_w),
                               dtype=np.float32, layout='NCDHW'),
                mx.io.DataDesc(name="rois", shape=(self.batch_size, self.n_frame//self.temporal_scale, self.n_bbox, 5),
                               dtype=np.float32,)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc(name="softmax_label", shape=(self.batch_size, self.n_bbox,), dtype=np.float32, layout='N')]


    def reset(self):
        self.i_sample = 0
        if self.train:
            # random shuffle
            self.df = self.df.sample(frac=1)

    def get_bbox_and_label(self, actions, flip, frame_st, p):
        #print("test")
        bboxes = np.zeros((self.n_frame // self.temporal_scale, self.n_bbox, 5))
        labels = np.zeros(self.n_bbox)
        cnt = 0
        for i, action in enumerate(actions):
            if cnt >= self.n_bbox:
                break
            s1, e1, s2, e2 = frame_st, frame_st + self.n_frame - 1, action[1], action[2]
            if s1 > s2:
                s1, e1, s2, e2 = s2, e2, s1, e1
            # find qualified overlap
            if e1 - s2 + 1 >= self.min_overlap:
                # get bbox
                large_bbox = action[3:7]

                last_bbox_idx = (len(action[7])-1) if frame_st-action[1]+self.n_frame-1 >= len(action[7]) else \
                    frame_st-action[1]+self.n_frame-1
                last_bbox = action[7][last_bbox_idx] if action[7][last_bbox_idx][0]==-1 else action[7][last_bbox_idx-1]
                x1, y1, x2, y2 = large_bbox if self.use_large_bbox else last_bbox
                if self.train and flip:
                    x1, x2 = 720 - x2, 720 - x1

                # set bboxes for mxnet
                bboxes[:, cnt, 0] = np.arange(self.n_frame//self.temporal_scale) + int(p % self.batch_per_device) * \
                                    (self.n_frame//self.temporal_scale)
                bboxes[:, cnt, 1] = x1
                bboxes[:, cnt, 2] = y1
                bboxes[:, cnt, 3] = x2
                bboxes[:, cnt, 4] = y2

                # set label
                labels[cnt] = action[0]
                cnt += 1

        # fill remaining elements
        for i in range(cnt, 3):
            bboxes[:, i, :] = bboxes[:, cnt-1, :]
            labels[i] = labels[cnt-1]
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
        self.flips = np.random.choice(a=[False, True], size=(self.batch_size,), p=[0.5, 0.5])


        if self.i_sample < self.n_sample:
            df_batch = self.df.iloc[self.i_sample: min(self.n_sample, self.i_sample + self.batch_size)]
            # at end of epoch, number of sample remains may be smaller than batch size
            if len(df_batch) < self.batch_size:
                df_sample = self.df.sample(n=self.batch_size-len(df_batch))
                df_batch = pd.concat([df_batch, df_sample])
            assert len(df_batch) == self.batch_size

            # get random frame_idxs
            if self.train:
                flips = np.random.choice(a=[False, True], size=(self.batch_size,), p=[0.5, 0.5])
            else:
                flips = np.zeros(self.batch_size, dtype=bool)

            frame_idxs = [0 if row['length']<=self.n_frame else random.randrange(row['length'] - self.n_frame + 1) \
                          for index, row in df_batch.iterrows()]

            video = sample_clips(df_batch['filename'].values, df_batch['length'].values, frame_idxs, flips, self.batch_size, self.n_frame,
                                 self.scale_w, self.scale_h, self.train)

            bboxes = np.zeros((self.batch_size, self.n_frame // self.temporal_scale, self.n_bbox, 5))
            labels = np.zeros((self.batch_size, self.n_bbox))
            for i in range(len(df_batch)):
                tmp_bbox, tmp_label = self.get_bbox_and_label(df_batch.iloc[i]['actions'], flips[i], frame_idxs[i], i)
                bboxes[i] = tmp_bbox
                labels[i] = tmp_label

            #print(video.shape, bboxes.shape, labels.shape)
            ret = mx.io.DataBatch(data=[mx.nd.array(video), mx.nd.array(bboxes)],
                                  label=[mx.nd.array(labels),],
                                  provide_data=self.provide_data,
                                  provide_label=self.provide_label)

            self.i_sample += self.batch_size
            return ret
        else:
            raise StopIteration
