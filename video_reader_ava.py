import logging
import multiprocessing as mp
import ctypes
import numpy as np
import cv2
import random
import os
import os.path as osp


logger = logging.getLogger(__name__)

PROCESSES = 32
MAX_FLOAT_NUM = 1024 * 3 * 32 * 224 * 224
ret_base = mp.RawArray(ctypes.c_float, MAX_FLOAT_NUM)
counter = mp.RawValue(ctypes.c_int, 0)
DATA_ROOT = '/mnt/truenas/scratch/yikangliao/dataset/AVA/train-validation'
video_id_to_rel_path = {}

for item in os.listdir(DATA_ROOT):
    video_id_to_rel_path[osp.splitext(item)[0]] = item

# TODO: check if end frame idx is valid

#sampled_df, flips, p, batch_size, n_frame, scale_w,
#                                         scale_h, is_train
def sample_clip_func((info, p, flips, batch_size, n_frame, scale_w, scale_h, sample_half_time, is_train)):
    video_id, timestamp, person_id = info
    video_path = osp.join(DATA_ROOT, video_id_to_rel_path[video_id])
        
    
    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * scale_h * scale_w).reshape(
        (batch_size, 3, n_frame, scale_h, scale_w))

    # print('video_path', video_path, timestamp, person_id)
    v = cv2.VideoCapture(video_path)
    assert v.isOpened(), 'video open err:' + video_path

    fps = v.get(cv2.CAP_PROP_FPS)
    start_frame_idx = int(round((timestamp - sample_half_time) * fps))
    # center_frame_idx = int(round(timestamp * fps))
    end_frame_idx = int(round((timestamp + sample_half_time) * fps))

    width, height, length = v.get(cv2.CAP_PROP_FRAME_WIDTH), v.get(
        cv2.CAP_PROP_FRAME_HEIGHT), v.get(cv2.CAP_PROP_FRAME_COUNT)
    assert scale_w <= width and scale_h <= height, \
        '%d <= %d ; %d <= %d ; %d <= %d' % (
        end_frame_idx, length, scale_w, width, scale_h, height)

    tmp = None
    frame_cnt = 0
    while tmp is None:
        v = cv2.VideoCapture(video_path)

        # random sample frame
        tmp = np.zeros((n_frame, scale_h, scale_w, 3), dtype=np.float32)
        
        frame_st = random.randrange(start_frame_idx, end_frame_idx - n_frame + 1)
        v.set(cv2.CAP_PROP_POS_FRAMES, frame_st)

        for i in xrange(n_frame):
            read_ret, f = v.read()
            assert read_ret
            if f is not None:
                tmp[i, ...] = cv2.resize(f, (scale_w, scale_h))
                frame_cnt += 1
            else:
                tmp = None
                counter.value = 1
                logging.warning("%s, frame st %d, length %d, n_frame %d" % (video_id, frame_st, length, n_frame))
                break
    v.release()

    # tmp is D,H,W,C
    # Temporal transform: looping
    if frame_cnt < n_frame:
        tmp[-(n_frame - frame_cnt):] = tmp[:(n_frame - frame_cnt)]

    tmp = tmp.transpose((3, 0, 1, 2))
    # now tmp is C,D,H,W
    # random flip the video horizontally
    if is_train and flips[p]:
        tmp = np.flip(tmp, 3)

    ret[p, ...] = tmp

# sample_clips(df_batch, flips, self.batch_size, self.n_frame,
#                                            self.scale_w, self.scale_h, self.train)
def sample_clips(sampled_df, flips, batch_size, n_frame, scale_w=171, scale_h=128, sample_half_time=1., is_train=True):

    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * scale_h * scale_w).reshape(
        (batch_size, 3, n_frame, scale_h, scale_w))
    counter = 0

    process_pool.map(sample_clip_func, [(single_sampled_df[0], p, flips, batch_size, n_frame, scale_w,
                                         scale_h, sample_half_time, is_train) for p,single_sampled_df in enumerate(sampled_df)])

    # for p, single_sampled_df in enumerate(sampled_df):
    #     sample_clip_func((single_sampled_df[0], p, flips, batch_size, n_frame, scale_w,
    #      scale_h, sample_half_time, is_train))

    # for p in xrange(len(filenames)):
    #     sample_clip_func((filenames, lengths, frame_idxs, flips, p, batch_size, n_frame, scale_w, scale_h, is_train))

    assert ret.dtype == np.float32 and ret.shape == (batch_size, 3, n_frame, scale_h, scale_w)

    # normalize here
    m = np.mean(ret, axis=(0, 2, 3, 4))
    std = np.std(ret, axis=(0, 2, 3, 4))
    for i in range(3):
        ret[:, i, :, :, :] = (ret[:, i, :, :, :] - m[i]) / (std[i] + 1e-3)

    if counter:
        logger.fatal("read invalid frames")
        exit(1)


    return ret

process_pool = mp.Pool(processes=PROCESSES)



