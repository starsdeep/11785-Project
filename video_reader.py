import logging
import multiprocessing as mp
import ctypes
import numpy as np
import cv2
import random

logger = logging.getLogger(__name__)

PROCESSES = 64
MAX_FLOAT_NUM = 1024 * 3 * 32 * 224 * 224
ret_base = mp.RawArray(ctypes.c_float, MAX_FLOAT_NUM)
counter = mp.RawValue(ctypes.c_int, 0)

def get_actions(actions, n_frame, frame_st, min_overlap):
    action_ids = []
    for i, action in enumerate(actions):
        s1, e1, s2, e2 = frame_st, frame_st+n_frame-1, action[1], action[2]
        if s1>s2:
            s1, e1, s2, e2 = s2, e2, s1, e1
        if e1-s2+1 >= min_overlap:
            action_ids.append(i)
    return action_ids


def sample_clip_func((filenames, lengths, frame_idxs, flips, p, batch_size, n_frame, scale_w, scale_h, is_train)):
    filename, frame_st = filenames[p], frame_idxs[p]
    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * scale_h * scale_w).reshape(
        (batch_size, 3, n_frame, scale_h, scale_w))
    tmp = None
    while tmp is None:
        v = cv2.VideoCapture(filename)
        width, height, length = v.get(cv2.CAP_PROP_FRAME_WIDTH), v.get(cv2.CAP_PROP_FRAME_HEIGHT), lengths[p]

        assert scale_w <= width and scale_h <= height, \
            '%d <= %d ; %d <= %d ; %d <= %d' % (n_frame, length, scale_w, width, scale_h, height)
        length = int(length)
        if length < n_frame:
            logger.info("%s length %d < %d" % (filename, length, n_frame))

        # random sample frame
        tmp = np.zeros((n_frame, scale_h, scale_w, 3), dtype=np.float32)
        v.set(cv2.CAP_PROP_POS_FRAMES, frame_st)

        for i in xrange(min(n_frame, length)):
            _, f = v.read()
            if f is not None:
                tmp[i, ...] = cv2.resize(f, (scale_w, scale_h))
            else:
                tmp = None
                counter.value = 1
                logging.warning("%s, frame st %d, length %d, n_frame %d" % (filename, frame_st, length, n_frame))
                break

    # tmp is D,H,W,C
    # Temporal transform: looping
    if length < n_frame:
        tmp[-(n_frame - length):] = tmp[:(n_frame - length)]

    tmp = tmp.transpose((3, 0, 1, 2))
    # now tmp is C,D,H,W
    # random flip the video horizontally
    if is_train and flips[p]:
        tmp = np.flip(tmp, 3)

    ret[p, ...] = tmp


def sample_clips(filenames, lengths, frame_idxs, flips, batch_size, n_frame, scale_w=171, scale_h=128, is_train=True):

    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * scale_h * scale_w).reshape(
        (batch_size, 3, n_frame, scale_h, scale_w))
    counter = 0

    process_pool.map(sample_clip_func, [(filenames, lengths, frame_idxs, flips, p, batch_size, n_frame, scale_w,
                                         scale_h, is_train) for p in xrange(len(filenames))])

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