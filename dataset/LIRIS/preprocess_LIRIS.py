import os, sys


from pprint import pprint as p
p(sys.path)



import os.path as osp
import cv2
import numpy as np
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import pickle
import pprint
import pandas as pd

# classes = set()
class_names = [
'Discussion between two or more people',
'Give an object to another person',
'Put/take an object into/from a box/desk',
'Enter/leave a room (pass through a door) without unlocking',
'Try to enter a room (unsuccessfully)',
'Unlock and enter (or leave) a room',
'Leave baggage unattended',
'Handshaking',
'Typing on a keyboard',
'Telephone conversation',
]

img_shape = None
max_concurrent_actions = 0

# MISS ANNO /mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation/vid0110.xml 173 27 4


def parseXML(xml_path):
    """
    Args:
        xml_path: str, absolute path to xml file
    Returns:
        a list of action event info, include category, start_fidx, end_fidx, xmin, ymin, xmax, ymax, x1_last, y1_last,
        x2_last, y2_last
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    if len(root[0]) == 1:
        print(xml_path, 'no actions!')
        return None
    actions = root[0][1:]
    print(xml_path, '# actions:', len(actions))
    res = []

    for action in actions:
        category = int(action.get('class')) - 1

        start_fidx = int(action[0].get('framenr')) - 1
        
        if start_fidx == 0:
            start_fidx += 1
            action = action[1:]

        end_fidx = int(action[-1].get('framenr')) - 1
        xmin, ymin, xmax, ymax = None, None, None, None

        bbox_list = []

        action_subidx = 0
        for fidx in range(start_fidx, end_fidx+1):

            cur_framenr = int(action[action_subidx].get('framenr')) - 1
            if cur_framenr != fidx:
                bbox_list.append([-1,-1,-1,-1])
                print('MISS ANNO', xml_path, fidx, start_fidx, category)
                continue

            bbox = action[action_subidx]
            x = int(bbox.get('x'))
            y = int(bbox.get('y'))
            w = int(bbox.get('width'))
            h = int(bbox.get('height'))
            bbox_list.append([x, y, x+w, y+h])
            # TODO: debug by visu
            xmin = x if xmin is None else min(xmin, x)
            ymin = y if ymin is None else min(ymin, y)
            xmax = x + w if xmax is None else max(xmax, x + w)
            ymax = y + h if ymax is None else max(ymax, y + h)

            action_subidx += 1

        try:
            assert(len(bbox_list) == end_fidx - start_fidx + 1)
        except:
            for idx,bbox in enumerate(action):
                print(start_fidx+idx, int(bbox.get('framenr')) - 1)
            print(len(bbox_list), end_fidx - start_fidx + 1, start_fidx, end_fidx)
            sys.exit(1)

        res.append((category, start_fidx, end_fidx, xmin, ymin, xmax, ymax, np.array(bbox_list, dtype=np.int)))
    res = sorted(res, key=lambda x:x[1])
    return res


def process_video_new(video_path, label_list):
    num_frames = max([int(item[4:10]) for item in os.listdir(video_path) if item[0] != '.']) + 1
    bboxes_per_frame = [[] for i in range(num_frames)]

    for aidx, action in enumerate(label_list):
        start_fidx, end_fidx = action[1:3]

        for fidx in range(start_fidx, end_fidx+1):
            bboxes_per_frame[fidx].append( (aidx, action) )

    added_actions = set()
    valid_fidx = []
    new_action_list = []

    left_shift = 0
    for cur_fidx in range(num_frames):
        if len(bboxes_per_frame[cur_fidx]) == 0:
            left_shift += 1
            continue

        valid_fidx.append(cur_fidx)
        for aidx, action in bboxes_per_frame[cur_fidx]:
            if aidx in added_actions:
                continue
            else:
                new_action = [action[0], action[1] - left_shift, action[2] - left_shift, *action[3:]]
                new_action_list.append(new_action)
                added_actions.add(aidx)

    return valid_fidx, new_action_list


def write_video_save_bbox_new(video_path, dst_path, valid_fidx, action_list):
    dst_path_w_bbox = dst_path+'_w_bbox'
    print('dst paths', dst_path, dst_path_w_bbox)
    
    # if not osp.exists(dst_path):
        # os.makedirs(dst_path)
    # if not osp.exists(dst_path_w_bbox):
        # os.makedirs(dst_path_w_bbox)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(dst_path+'.avi', fourcc, 8, (720, 576))
    out2 = cv2.VideoWriter(dst_path_w_bbox+'.avi', fourcc, 8, (720, 576))

    for new_fidx, fidx in enumerate(valid_fidx):
        img_path = osp.join(video_path, 'rgb-{:06d}.jpg'.format(fidx))
        img = cv2.imread(img_path, 1)
        out.write(img)

        for action in action_list:
            if action[1] <= new_fidx and action[2] >= new_fidx:
                class_idx = action[0]
                xmin, ymin, xmax, ymax = action[3:7]
                # xmin_last, ymin_last, xmax_last, ymax_last = action[7:]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
                cur_xmin, cur_ymin, cur_xmax, cur_ymax = action[-1][new_fidx - action[1]]
                if cur_xmin != -1:
                    cv2.rectangle(img, (cur_xmin, cur_ymin), (cur_xmax, cur_ymax), color=(255,0,0), thickness=2)
                cv2.putText(img, class_names[class_idx], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0,255,0), thickness=2)

        out2.write(img)
    
    out.release()
    out2.release()


def plot_lifetime(y_pos, y_labels, widths, lefts, save_path):
    fig, ax = plt.subplots()
    ax.barh(y_pos, width=widths, left=lefts, height=0.1 * len(y_pos))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    ax.set_xlabel('frame index')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def make_plot(label_list, save_path):
    y_pos = np.arange(len(label_list))

    y_labels = []
    widths = []
    lefts = []

    for action in label_list:
        start_fidx, end_fidx = action[1:3]
        widths.append(end_fidx - start_fidx)
        y_labels.append(class_names[action[0]] + ' ' + str(end_fidx-start_fidx))
        lefts.append(start_fidx)

    plot_lifetime(y_pos, y_labels, widths, lefts, save_path)

def visu_single_action(video_path, label_list):
    for aidx, action in enumerate(label_list):

        print('action class:', action[0])
        start_fidx, end_fidx = action[1:3]
        xmin, ymin, xmax, ymax = action[3:]
        for fidx in range(start_fidx, end_fidx+1):
            img_path = osp.join(video_path, 'rgb-{:06d}.jpg'.format(fidx))

            img = cv2.imread(img_path, 1)
            if img is None:
                print('IMG IS NONE!', img_path)
                continue

            global img_shape
            if img_shape is None:
                img_shape == img.shape
            else:
                assert len(img_shape) == 3
                assert img_shape[0] == img.shape[0]
                assert img_shape[1] == img.shape[1]

            # print(img.shape)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
            cv2.putText(img, class_names[action[0]], (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0,255,0), thickness=2)
            cv2.imshow('window', img)

            if cv2.waitKey(1) & 0xFF == ord('w'):
                cv2.destroyAllWindows()
                print('w pressed')
                sys.exit(1)

        # cv2.destroyAllWindows()
        # cv2.waitKey(1)



def visu_whole_video(video_path, label_list):
    global max_concurrent_actions

    num_frames = max([int(item[4:10]) for item in os.listdir(video_path) if item[0] != '.']) + 1
    bboxes_per_frame = [[] for i in range(num_frames)]

    for aidx, action in enumerate(label_list):
        start_fidx, end_fidx = action[1:3]
        xmin, ymin, xmax, ymax = action[3:]

        for fidx in range(start_fidx, end_fidx+1):
            bboxes_per_frame[fidx].append( (action[0], xmin, ymin, xmax, ymax) )


    cur_cnt_list = [len(item) for item in bboxes_per_frame]
    cur_max_concur = max(cur_cnt_list)
    max_frame_idx = cur_cnt_list.index(cur_max_concur)
    print('max_frame_idx', max_frame_idx)
    max_concurrent_actions = max(max_concurrent_actions, cur_max_concur)


    non_empty_frames = [(fidx, l) for fidx, l in enumerate(bboxes_per_frame) if len(l) > 0]

    for fidx, l in non_empty_frames:    
        img_path = osp.join(video_path, 'rgb-{:06d}.jpg'.format(fidx))
        img = cv2.imread(img_path, 1)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            if img is None:
                print('IMG IS NONE', img_path)
                continue
            else:
                print('IMG HAS ZERO SHAPE', img_path)
                continue

        for class_idx, xmin, ymin, xmax, ymax in l:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
            cv2.putText(img, class_names[class_idx], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0,255,0), thickness=2)
        cv2.imshow('window', img)

        if cv2.waitKey(1) & 0xFF == ord('w'):
            cv2.destroyAllWindows()
            print('w pressed')
            sys.exit(1)

def write_video_save_bbox(video_path, dst_path, clip_bbox_list, clip_len=16):
    dst_path_w_bbox = dst_path+'_w_bbox'
    save_list = []

    if not osp.exists(dst_path):
        os.makedirs(dst_path)
    if not osp.exists(dst_path_w_bbox):
        os.makedirs(dst_path_w_bbox)
    

    for idx, clip in enumerate(clip_bbox_list):

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # print('out', osp.join(dst_path, str(idx)+'.avi'))

        out = cv2.VideoWriter(osp.join(dst_path, str(idx)+'.avi'), fourcc, 8, (720, 576))
        out2 = cv2.VideoWriter(osp.join(dst_path_w_bbox, str(idx)+'.avi'), fourcc, 8, (720, 576))

        start_fidx, bbox_list = clip
        save_list.append((osp.join(dst_path, str(idx)+'.avi'), start_fidx, bbox_list))

        for fidx in range(start_fidx, start_fidx+clip_len):
            img_path = osp.join(video_path, 'rgb-{:06d}.jpg'.format(fidx))
            img = cv2.imread(img_path, 1)
            out.write(img)

            for bbox in bbox_list:
                class_idx = bbox[0]
                xmin, ymin, xmax, ymax = bbox[1]
                if xmin != -1:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
                    cv2.putText(img, class_names[class_idx], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0,255,0), thickness=2)

            out2.write(img)

        out.release()
        out2.release()

    with open(dst_path+'.pickle', 'wb') as fd:
        pickle.dump(save_list, fd)

def process_video(video_path, label_list, clip_len=16):
    # label_list = sorted(label_list, key=lambda t:(t[1], t[2]))
    num_frames = max([int(item[4:10]) for item in os.listdir(video_path) if item[0] != '.']) + 1
    bboxes_per_frame = [[] for i in range(num_frames)]

    for aidx, action in enumerate(label_list):
        start_fidx, end_fidx = action[1:3]
        xmin, ymin, xmax, ymax = action[3:]

        for fidx in range(start_fidx, end_fidx+1):
            bboxes_per_frame[fidx].append( action )

    # cur_cnt_list = [len(item) for item in bboxes_per_frame]
    # cur_max_concur = max(cur_cnt_list)
    # max_frame_idx = cur_cnt_list.index(cur_max_concur)
    # print('max_frame_idx', max_frame_idx)
    # max_concurrent_actions = max(max_concurrent_actions, cur_max_concur)
    
    # non_empty_frames = [(fidx, l) for fidx, l in enumerate(bboxes_per_frame) if len(l) > 0]
    # pointer = 0


    clip_bbox_list = [] # (start_fidx, [(class_idx, xmin, ymin, xmax, ymax), ...] -- list of bboxes and action category)

    cur_fidx = 0
    while len(bboxes_per_frame[cur_fidx]) == 0:
        cur_fidx += 1
    
    # cur_fidx = non_empty_frames[0][0]
    while True:
        cur_bbox_list = []
        for action in bboxes_per_frame[cur_fidx]:
            if action[2] + 1 - action[1] < clip_len:
                continue
            else:
                cur_bbox_list.append( (action[0], action[3:7]) )

        if len(cur_bbox_list):
            clip_bbox_list.append( (cur_fidx, cur_bbox_list) )

        cur_fidx += clip_len
        if cur_fidx >= num_frames:
            break

        while cur_fidx < num_frames and len(bboxes_per_frame[cur_fidx]) == 0:
            cur_fidx += 1
        if cur_fidx >= num_frames:
            break

    return clip_bbox_list


if __name__ == '__main__':
    plt.tight_layout()

    root_dir = '/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation'
    dst_dir = '/mnt/truenas/scratch/xiuye/data/LIRIS_processed2/training-validation'
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    # fnames = ['/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation/vid0046']

    # cv2.namedWindow('window')
    # cv2.moveWindow('window', 20, 20)

    fnames = [item for item in os.listdir(root_dir) if not item.endswith('.xml') and 'processed' not in item]

    pprint.pprint(fnames)


    valid_fnames = []
    valid_act_list = []
    valid_video_len = []

    for fname in ['vid0139']:#fnames:
        video_path = osp.join(root_dir, fname)
        res = parseXML(osp.join(root_dir, fname+'.xml'))
        # print('res')
        # print(res)
        if res is not None:
            print(video_path)
            valid_fidx, new_action_list = process_video_new(video_path, res)
            write_video_save_bbox_new(video_path, osp.join(dst_dir, fname+'_processed'), valid_fidx, new_action_list)
            print(len(valid_fidx))

            valid_fnames.append(osp.join(dst_dir, fname+'_processed.avi'))
            valid_act_list.append(new_action_list)
            valid_video_len.append(len(valid_fidx))
            # write_video_save_bbox_new(video_path, dst_path, valid_fidx, action_list)


            # clip_bbox_list = process_video(osp.join(root_dir, fname), res)
            # write_video_save_bbox(osp.join(root_dir, fname), osp.join(dst_dir, fname), clip_bbox_list, clip_len=16)
            # make_plot(res, osp.join('/home/xiuye.gu/video/lifetime_plots/training-validation', fname+'.png'))
            # visu_whole_video(osp.join(root_dir, fname), res)

    df = pd.DataFrame({
        'filename': valid_fnames,
        'actions': valid_act_list,
        'length': valid_video_len
        })

    # df.to_pickle(dst_dir+'-df.pickle', protocol=2)
    



    # print('img_shape', img_shape)
    # print('max_concurrent_actions', max_concurrent_actions)

    # print('all classes', sorted(list(classes)))
    # parseXML('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation/vid0003.xml')
    # parseXML('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation/vid0001.xml')
    # parseXML('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation/vid0159.xml')

