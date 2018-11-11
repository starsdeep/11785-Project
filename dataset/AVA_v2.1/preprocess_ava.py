import pandas as pd
import numpy as np

def find_nn_bboxes(bboxes):
    if bboxes.shape[0] == 1:
        return np.array([[np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    elif bboxes.shape[0] == 2:
        return bboxes[[1,0], :]
    
    centers = np.empty((bboxes.shape[0], 2))
    centers[:, 0] = bboxes[:, 0] + bboxes[:, 2]
    centers[:, 1] = bboxes[:, 1] + bboxes[:, 3]
    
    result = np.empty_like(bboxes)
    
    for i in range(bboxes.shape[0]):
        min_idx = None
        min_dist = None
        for j in range(bboxes.shape[0]):            
            if i == j:
                continue
            dist = np.linalg.norm(centers[i, :] - centers[j, :])
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_idx = j
        result[i, :] = bboxes[min_idx, :]
    return result

def process_csv(fname, dst_name):
    df = pd.read_csv(fname, sep=',', names=['video_id', 'timestamp', 'xmin', 'ymin', 'xmax', 'ymax', 'action_id', 'person_id'])
    sorted_df = df.sort_values(by = ['video_id', 'timestamp', 'person_id', 'action_id'])
    grouped_id_time = sorted_df.groupby(['video_id', 'timestamp'])

    video_ids = []
    timestamps = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    nn_xmin = []
    nn_ymin = []
    nn_xmax = []
    nn_ymax = []
    action_id_lists = []
    person_ids = []
    
    for name, group in grouped_id_time:
    #     print('ID: ', name)
    #     display(group)
        
        bboxes_list = []
        inner_group = group.groupby(['person_id'])
        
        for inner_name, inner_inner_group in inner_group:
            video_ids.append(name[0])
            timestamps.append(name[1])
            person_ids.append(inner_name)
            action_id_lists.append(inner_inner_group.action_id.tolist())
            
            
            bboxes_list.append([inner_inner_group.xmin.values[0], 
                              inner_inner_group.ymin.values[0], 
                              inner_inner_group.xmax.values[0], 
                              inner_inner_group.ymax.values[0]])
            
        bboxes_np = np.array(bboxes_list)
        nn_bboxes = find_nn_bboxes(bboxes_np)
        
        xmin.extend(bboxes_np[:, 0].tolist())
        ymin.extend(bboxes_np[:, 1].tolist())
        xmax.extend(bboxes_np[:, 2].tolist())
        ymax.extend(bboxes_np[:, 3].tolist())
        
        nn_xmin.extend(nn_bboxes[:, 0].tolist())
        nn_ymin.extend(nn_bboxes[:, 1].tolist())
        nn_xmax.extend(nn_bboxes[:, 2].tolist())
        nn_ymax.extend(nn_bboxes[:, 3].tolist())
        
    #     if cnt == 30:
    #         break
    #     cnt += 1
        
    # print(len(video_ids))
    # print(len(timestamps))
    # print(len(xmin))
    # print(len(ymin))
    # print(len(xmax))
    # print(len(ymax))
    # print(len(nn_xmin))
    # print(len(nn_ymin))
    # print(len(nn_xmax))
    # print(len(nn_ymax))
    # print(len(xmin))
    processed_df = pd.DataFrame({
        'video_id':video_ids,
        'timestamp':timestamps,
        'person_id':person_ids,
        'action_id':action_id_lists,
        'xmin':xmin,
        'ymin':ymin,
        'xmax':xmax,
        'ymax':ymax,
        'nn_xmin':nn_xmin,
        'nn_ymin':nn_ymin,
        'nn_xmax':nn_xmax,
        'nn_ymax':nn_ymax
    })

    processed_df.to_pickle(dst_name, protocol=2)

#process_csv('/mnt/truenas/scratch/yikangliao/dataset/AVA/ava_val_v2.1.csv', 'processed_ava_val.pickle')
#process_csv('/mnt/truenas/scratch/yikangliao/dataset/AVA/ava_train_v2.1.csv', 'processed_ava_train.pickle')

def get_person_interaction_lst(row):
    action_lst = row['action_id']
    return [action-63 for action in action_lst if action>=64]


# padding nearest neighbor bbox with the whole image
df_train = pd.read_pickle('/home/yikang.liao/workspace/cvpr19/dataset/AVA_v2.1/processed_ava_train.pickle')
df_val = pd.read_pickle('/home/yikang.liao/workspace/cvpr19/dataset/AVA_v2.1/processed_ava_val.pickle')
df_train['person_interaction_actions'] = df_train.apply(lambda row: get_person_interaction_lst(row), axis=1)
df_val['person_interaction_actions'] = df_val.apply(lambda row: get_person_interaction_lst(row), axis=1)
idx_train_person_interaction = df_train.apply(lambda row: len(row['person_interaction_actions'])>0, axis=1)
idx_val_person_interaction = df_val.apply(lambda row: len(row['person_interaction_actions'])>0, axis=1)


idx_train = df_train.nn_xmax.isnull()
idx_val = df_val.nn_xmax.isnull()

df_train['nn_xmax'][idx_train] = 1.0
df_train['nn_xmin'][idx_train] = 0.0
df_train['nn_ymax'][idx_train] = 1.0
df_train['nn_ymin'][idx_train] = 0.0

df_val['nn_xmax'][idx_val] = 1.0
df_val['nn_xmin'][idx_val] = 0.0
df_val['nn_ymax'][idx_val] = 1.0
df_val['nn_ymin'][idx_val] = 0.0

print(df_train.head(50))
df_train.to_pickle('/home/yikang.liao/share/dataset/AVA/df_train_pad_background.pickle')
df_val.to_pickle('/home/yikang.liao/share/dataset/AVA/df_val_pad_background.pickle')
# person interaction
df_train_person_interaction = df_train[idx_train_person_interaction]
df_val_person_interaction = df_val[idx_val_person_interaction]
df_train_person_interaction.drop(columns=['action_id',], inplace=True)
df_val_person_interaction.drop(columns=['action_id',], inplace=True)
df_train_person_interaction.rename(columns={'person_interaction_actions': 'action_id'}, inplace=True)
df_val_person_interaction.rename(columns={'person_interaction_actions': 'action_id'}, inplace=True)
print("train %d, val %d, train person %d, val person %d" % (len(df_train), len(df_val), len(df_train_person_interaction), len(df_val_person_interaction)))
df_train_person_interaction.to_pickle('/home/yikang.liao/share/dataset/AVA/df_train_pad_background_person_interaction.pickle')
df_val_person_interaction.to_pickle('/home/yikang.liao/share/dataset/AVA/df_val_pad_background_person_interaction.pickle')



# padding nearest neighbor bbox with the bboxe itself
df_train['nn_xmax'][idx_train] = df_train['xmax'][idx_train]
df_train['nn_xmin'][idx_train] = df_train['xmin'][idx_train]
df_train['nn_ymax'][idx_train] = df_train['ymax'][idx_train]
df_train['nn_ymin'][idx_train] = df_train['ymin'][idx_train]

df_val['nn_xmax'][idx_val] = df_val['xmax'][idx_val]
df_val['nn_xmin'][idx_val] = df_val['xmin'][idx_val]
df_val['nn_ymax'][idx_val] = df_val['ymax'][idx_val]
df_val['nn_ymin'][idx_val] = df_val['ymin'][idx_val]

print(df_train.head(50))
df_train.to_pickle('/home/yikang.liao/share/dataset/AVA/df_train_pad_repeat.pickle')
df_val.to_pickle('/home/yikang.liao/share/dataset/AVA/df_val_pad_repeat.pickle')
# person interaction
df_train_person_interaction = df_train[idx_train_person_interaction]
df_val_person_interaction = df_val[idx_val_person_interaction]
df_train_person_interaction.drop(columns=['action_id',], inplace=True)
df_val_person_interaction.drop(columns=['action_id',], inplace=True)
df_train_person_interaction.rename(columns={'person_interaction_actions': 'action_id'}, inplace=True)
df_val_person_interaction.rename(columns={'person_interaction_actions': 'action_id'}, inplace=True)
print("train %d, val %d, train person %d, val person %d" % (len(df_train), len(df_val), len(df_train_person_interaction), len(df_val_person_interaction)))
df_train_person_interaction.to_pickle('/home/yikang.liao/share/dataset/AVA/df_train_pad_repeat_person_interaction.pickle')
df_val_person_interaction.to_pickle('/home/yikang.liao/share/dataset/AVA/df_val_pad_repeat_person_interaction.pickle')








