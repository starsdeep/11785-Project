import numpy as np
import mxnet as mx
import math
import logging

logger = logging.getLogger(__name__)


def relation_feature(body, rois, batch_per_device, nongt_dim, geometric_dim):
    unique_rois = mx.sym.slice_axis(rois, axis=1, begin=0, end=1)

    reshaped_rois = mx.sym.reshape(data=unique_rois, shape=(-1, 5))
    sliced_rois = mx.sym.slice_axis(data=reshaped_rois, axis=1, begin=1, end=None)
    rois_lst = mx.sym.split(data=sliced_rois, axis=0, num_outputs=batch_per_device)
    position_matrix_lst = [extract_position_matrix(rois_lst[i], nongt_dim=nongt_dim) for i in range(batch_per_device)]
    position_embedding_lst = [extract_position_embedding(position_matrix_lst[i], feat_dim=geometric_dim)
                              for i in range(batch_per_device)]

    body_lst = mx.sym.split(data=body, axis=0, num_outputs=batch_per_device)
    attention_lst = [attention_module_multi_head(body_lst[i], position_embedding_lst[i], nongt_dim=nongt_dim, fc_dim=8,
                                                 feat_dim=512, index=i, group=8, dim=(512, 512, 512))
                     for i in range(batch_per_device)]
    body = mx.sym.concat(*attention_lst, dim=0, name="relation")
    return body


def extract_position_embedding(position_mat, feat_dim, wave_length=1000, index=1):
    # position_mat, [num_rois, nongt_dim, 4]
    feat_range = mx.sym.arange(0, feat_dim / 8)
    dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                     rhs=(8. / feat_dim) * feat_range)
    dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
    position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
    div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
    sin_mat = mx.sym.sin(data=div_mat)
    cos_mat = mx.sym.cos(data=div_mat)
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
    return embedding

def extract_position_matrix(bbox, nongt_dim):
    """ Extract position matrix

    Args:
        bbox: [num_boxes, 4]

    Returns:
        position_matrix: [num_boxes, nongt_dim, 4]
    """
    xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                          num_outputs=4, axis=1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                     rhs=mx.sym.transpose(center_x))
    delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
    delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
    delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                     rhs=mx.sym.transpose(center_y))
    delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
    delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
    delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                       rhs=mx.sym.transpose(bbox_width))
    delta_width = mx.sym.log(delta_width)
    delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                        rhs=mx.sym.transpose(bbox_height))
    delta_height = mx.sym.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim)
        concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
    position_matrix = mx.sym.concat(*concat_list, dim=2)
    return position_matrix

def attention_module_multi_head(roi_feat, position_embedding,
                                nongt_dim, fc_dim, feat_dim,
                                dim=(1024, 1024, 1024),
                                group=16, index=1):
    """ Attetion module with vectorized version

    Args:
        roi_feat: [num_rois, feat_dim]
        position_embedding: [num_rois, nongt_dim, emb_dim]
        nongt_dim:
        fc_dim: should be same as group
        feat_dim: dimension of roi_feat, should be same as dim[2]
        dim: a 3-tuple of (query, key, output)
        group:
        index:

    Returns:
        output: [num_rois, ovr_feat_dim, output_dim]
    """
    dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
    nongt_roi_feat = mx.symbol.slice_axis(data=roi_feat, axis=0, begin=0, end=nongt_dim)
    # [num_rois * nongt_dim, emb_dim]
    position_embedding_reshape = mx.sym.Reshape(position_embedding, shape=(-3, -2))
    # position_feat_1, [num_rois * nongt_dim, fc_dim]
    position_feat_1 = mx.sym.FullyConnected(name='pair_pos_fc1_' + str(index),
                                            data=position_embedding_reshape,
                                            num_hidden=fc_dim)
    position_feat_1_relu = mx.sym.Activation(data=position_feat_1, act_type='relu')
    # aff_weight, [num_rois, nongt_dim, fc_dim]
    aff_weight = mx.sym.Reshape(position_feat_1_relu, shape=(-1, nongt_dim, fc_dim))
    # aff_weight, [num_rois, fc_dim, nongt_dim]
    aff_weight = mx.sym.transpose(aff_weight, axes=(0, 2, 1))

    # multi head
    assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
    q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                   data=roi_feat,
                                   num_hidden=dim[0])
    q_data_batch = mx.sym.Reshape(q_data, shape=(-1, group, dim_group[0]))
    q_data_batch = mx.sym.transpose(q_data_batch, axes=(1, 0, 2))
    k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                      data=nongt_roi_feat,
                                      num_hidden=dim[1])
    k_data_batch = mx.sym.Reshape(k_data, shape=(-1, group, dim_group[1]))
    k_data_batch = mx.sym.transpose(k_data_batch, axes=(1, 0, 2))
    v_data = nongt_roi_feat
    # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
    aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
    # aff_scale, [group, num_rois, nongt_dim]
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
    aff_scale = mx.sym.transpose(aff_scale, axes=(1, 0, 2))

    assert fc_dim == group, 'fc_dim != group'
    # weighted_aff, [num_rois, fc_dim, nongt_dim]
    weighted_aff = mx.sym.log(mx.sym.maximum(left=aff_weight, right=1e-6)) + aff_scale
    aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='softmax_' + str(index))
    # [num_rois * fc_dim, nongt_dim]
    aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-3, -2))
    # output_t, [num_rois * fc_dim, feat_dim]
    output_t = mx.symbol.dot(lhs=aff_softmax_reshape, rhs=v_data)
    # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
    output_t = mx.sym.Reshape(output_t, shape=(-1, fc_dim * feat_dim, 1, 1))
    # linear_out, [num_rois, dim[2], 1, 1]
    linear_out = mx.symbol.Convolution(name='linear_out_' + str(index), data=output_t,
                                       kernel=(1, 1), num_filter=dim[2], num_group=fc_dim)
    output = mx.sym.Reshape(linear_out, shape=(0, 0))
    return output


