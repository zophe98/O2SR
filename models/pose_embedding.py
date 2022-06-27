import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import pickle as pkl
import os
import numpy as np

def torch_batch_bbox_distance_slow(boxes:torch.Tensor, x1y1x2y2=True):
    # boxes (B,N,4)
    B, N, d = boxes.size()
    distance_matrix = torch.zeros((B, N, N))
    for b in range(B):
        for i in range(N):
            for j in range(N):
                distance_matrix[b][i][j] = torch_bbox_distance(boxes[b, i:i + 1, :],
                                                     boxes[b, j:j + 1, :], x1y1x2y2)
    return distance_matrix
def torch_batch_bbox_distance_fast(boxes:torch.Tensor, x1y1x2y2=True):
    B, N, d = boxes.size()
    boxes_left = boxes.unsqueeze(2)  # (B,N,1,4)
    boxes_left = boxes_left.repeat(1, 1, N, 1)  # (B,N,N,4)
    boxes_left = boxes_left.view(B * N * N, 4)  # (B,N*N,4)

    boxes_right = boxes.unsqueeze(1)  # (B,1,N,4)
    boxes_right = boxes_right.repeat(1, N, 1, 1)  # (B,N,N,4)
    boxes_right = boxes_right.view(B * N * N, 4)  # (B,N*N,4)

    distance_matrix = torch_bbox_distance(boxes_left, boxes_right, x1y1x2y2)
    distance_matrix = distance_matrix.view(B, N, N)

    return distance_matrix

def torch_bbox_distance(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        # (N,1)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    min_distance = (b1_x1 - b2_x1) * (b1_x1 - b2_x1)  + (b1_y1 - b2_y1) * (b1_y1 - b2_y1)

    return min_distance

def torch_batch_bbox_iou_slow(boxes1, x1y1x2y2=True, single_pixel_box=0):
    B, N, d = boxes1.size()
    iou_matrix = torch.zeros((B,N,N))
    for b in range(B):
        for i in range(N):
            for j in range(N):
                iou_matrix[b][i][j] = torch_bbox_iou(boxes1[b,i:i+1,:],
                                                         boxes1[b,j:j+1,:],x1y1x2y2,single_pixel_box)
    return iou_matrix

def torch_batch_bbox_iou_fast(boxes1:torch.Tensor, x1y1x2y2=True, single_pixel_box=0):
    B,N,d = boxes1.size()
    boxes_left = boxes1.unsqueeze(2)            # (B,N,1,4)
    boxes_left = boxes_left.repeat(1,1,N,1)     # (B,N,N,4)
    boxes_left = boxes_left.view(B*N*N, 4)      # (B,N*N,4)

    boxes_right = boxes1.unsqueeze(1)           # (B,1,N,4)
    boxes_right = boxes_right.repeat(1,N,1,1)   # (B,N,N,4)
    boxes_right = boxes_right.view(B*N*N, 4)    # (B,N*N,4)

    iou_matrix = torch_bbox_iou(boxes_left,boxes_right,x1y1x2y2,single_pixel_box)
    iou_matrix = iou_matrix.view(B,N,N)

    return iou_matrix

def torch_bbox_iou(box1, box2, x1y1x2y2=True, single_pixel_box=0):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + single_pixel_box, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + single_pixel_box, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + single_pixel_box) * (b1_y2 - b1_y1 + single_pixel_box)
    b2_area = (b2_x2 - b2_x1 + single_pixel_box) * (b2_y2 - b2_y1 + single_pixel_box)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def torch_extract_position_embedding(position_mat,
                                     feat_dim,
                                     wave_length=1000,
                                     device=torch.device("cuda")):
    # position_mat [B,T,N,N,4]
    # (feat_dim / 8)
    feat_range = torch.arange(0, feat_dim / 8)
    # (feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,))*wave_length,
                        (8. / feat_dim) * feat_range)
    # (1,1,1,1,feat_dim / 8)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    # (B,T,N,N,4) -> (B,T,N,N,4,1)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=-1)
    # (B,T,N,N,4, feat_dim / 8)
    div_mat = torch.div(position_mat.to(device), dim_mat)

    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # (B,T,N,N,4, feat_dim / 4)
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # (B,T,N,N,4 * feat_dim / 4)
    embedding = embedding.view(*(embedding.size()[0:-2]), feat_dim)
    return embedding

def torch_extract_box_embedding(object_box,
                                feat_dim,
                                wave_length=1000,
                                device=torch.device("cuda")):
    # object_box (B,T,N,4)

    # (feat_dim / 8)
    feat_range = torch.arange(0, feat_dim / 8)
    # (feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,))*wave_length,
                        (8. / feat_dim) * feat_range)
    # (1,1,1,feat_dim / 8)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    # (B,T,N,4) -> (B,T,N,4,1)
    object_box = torch.unsqueeze(100.0 * object_box, dim=-1)
    # (B,T,N,4, feat_dim / 8)
    div_mat = torch.div(object_box.to(device), dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # (B,T,N,4, feat_dim / 4)
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # (B,T,N,4 * feat_dim / 4)
    embedding = embedding.view(*(embedding.size()[0:-2]), feat_dim)
    return embedding

# Calculate the relative spatial distance of the object
def make_position_matrix(bbox, mask = None):
    """
    Args:
         bbox (B,T,N,4)
         mask (B,T,N)
    Returns:
        bbox (B,T,N,N,4)
    """
    # xmin (B,T,N,1)
    xmin, ymin, xmax, ymax = torch.split(bbox,1,dim=-1)
    #  (B,T,N,1)
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    # (B,T,N,N)
    delta_x = center_x - torch.transpose(center_x, -1, -2)
    delta_x = torch.div(delta_x, bbox_width)
    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)

    delta_y = center_y - torch.transpose(center_y, -1, -2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)

    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, -1, -2))
    delta_width[delta_width < threshold] = threshold
    delta_width = torch.log(delta_width)

    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, -1, -2))
    delta_height[delta_height < threshold] = threshold
    delta_height = torch.log(delta_height)

    concat_list = [delta_x, delta_y, delta_width, delta_height]
    concat_list = [t.unsqueeze(-1) for t in concat_list]
    # (B,T,N,N,4)
    position_matrix = torch.cat(concat_list, -1)
    if mask is not None:
        # (B,T,N) -> (B,T,1,N,1)
        mask = mask.unsqueeze(-2).unsqueeze(-1)
        # The distance between the two objects is very great
        position_matrix.masked_fill_(mask == 0.0, 1e+3)

    pos_embedding = position_matrix
    return pos_embedding

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, requires_grad=False)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class BoxEmbedding(nn.Module):
    def __init__(self, hidden_dim=-1, output_dim=-1, mode='xyxy', pose_dim = 4):
        assert mode in ['xyxy', 'xywh']
        super(BoxEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode
        if hidden_dim != -1:
            self.bbox_embedding = nn.Sequential(
                nn.Conv2d(pose_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim,output_dim,kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
            )
        else:
            self.bbox_embedding = nn.Sequential(
                nn.Conv2d(pose_dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
            )

    def transfer_box(self, bbox):
        # From xyxy to xywh
        if self.mode == 'xywh':
            return bbox
        bbox[:,:,:,2] = bbox[:,:,:,2] - bbox[:,:,:,0]
        bbox[:,:,:,3] = bbox[:,:,:,3] - bbox[:,:,:,1]

        return bbox
    def forward(self, bbox):
        """
        Args:
            bbox: Object detection box (batchsize, num_frames, num_objects, 4)
        Returns:
            bbox_embdding : (batchsize, num_frames, num_objects, module_dim)
        """
        bbox = self.transfer_box(bbox)
        # (batchsize, num_frames, num_objects, output_dim)
        bbox_embedding = self.bbox_embedding(bbox.permute(0,3,1,2)).permute(0,2,3,1)

        # return bbox_embedding, bbox
        return bbox_embedding

def build_spatial_graph(
        batch_bbox:torch.Tensor,
        criterion = 'distance',
        included_self=False,
        is_softmax=False,
        temperature=1.0,
        device=torch.device("cuda")):
    """
    :param batch_bbox:  (B,N,4)
    :param criterion: distance, iou
    :return:
       (B,N,N)
    """
    B,N,d = batch_bbox.size()
    # assert criterion in ['distance', 'iou']
    if criterion == 'distance':
        weight = torch_batch_bbox_distance_fast(batch_bbox, x1y1x2y2=True)
        weight = (-1.0 * weight)
    elif criterion == 'iou':
        weight = torch_batch_bbox_iou_fast(batch_bbox, x1y1x2y2=True)
    else:
        weight = torch.ones((B,N,N), dtype=batch_bbox.dtype,device=batch_bbox.device)

    weight = temperature * weight   # 温度系数越小, 则差异越小

    if not included_self:
        diagonal = torch.eye(N, dtype=weight.dtype,device=weight.device)
        diagonal = diagonal.view(1,N,N)

        if criterion == 'distance':
            weight = weight.masked_fill_(diagonal == 1.0, -1e+9)
        else:
            weight = weight.masked_fill_(diagonal == 1.0, 0.0)

    if is_softmax:
        graph = F.softmax(weight, dim=-1)
    else:
        graph = weight

    return graph


