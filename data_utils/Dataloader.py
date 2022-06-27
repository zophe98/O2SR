"""
question + answer
"""
from typing import Tuple, List, Dict, Sequence
import numpy as np
import json
import os
import pickle as pkl
import torch
import math
import glob
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import pprint
import h5py
from .utils import load_vocab, list2tensor

class VideoQADataset(Dataset):
    def __init__(self,**kwargs):
        super(VideoQADataset, self).__init__()
        self.dataset_name = str(kwargs.pop('dataset_name'))

        self.sampled_frame = 16
        # self.sampled_frame = -1

        self.load_dataset(**kwargs)

        # convert data to tensor
        self.questions = torch.LongTensor(np.asarray(self.questions))
        self.questions_len = torch.LongTensor(np.asarray(self.questions_len))
        self.video_ids = torch.LongTensor(np.asarray(self.video_ids))

        if not np.any(self.ans_candidates):
            self.question_type = 'open_ended'
        else:
            self.question_type = 'multichoices'
            self.question_answers = torch.LongTensor(np.asarray(self.question_answers))
            self.question_answers_len = torch.LongTensor(np.asarray(self.question_answers_len))
            self.ans_candidates = torch.LongTensor(np.asarray(self.ans_candidates))
            self.ans_candidates_len = torch.LongTensor(np.asarray(self.ans_candidates_len))

    def load_dataset(self, **kwargs):
        self.feature_type = str(kwargs.pop('feature_type'))

        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        self.vocab = load_vocab(vocab_json_path)

        # ==== Loading the question and answer database ====
        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')

        with open(question_pt_path, 'rb') as f:
            obj = pkl.load(f)
            self.questions = obj['questions']
            self.questions_len = obj['questions_len']
            self.video_ids = obj['video_ids']
            self.video_names = obj['video_names']
            self.q_ids = obj['question_id']
            self.answers = obj['answers']
            self.glove_matrix = obj['glove']

            self.ans_candidates = np.zeros((1,5,1))
            self.ans_candidates_len = np.zeros((1,5))
            self.question_answers = np.zeros((1,5,1))
            self.question_answers_len = np.zeros((1,5))

            if question_type in ['action', 'transition']:
                # (5, max_candidate_len)
                self.ans_candidates = obj['ans_candidates']
                self.ans_candidates_len = obj['ans_candidates_len']

                self.question_answers = obj["question_answes"]
                self.question_answers_len = obj["question_answer_len"]

        # use some sample to debug
        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                self.questions = self.questions[:trained_num]
                self.questions_len = self.questions_len[:trained_num]
                self.video_ids = self.video_ids[:trained_num]
                self.video_names = self.video_names[:trained_num]
                self.q_ids = self.q_ids[:trained_num]
                self.answers = self.answers[:trained_num]
                if question_type in ['action', 'transition']:
                    self.question_answers = self.question_answers[:trained_num]
                    self.question_answers_len = self.question_answers_len[:trained_num]
                    self.ans_candidates = self.ans_candidates[:trained_num]
                    self.ans_candidates_len = self.ans_candidates_len[:trained_num]

        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                self.questions = self.questions[:val_num]
                self.questions_len = self.questions_len[:val_num]
                self.video_ids = self.video_ids[:val_num]
                self.video_names = self.video_names[:val_num]
                self.q_ids = self.q_ids[:val_num]
                self.answers = self.answers[:val_num]
                if question_type in ['action', 'transition']:
                    self.question_answers = self.question_answers[:val_num]
                    self.question_answers_len = self.question_answers_len[:val_num]
                    self.ans_candidates = self.ans_candidates[:val_num]
                    self.ans_candidates_len = self.ans_candidates_len[:val_num]

        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                self.questions = self.questions[:test_num]
                self.questions_len = self.questions_len[:test_num]
                self.video_ids = self.video_ids[:test_num]
                self.video_names = self.video_names[:test_num]
                self.q_ids = self.q_ids[:test_num]
                self.answers = self.answers[:test_num]
                if question_type in ['action', 'transition']:
                    self.question_answers = self.question_answers[:test_num]
                    self.question_answers_len = self.question_answers_len[:test_num]
                    self.ans_candidates = self.ans_candidates[:test_num]
                    self.ans_candidates_len = self.ans_candidates_len[:test_num]

        # ==== Loading the full feature paths ====
        # "feature_data_dir / {video_name}.pkl"
        self.data_source = str(kwargs['data_source'])
        if self.data_source in ['gnn', 'gnn_new2']:
            print('loading appearance feature from %s' % (kwargs['objects_feat']))
            self.object_feature_paths = str(kwargs['objects_feat'])

            self.motion_object_feature_path = str(kwargs.pop('motion_objects_feat'))
            print('loading motion feature from %s' % ( self.motion_object_feature_path ))
        else:
            raise RuntimeError("Please input correct object feat file")

    def get_video_feature_pkl(self, video_name) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor] :
        feature_path = self.object_feature_paths.format(video_name)
        with open(feature_path, 'rb') as f:
            feature_info = pkl.load(f)
            # [ video_len * tensor(args.num_bboxes,4) ]
            bbox = feature_info['bbox']

            # np(video_len, num_bboxes, feature_dim)
            object_feature = feature_info['object_feature']
            # np(video_len, feature_dim)
            glob_feature = feature_info['glob_feature']
            # (video_len,)
            num_object = torch.LongTensor(feature_info['num_object'])

            object_feature = torch.from_numpy(object_feature)
            glob_feature = torch.from_numpy(glob_feature)

            bbox = [torch.Tensor(box).unsqueeze(0) for box in bbox]
            # tensor(video_len, num_bbox, 4)
            bbox = torch.cat(bbox, dim=0)

        return object_feature, glob_feature, bbox, num_object

    def get_video_motion_feature_pkl(self, video_name) -> Tuple[torch.Tensor,torch.Tensor] :

        feature_path = self.motion_object_feature_path.format(video_name)
        with open(feature_path, 'rb') as f:
            feature_info = pkl.load(f)
            # np(video_len, num_bboxes, feature_dim)
            object_feature = feature_info['object_feature']
            # np(video_len, feature_dim)
            glob_feature = feature_info['glob_feature']
            object_feature = torch.from_numpy(object_feature)
            glob_feature = torch.from_numpy(glob_feature)

        return object_feature, glob_feature

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        answer = self.answers[index] if self.answers is not None else None

        question = self.questions[index]
        question_len = self.questions_len[index]

        if self.question_type == 'multichoices':
            ans_candidates = self.ans_candidates[index]
            ans_candidates_len = self.ans_candidates_len[index]

            # TODO: 使用问题 + 候选项 进行训练
            question_candidates, question_candidates_len = self.question_answers[index], self.question_answers_len[index]

            # TODO: 只是由候选项进行训练, 用于衡量 visual-answer 的 数据集bias
            # question_candidates, question_candidates_len = ans_candidates, ans_candidates_len
        else:
            question_candidates, question_candidates_len = question, question_len

        video_idx = self.video_ids[index].item()
        question_idx = self.q_ids[index]
        video_name = self.video_names[index]

        motion_object_feature, motion_glob_feature = None, None

        object_feature, glob_feature, bbox, num_object = self.get_video_feature_pkl(video_name)

        if self.feature_type == '3d':
            object_feature, glob_feature = self.get_video_motion_feature_pkl(video_name)
        elif self.feature_type == 'both':
            try:
                motion_object_feature, motion_glob_feature = self.get_video_motion_feature_pkl(video_name)
            except Exception as e:
                raise RuntimeError('file {} error'.format(video_name))


        item_info = {
            'video_idx': video_idx,  # 0
            'question_idx' : question_idx,
            'video_name' : video_name,
            'answer': answer,  # int(4)
            'question_candidates': question_candidates,  # tensor (5, q_a_length)
            'question_candidates_len': question_candidates_len,  # tensor(5,)
            'object_feat': object_feature,  # tensor(video_len, num_obj, d)
            'num_object': num_object,  # tensor(video_len)
            'glob_feat': glob_feature,  # tensor(video_len, d)
            'motion_object_feat': motion_object_feature,  # tensor(video_len, num_obj, d)
            'motion_glob_feat': motion_glob_feature,  # tensor(video_len, d)
            'bbox': bbox,  # tensor(video_len, num_obj, 4)
            'PAD': self.vocab['question_token_to_idx']['PAD']
        }

        return item_info

# Convert the list data of dict() into a batch data
def collate_fn(batch_data):
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # (B,L) -> (B,L,1)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask

    item_info = {}
    """
    An item returned by Dataset is a dictionary. 
    A batch here first returns a list of dictionaries. 
    We need to reassemble it and send it to the GPU.
    """
    pad_idx = batch_data[0]['PAD']
    for key in batch_data[0].keys():
        item_info[key] = [data[key] for data in batch_data]

    all_question = list2tensor(item_info['question_candidates'])
    all_question_len = list2tensor(item_info['question_candidates_len'])
    question_mask = (all_question != pad_idx)


    # list -> tensor
    all_answers = torch.from_numpy(np.asarray(item_info['answer']))
    all_obj_fea = item_info['object_feat']
    all_num_obj = item_info['num_object']
    all_glob_fea = item_info['glob_feat']
    all_bbox = item_info['bbox']

    batchsize = len(all_obj_fea)

    fea_dim = all_obj_fea[0].size(-1)
    max_video_len = max([box.size(0) for box in all_bbox])
    max_num_obj = max([box.size(1) for box in all_bbox])
    padded_obj_features = torch.zeros((batchsize, max_video_len, max_num_obj, fea_dim),
                                      dtype=torch.float32)
    padded_glob_features = torch.zeros((batchsize, max_video_len, fea_dim),
                                       dtype=torch.float32)
    obj_mask = torch.zeros((batchsize, max_video_len, max_num_obj), dtype=torch.float32)

    for i, obj_fea in enumerate(all_obj_fea):
        video_len, num_obj, _ = obj_fea.size()
        padded_obj_features[i,:video_len, :num_obj, :] = obj_fea
        padded_glob_features[i,:video_len,:] = all_glob_fea[i]
        for j in range(video_len):
            obj_mask[i,j,:all_num_obj[i][j]] = 1.0

    all_motion_object_feat = item_info['motion_object_feat']            # (B,T,N,d)
    all_motion_glob_feature = item_info['motion_glob_feat']          # (B,T,d)
    if all_motion_glob_feature[0] is None:
        padded_all_motion_object_feat = torch.zeros(batchsize,1,1,fea_dim)
        padded_all_motion_glob_object_feat = torch.zeros(batchsize,1,fea_dim)
    else:
        padded_all_motion_object_feat = torch.zeros((batchsize, max_video_len, max_num_obj, fea_dim),
                                          dtype=torch.float32)
        padded_all_motion_glob_object_feat = torch.zeros((batchsize, max_video_len, fea_dim),
                                           dtype=torch.float32)
        for i, obj_fea in enumerate(all_motion_object_feat):
            video_len, num_obj, _ = obj_fea.size()
            padded_all_motion_object_feat[i, :video_len, :num_obj, :] = obj_fea
            padded_all_motion_glob_object_feat[i, :video_len, :] = all_motion_glob_feature[i]

    # 填充坐标特征
    padded_bbox = torch.zeros((batchsize, max_video_len, max_num_obj, all_bbox[0].size(-1)),
                              dtype=torch.float32)
    for i, bbox in enumerate(all_bbox):
        video_len, num_obj, _ = bbox.size()
        padded_bbox[i, :video_len, :num_obj, :] = bbox
    """
                                            Multi-Choice    Open-ended
    padded_glob_features                    (B,T,d)         (B,T,d)
    padded_obj_features                     (B,T,N,d)       (B,T,N,d)
    padded_all_motion_glob_object_feat      (B,T,d)         (B,T,d)
    padded_all_motion_object_feat           (B,T,N,d)       (B,T,N,d)
    padded_bbox                             (B,T,N,4/6)     (B,T,N,4/6)
    obj_mask                                (B,T,N)         (B,T,N)
    all_question                            (B,5,L)         (B,L)
    all_question_len                        (B,5)           (B,)
    question_mask                           (B,5,L)         (B,L)
    all_answers                             (B,)            (B,)
    """
    return padded_glob_features, padded_obj_features,padded_all_motion_glob_object_feat, padded_all_motion_object_feat, \
            padded_bbox,obj_mask,all_question,all_question_len,question_mask, all_answers


def test_collate_fn(batch_data):
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # (B,L) -> (B,L,1)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask

    item_info = {}

    pad_idx = batch_data[0]['PAD']
    for key in batch_data[0].keys():
        item_info[key] = [data[key] for data in batch_data]

    # (batchsize, max_quesiton_len), 进行填充
    all_question = list2tensor(item_info['question_candidates'])
    # (batchsize,)
    all_question_len = list2tensor(item_info['question_candidates_len'])
    question_mask = (all_question != pad_idx)


    question_idxs = item_info['question_idx']
    video_idxs = item_info['video_idx']
    video_names = item_info['video_name']

    # list -> tensor
    all_answers = torch.from_numpy(np.asarray(item_info['answer']))
    all_obj_fea = item_info['object_feat']
    all_num_obj = item_info['num_object']
    all_glob_fea = item_info['glob_feat']
    all_bbox = item_info['bbox']

    # 我们需要填充视频特征 object_feature (batchsize, max_video_len, max_obj_len, d)
    batchsize = len(all_obj_fea)

    fea_dim = all_obj_fea[0].size(-1)
    max_video_len = max([box.size(0) for box in all_bbox])
    max_num_obj = max([box.size(1) for box in all_bbox])
    padded_obj_features = torch.zeros((batchsize, max_video_len, max_num_obj, fea_dim),
                                      dtype=torch.float32)
    padded_glob_features = torch.zeros((batchsize, max_video_len, fea_dim),
                                       dtype=torch.float32)
    obj_mask = torch.zeros((batchsize, max_video_len, max_num_obj), dtype=torch.float32)

    for i, obj_fea in enumerate(all_obj_fea):
        video_len, num_obj, _ = obj_fea.size()
        padded_obj_features[i,:video_len, :num_obj, :] = obj_fea
        padded_glob_features[i,:video_len,:] = all_glob_fea[i]
        for j in range(video_len):
            obj_mask[i,j,:all_num_obj[i][j]] = 1.0

    # 填充运动特征, 但是需要看是不是为 None
    all_motion_object_feat = item_info['motion_object_feat']            # (B,T,N,d)
    all_motion_glob_feature = item_info['motion_glob_feat']          # (B,T,d)
    if all_motion_glob_feature[0] is None:
        padded_all_motion_object_feat = torch.zeros(batchsize,1,1,fea_dim)
        padded_all_motion_glob_object_feat = torch.zeros(batchsize,1,fea_dim)
    else:
        # 进行填充
        padded_all_motion_object_feat = torch.zeros((batchsize, max_video_len, max_num_obj, fea_dim),
                                          dtype=torch.float32)
        padded_all_motion_glob_object_feat = torch.zeros((batchsize, max_video_len, fea_dim),
                                           dtype=torch.float32)
        for i, obj_fea in enumerate(all_motion_object_feat):
            video_len, num_obj, _ = obj_fea.size()
            padded_all_motion_object_feat[i, :video_len, :num_obj, :] = obj_fea
            padded_all_motion_glob_object_feat[i, :video_len, :] = all_motion_glob_feature[i]

    padded_bbox = torch.zeros((batchsize, max_video_len, max_num_obj, all_bbox[0].size(-1)),
                              dtype=torch.float32)
    for i, bbox in enumerate(all_bbox):
        video_len, num_obj, _ = bbox.size()
        padded_bbox[i, :video_len, :num_obj, :] = bbox
    """
                                            Multi-Choice     Open-ended
    padded_glob_features                    (B,T,d)         (B,T,d)
    padded_obj_features                     (B,T,N,d)       (B,T,N,d)
    padded_all_motion_glob_object_feat      (B,T,d)         (B,T,d)
    padded_all_motion_object_feat           (B,T,N,d)       (B,T,N,d)
    padded_bbox                             (B,T,N,4/6)     (B,T,N,4/6)
    obj_mask                                (B,T,N)         (B,T,N) 
    all_question                            (B,5,L)         (B,L)
    all_question_len                        (B,5)           (B,)
    question_mask                           (B,5,L)         (B,L)
    all_answers                             (B,)            (B,)                             1                1
    """
    return padded_glob_features, padded_obj_features,padded_all_motion_glob_object_feat, padded_all_motion_object_feat, \
            padded_bbox,obj_mask,all_question,all_question_len,question_mask, all_answers,  \
           question_idxs, video_idxs, video_names


