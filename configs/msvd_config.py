from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.gpu_id = 1
__C.num_workers = 4
__C.multi_gpus = False
__C.seed = 666
__C.model_name = 'gan1'
__C.exp_name = 'defaultExp'
__C.server = '2080'

# training options
__C.train = edict()
__C.train.restore = False
__C.train.lr = 0.0001
__C.train.batch_size = 64
__C.train.warm_up_epoch = 5
__C.train.max_epochs = 50
__C.train.train_num = 0 # Default 0 for full train set
__C.train.restore = False
__C.train.glove = True
__C.train.lr_decay = 'cos'
__C.train.loss_weight = 'none'
__C.train = dict(__C.train)
__C.train.flag = True

# model options
__C.model = edict()
__C.model.vision_dim = 2048
__C.model.word_dim = 300
__C.model.module_dim = 512
__C.model.num_head = 8
__C.model.use_box = True
__C.model.use_image = True
__C.model.use_temporal_code = True
__C.model.num_stgcn = 1
__C.model.dropout = 0.1
__C.model.pose_dim = 4
__C.model.pose_mode = 'xyxy'
__C.model.gcn_layer = 2
__C.model = dict(__C.model)

# validation
__C.val = edict()
__C.val.flag = True
__C.val.val_num = 0 # Default 0 for full val set
__C.val = dict(__C.val)

# test
__C.test = edict()
__C.test.flag = True
__C.test.test_num = 0 # Default 0 for full test set
__C.test.write_preds = False
__C.test = dict(__C.test)

# dataset options
__C.dataset = edict()
__C.dataset.name = 'msvd-qa' # ['tgif-qa', 'msrvtt-qa', 'msvd-qa']
__C.dataset.type = 'both' # '2d', '3d', 'both'

__C.dataset.source = 'gnn_new2'
__C.dataset.num_object = 5
__C.dataset.max_video_len = 20
__C.dataset.sampled_frame = 10

__C.dataset.question_type = 'frameqa' #['frameqa', 'count', 'transition', 'action', 'none']
__C.dataset.question_dir = '*/datasets/output/msvd-qa'
__C.dataset.vocab_json = '{}_vocab.json'
__C.dataset.question_pt = '{}_{}_questions.pkl'
__C.dataset.train_question_pt = '{}_train_questions.pkl'
__C.dataset.val_question_pt = '{}_val_questions.pkl'
__C.dataset.test_question_pt = '{}_test_questions.pkl'
__C.dataset.feat_dir='*/MSVD-QA/output'
__C.dataset.feat_pt='{}.pkl'
__C.dataset.save_dir='*/MSVD-QA/exp_logs'
__C.dataset = dict(__C.dataset)

# credit https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py
"""use yaml_cfg to update cfg
"""
def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v

def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_cfg(yaml_cfg, __C)
