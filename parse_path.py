import os, sys
import argparse
from easydict import EasyDict as edict


def add_path(args:argparse.Namespace, cfg:edict):
    if args.server == '2082':
        if cfg.dataset.name == 'tgif-qa':
            cfg.dataset.question_dir = ''
            cfg.dataset.feat_dir = ''
            cfg.dataset.motion_feat_dir = ''
            cfg.dataset.save_dir = ''
        elif cfg.dataset.name == 'next-qa':
            cfg.dataset.question_dir = ''
            cfg.dataset.feat_dir = ''
            cfg.dataset.motion_feat_dir = ''
            cfg.dataset.save_dir = ''
        else:
            raise NotImplementedError(cfg.dataset.name)
    else:
        raise NotImplementedError('Not implement {}'.format(args.server))

    return args, cfg