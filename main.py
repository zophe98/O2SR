import os, sys
import argparse
import logging
import numpy as np
from termcolor import colored

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from parse_path import add_path
from Utils import Train,Val,Test
from Utils import save_checkpoint,set_all_random_seed,step_decay
from optimizers import GradualWarmupScheduler
from models.submodules.embed_loss import MultipleChoiceLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from data_utils.Dataloader import VideoQADataset,collate_fn

from configs.tgif_config import cfg_from_file as tgif_cfg_from_file
from configs.tgif_config import cfg as tgif_cfg

from configs.msvd_config import cfg as msvd_cfg
from configs.msvd_config import cfg_from_file as msvd_cfg_from_file

from configs.msrvtt_config import cfg as msrvtt_cfg
from configs.msrvtt_config import cfg_from_file as msrvtt_cfg_from_file

from configs.next_config import cfg as next_cfg
from configs.next_config import cfg_from_file as next_cfg_from_file

from models import cond_both
import models


def get_model(model_name):
    return getattr(models, model_name)

def train(cfg):
    logging.info("Create train_loader ,val_loader and test_loader .........")

    train_loader_kwargs = {
        'dataset_name':cfg.dataset.name,
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.train_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'objects_feat': cfg.dataset.object_feat,
        'train_num': cfg.train.train_num,
        'data_source': cfg.dataset.source,
        'feature_type': cfg.dataset.type,  # '2d', '3d', 'both'
        'motion_objects_feat': cfg.dataset.motion_feat_dir,
    }

    videoqa_dataset = VideoQADataset(**train_loader_kwargs)
    train_loader = DataLoader(
        videoqa_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        collate_fn=collate_fn
    )
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))

    if cfg.val.flag:
        val_loader_kwargs = {
            'dataset_name': cfg.dataset.name,
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'objects_feat': cfg.dataset.object_feat,
            'val_num': cfg.val.val_num,
            'data_source': cfg.dataset.source,
            'feature_type': cfg.dataset.type,  # '2d', '3d', 'both'
            'motion_objects_feat': cfg.dataset.motion_feat_dir,
        }

        val_loader = DataLoader(
            VideoQADataset(**val_loader_kwargs),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    if cfg.test.flag:
        test_loader_kwargs = {
            'dataset_name': cfg.dataset.name,
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.test_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'objects_feat': cfg.dataset.object_feat,
            'test_num': cfg.test.test_num,
            'data_source': cfg.dataset.source,
            'feature_type': cfg.dataset.type,  # '2d', '3d', 'both'
            'motion_objects_feat': cfg.dataset.motion_feat_dir,
        }

        test_loader = DataLoader(
            VideoQADataset(**test_loader_kwargs),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))

    logging.info("Create model........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("use device: {}".format(device))

    if cfg.dataset.question_type == 'frameqa':
        num_class = len(videoqa_dataset.vocab['answer_token_to_idx'])
    else:
        num_class = 1

    if cfg.dataset.question_type in ['frameqa', 'count']:
        vocab_size = len(videoqa_dataset.vocab['question_token_to_idx'])
    else:
        vocab_size = len(videoqa_dataset.vocab['question_answer_token_to_idx'])

    if cfg.train.glove:
        logging.info('load glove vectors')
        word_embedding = torch.FloatTensor(videoqa_dataset.glove_matrix).to(device)
    else:
        word_embedding = None

    model_kwargs = {
        'vocab_size': vocab_size,
        'num_classes': num_class,
        'word_embedding_dim': cfg.model.word_dim,
        'vision_dim': cfg.model.vision_dim,
        'module_dim': cfg.model.module_dim,
        'num_head': cfg.model.num_head,
        'dropout': cfg.model.dropout,
        'task': cfg.dataset.question_type,
        'use_box': cfg.model.use_box,
        'use_image': cfg.model.use_image,
        'use_temporal_code': cfg.model.use_temporal_code,
        'word_embedding': None,
        'num_stgcn': cfg.model.num_stgcn,
        'pose_dim': cfg.model.pose_dim,
        'pose_mode': cfg.model.pose_mode,
        'gcn_layer': cfg.model.gcn_layer,
        'num_object': cfg.dataset.num_object,
        'max_video_len': cfg.dataset.max_video_len,
    }

    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    model_kwargs['word_embedding'] = word_embedding

    model = get_model(cfg.model_name).stgcn(**model_kwargs)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('main model num of params: {}'.format(pytorch_total_params))

    model = model.to(device)

    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=None)

    model_paras = [p for k, p in model.named_parameters() ]


    if cfg.train.lr_decay == 'cos':
        optimizer = optim.Adam(model_paras, cfg.train.lr / cfg.train.warm_up_epoch)
    else:
        optimizer = optim.Adam(model_paras, cfg.train.lr)

    start_epoch = 0

    if cfg.dataset.question_type == 'count':
        best_val = 100.0
        best_test = 100.0
        best_val_test = 100.0
    else:
        best_val = 0
        best_test = 0
        best_val_test = 0

    best_epoch = 0
    best_epoch_test = 0
    best_epoch_val_test = 0

    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        model_ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'best_model.pt')
        model_state = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state['model_state'])

        optim_ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'best_optimizer.pt')
        optim_state = torch.load(optim_ckpt,map_location=lambda storage, loc: storage)
        start_epoch = optim_state['epoch'] + 1
        optimizer.load_state_dict(optim_state['optimizer'])

    if cfg.train.lr_decay == 'cos':
        # consine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=cfg.train.warm_up_epoch,
            total_epoch=cfg.train.warm_up_epoch, after_scheduler=scheduler)
    else:
        scheduler = None
        scheduler_warmup = None

    if cfg.dataset.question_type in ['frameqa']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.dataset.question_type == 'count':
        criterion = nn.MSELoss().to(device)
    else:
        # criterion = MultipleChoiceLoss(num_option=5, margin=1, size_average=True).to(device)
        criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, reduction='mean')

    logging.info("Start training........")

    for epoch in range(start_epoch, cfg.train.max_epochs):

        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch + 1), "green", attrs=["bold"])))
        avg_accuracy, avg_loss, avg_mse, caption_loss = \
            Train(cfg,train_loader,epoch,model, device,criterion,
                  optimizer,scheduler_warmup)
        logging.info("Train Epoch = %s  avg_loss = %.3f  avg_acc = %.3f avg_mse = %.3f caption_loss %.3f" %
                     (epoch + 1, avg_loss, avg_accuracy, avg_mse, caption_loss))


        test_acc,test_loss = 0,0
        if cfg.test.flag:
            test_acc, test_loss = Test(cfg, test_loader, model, device, criterion, write_preds=False)

            if (test_acc >= best_test and cfg.dataset.question_type != 'count') or (
                    test_acc <= best_test and cfg.dataset.question_type == 'count'):
                best_test = test_acc
                best_epoch_test = epoch + 1
                # Save best model
                ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)
                save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, ckpt_dir, file_prefix='best_test')
                sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                sys.stdout.flush()

            logging.info('~~~~~~ Test Accuracy: %.4f ~~~~~~~' % test_acc)
            sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
                test_acc=colored("{:.4f}".format(test_acc), "red", attrs=['bold'])))
            sys.stdout.flush()

        if cfg.val.flag:
            output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir)
            val_acc,val_loss = Val(cfg,val_loader,model,device,criterion,write_preds=False)
            if (val_acc >= best_val and cfg.dataset.question_type != 'count') or (
                    val_acc <= best_val and cfg.dataset.question_type == 'count'):
                best_val = val_acc
                best_epoch = epoch + 1

                best_val_test = test_acc
                best_epoch_val_test = epoch + 1

                # Save best model
                ckpt_dir = os.path.join(cfg.dataset.save_dir,'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)
                save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, ckpt_dir,file_prefix='best')
                sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                sys.stdout.flush()
            logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % val_acc)
            sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                valid_acc=colored("{:.4f}".format(val_acc), "red", attrs=['bold'])))
            sys.stdout.flush()

        if cfg.train.lr_decay == 'step' :
            sys.stdout.write("\n")
            if cfg.dataset.question_type == 'count':
                if (epoch + 1) % 5 == 0:
                    optimizer = step_decay(cfg, optimizer)
            else:
                if (epoch + 1) % 10 == 0:
                    optimizer = step_decay(cfg, optimizer)
            sys.stdout.flush()

        logging.info("Epoch {} ,learning rate to {}".format(epoch, optimizer.param_groups[-1]['lr']))

    # Save final model
    logging.info("Train finished......")
    logging.info("Best val epoch = %s  best_val_acc = %.3f" % (best_epoch, best_val))
    logging.info("Best test epoch = %s  best_test_acc = %.3f" % (best_epoch_test, best_test))
    logging.info("Best val_test epoch = %s  best_val_test_acc = %.3f" % (best_epoch_val_test, best_val_test))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='tgif_qa_action.yml', type=str)
    parser.add_argument('--server',
                        type=str)
    parser.add_argument('--model_name',
                        type=str)
    parser.add_argument('--gpus',
                        type=str)
    parser.add_argument('--dataset',
                        type=str,
                        default='tgif-qa')
    parser.add_argument('--exp_prefix',
                        type=str,
                        default='defaultExp')
    args = parser.parse_args()

    if args.dataset == 'tgif-qa':
        cfg = tgif_cfg
        cfg_from_file = tgif_cfg_from_file
    elif args.dataset =='msvd-qa':
        cfg = msvd_cfg
        cfg_from_file = msvd_cfg_from_file
    elif args.dataset == 'msrvtt-qa':
        cfg = msrvtt_cfg
        cfg_from_file = msrvtt_cfg_from_file
    elif args.dataset == 'next-qa':
        cfg = next_cfg
        cfg_from_file = next_cfg_from_file
    else:
        raise NotImplementedError

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.model_name = args.model_name

    # Personal needs, change the data directory according to the server
    args, cfg = add_path(args, cfg)

    if not os.path.exists(cfg.dataset.save_dir):
        os.mkdir(cfg.dataset.save_dir)
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, args.exp_prefix)

    if not os.path.exists(cfg.dataset.save_dir):
        os.mkdir(cfg.dataset.save_dir)

    cfg.gpu_id = args.gpus
    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'next-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']

    # check if the question data folder exists
    assert os.path.exists(cfg.dataset.question_dir), '{} not exists.'.format(cfg.dataset.question_dir)
    # check if the visual data folder exists
    assert os.path.exists(cfg.dataset.feat_dir), '{} not exists.'.format(cfg.dataset.feat_dir)

    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir,cfg.model_name, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)

    log_file = os.path.join(cfg.dataset.save_dir ,'log')
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)
    fileHandler = logging.FileHandler(os.path.join(log_file, 'videoqa.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # concat absolute path of input files
    cfg.dataset.train_question_pt = os.path.join(cfg.dataset.question_dir,
                                                 cfg.dataset.train_question_pt.format(cfg.dataset.question_type))
    cfg.dataset.val_question_pt = os.path.join(cfg.dataset.question_dir,
                                                cfg.dataset.val_question_pt.format(cfg.dataset.question_type))
    cfg.dataset.test_question_pt = os.path.join(cfg.dataset.question_dir,
                                               cfg.dataset.test_question_pt.format(cfg.dataset.question_type))

    cfg.dataset.vocab_json = os.path.join(cfg.dataset.question_dir,
                                        cfg.dataset.vocab_json.format(cfg.dataset.question_type))

    # dir/{video_name}.pkl
    cfg.dataset.object_feat = os.path.join(
        cfg.dataset.feat_dir, cfg.dataset.feat_pt)
    cfg.dataset.motion_feat_dir = os.path.join(
        cfg.dataset.motion_feat_dir, cfg.dataset.feat_pt)

    for k,v in vars(cfg).items():
        logging.info("{}={}\n".format(k,v))


    if cfg.dataset.name not in ['tgif-qa','next-qa']:
        # open-ended question
        assert cfg.dataset.question_type == 'frameqa'

    set_all_random_seed(cfg.seed)

    train(cfg)


if __name__ == '__main__':
    main()