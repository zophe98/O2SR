import torch
import random
import numpy as np
import time
from torch.nn import init
import os, sys
import torch
import torch.nn as nn
import logging
from termcolor import colored
from tqdm import tqdm

class nvidia_prefetcher():
    def __init__(self, loader):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = [d.cuda(non_blocking=True) for d in self.next_data]
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def __len__(self):
        return self.length

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is None:
            raise StopIteration
        next_data = self.next_data
        self.preload()
        return next_data

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def Train(cfg, loader, epoch, model,
            device,loss_compute,
            optimizer,scheduler_warmup=None):

    print("Train......")
    model.train()

    total_acc, total_qa_loss,total_count = 0,0.0, 0
    total_tokens = 1
    total_caption_loss = 0.0
    batch_mse_sum = 0.0
    avg_loss = 0.0


    if scheduler_warmup is not None:
        scheduler_warmup.step(epoch=epoch)
    for i, batch in enumerate(tqdm(loader)):
        if i == 0:
            data_type = [(d.dtype, d.shape) for d in batch]
            print(data_type)

        loss = 0.0
        batch = todevice(batch, device)
        optimizer.zero_grad()

        # {'caption_in', 'answer'}
        # answers (B,)
        output, answers = model(*batch)

        # (B,5,1)  or (B,C)
        logits = output['answer']
        if cfg.dataset.question_type in ['action', 'transition','frameqa']:
            # output['answer]       torch.Size([32, 5, 1])
            # output['caption_in'] torch.Size([32, 35, 512])
            # (B,5) or (B, C)
            logits = torch.squeeze(logits)
            qa_loss = loss_compute(logits, answers)

            # ==== Compute Loss ====
            loss += qa_loss
            total_qa_loss += qa_loss.detach()
            avg_loss = total_qa_loss / (i + 1)

            # ==== Compute Accuracy ====
            predictions = torch.argmax(logits, dim=-1)  # (B, )
            aggreeings = (predictions == answers)  # (B, )
            total_acc += sum(aggreeings).item()
            total_count += answers.size(0)

        elif cfg.dataset.question_type in ['count']:
            # (B,1)
            answers = answers.unsqueeze(-1)

            qa_loss = loss_compute(logits, answers.float())
            loss += qa_loss
            total_qa_loss += qa_loss.detach()
            avg_loss = total_qa_loss / (i + 1)

            # ==== Compute MSE loss ====
            preds = (logits + 0.5).long().clamp(min=1, max=10)
            batch_mse = (preds - answers) ** 2
            batch_mse_sum += batch_mse.sum().item()
            total_count += answers.size(0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
        optimizer.step()

    sys.stdout.write("\n")
    sys.stdout.flush()

    return total_acc / total_count , avg_loss, batch_mse_sum / total_count, total_caption_loss / total_tokens

@torch.no_grad()
def Val(cfg, loader, model, device, loss_compute = None, write_preds = False, test = False):
    model.eval()
    if test:
        print("testing......")
    else:
        print("validating......")

    total_acc, total_count = 0.0, 0
    total_loss = 0.0
    all_preds = []
    gts = []
    all_questions = []

    question_idxs = []
    video_idxs = []
    video_names = []

    appr_attens = []
    motion_attens = []

    appr_obj_crosses = []
    motion_obj_crosses = []

    appr_frame_crosses = []
    motion_frame_crosses = []

    for batch in tqdm(loader):
        questions = []
        if len(batch) >= 17 and write_preds:
            question,question_idx, video_idx, video_name = list(batch[6]),list(batch[15]), list(batch[16]), list(batch[17])
            question_idxs.extend(question_idx)
            questions.extend(question)
            video_idxs.extend(video_idx)
            video_names.extend(video_name)

        batch = todevice(batch,device)

        # answer (B,)
        out,answers = model(*batch)
        if 'object_cross' in out:
            appr_obj_crosses.append(out['object_cross'][0].detach().cpu())
            motion_obj_crosses.append(out['object_cross'][1].detach().cpu())

        if 'frame_cross' in out:
            appr_frame_crosses.append(out['frame_cross'][0].detach().cpu())
            motion_frame_crosses.append(out['frame_cross'][1].detach().cpu())

        if 'temporal_atten' in out:
            appr_attens.append(out['temporal_atten'][0].detach().cpu())
            motion_attens.append(out['temporal_atten'][1].detach().cpu())

        # (B,C) or (B,5,C)
        logits = out['answer']
        batch_size = answers.size(0)
        total_count += batch_size

        if cfg.dataset.question_type == 'count':
            answers = answers.unsqueeze(-1)
            preds = (logits + 0.5).long().clamp(min=1, max=10)
            batch_mse = (preds - answers) ** 2
            total_acc += batch_mse.float().sum().item()
            if loss_compute is not None:
                loss = loss_compute(logits, answers.float())
                total_loss += loss.detach()
        else:
            # (B,5,1) -> (B,5)
            # (B,C) -> (B,C)
            logits = torch.squeeze(logits)
            if loss_compute is not None:
                loss = loss_compute(logits, answers)
                total_loss += loss.detach()
            preds = logits.detach().argmax(1)
            agreeings = (preds == answers)
            total_acc += (agreeings).float().sum().item()

        if write_preds:
            if cfg.dataset.question_type == 'count':
                # (B,)
                answers = answers
                # (B,)
                preds = torch.squeeze((logits + 0.5).long().clamp(min=1, max=10))
                question_answer_vocab = loader.dataset.vocab['question_idx_to_token']
            else:
                # (B,)
                answers = answers
                # (B,)
                preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition']:
                    question_answer_vocab = loader.dataset.vocab['question_answer_idx_to_token']
                else:
                    # question_answer_vocab = loader.dataset.vocab['question_answer_idx_to_token']
                    question_answer_vocab = loader.dataset.vocab['question_idx_to_token']
                    answer_vocab = loader.dataset.vocab['answer_idx_to_token']

            for predict in preds:
                if cfg.dataset.question_type in ['count', 'transition', 'action']:
                    all_preds.append(predict.item())
                else:
                    all_preds.append(answer_vocab[predict.item()])

            for gt in answers:
                if cfg.dataset.question_type in ['count', 'transition', 'action']:
                    gts.append(gt.item())
                else:
                    gts.append(answer_vocab[gt.item()])

            if cfg.dataset.question_type not in ['frameqa', 'count']:
                # 筛选出正确答案的 question+candidates
                # shape of questions (B,5,L)
                questions = torch.stack(questions, 0).transpose(0, 1).detach().cpu()  # (5,B,L)
                sample_index = answers.view(1, questions.size(1), 1).repeat(1, 1, questions.size(-1)).detach().cpu()
                # (5,B,L) -> (1,B,L) -> (B,L)
                questions = questions.gather(0, sample_index).squeeze(0)
            else:
                questions = torch.stack(questions, 0).detach().cpu()  # (B,L)

            for que in questions:
                tokens = []
                for q in que:
                    if q.item() in question_answer_vocab:
                        tokens.append(question_answer_vocab[q.item()])
                    else:
                        raise RuntimeError('识别不了的token {}'.format(q.item()))
                all_questions.append(tokens)

    acc = total_acc / total_count
    loss = total_loss / len(loader)
    if not write_preds:
        return acc, loss
    else:
        return acc,loss,all_preds,gts, video_idxs, question_idxs, video_names, (appr_obj_crosses, motion_obj_crosses),\
               (appr_frame_crosses, motion_frame_crosses), all_questions, \
                (appr_attens, motion_attens)

@torch.no_grad()
def Test(cfg, loader, model, device, loss_compute = None, write_preds = False):
    return Val(cfg, loader, model, device, loss_compute, write_preds, True)

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        # assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor if isinstance(t, torch.Tensor)]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError

    for m in modules:
        if hasattr(m, 'init_weights'):
            m.init_weights()
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

def set_all_random_seed(SEED):
    # to reproduce the results
    random.seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def save_checkpoint(epoch, model, optimizer, model_kwargs, ckpt_dir, file_prefix = 'best'):
    model_ckpt = os.path.join(ckpt_dir, file_prefix + '_model.pt')
    model_state = {
        'model_kwargs': model_kwargs,
        'model_state' : model.state_dict()
    }
    time.sleep(10)
    torch.save(model_state, model_ckpt)

    if optimizer is not None:
        optim_ckpt = os.path.join(ckpt_dir, file_prefix + '_optimizer.pt')

        optim_state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(optim_state, optim_ckpt)

def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer