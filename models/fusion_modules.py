import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class QuestionGuildeTemporalFusion(nn.Module):
    def __init__(self, module_dim=512):
        super(QuestionGuildeTemporalFusion, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim, bias=False)
        self.attn = nn.Linear(module_dim, 1, bias=False)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)
        self.atten_dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat, visual_mask, visual=False):

        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)  # (B,T,d)
        attn = self.attn(v_q_cat)  # (B, T, 1)

        visual_mask = visual_mask.unsqueeze(2) # (B,T,1)
        attn.data.masked_fill_(visual_mask.data == 0, -float("inf"))
        attn = F.softmax(attn, dim=1)  # (bz, T, 1)
        # remove nan from softmax on -inf
        attn.data.masked_fill_(attn.data != attn.data, 0)
        attn = self.atten_dropout(attn)

        v_distill = (attn * visual_feat).sum(1)
        if visual:
            return v_distill, attn
        else:
            # (B, D)
            return v_distill

class QuestionGuildeSpatialFusion(nn.Module):
    def __init__(self, module_dim, num_head=8, attn='gumbel_softmax', dropout=0.15):
        super(QuestionGuildeSpatialFusion, self).__init__()
        assert attn in ['softmax', 'gumbel_softmax']
        self.num_head = num_head
        self.module_dim = module_dim
        self.attn = attn
        assert self.module_dim % self.num_head == 0
        self.d_k = self.module_dim // self.num_head


        self.question_proj = nn.Linear(module_dim, self.d_k, bias=False)
        self.visual_proj = nn.Linear(module_dim, self.d_k, bias=False)

        self.attn_proj = nn.Sequential(
            nn.Linear(self.d_k * 2, self.d_k, bias=False),
            nn.ELU(),
            nn.Linear(self.d_k, self.num_head, bias=False)
        )

        self.dropout = nn.Dropout(dropout)
        self.atten_dropout = nn.Dropout(dropout)

    def forward(self, question_feature, visual_feature, visual_mask=None):
        # (B,D)
        q_proj = self.question_proj(question_feature)

        v_proj = self.visual_proj(visual_feature)

        # (B,1,1,D) -> (B,T,N,D)
        q_proj = q_proj.unsqueeze(1).unsqueeze(1).expand_as(v_proj)

        # (B, T, N, d_k * 2)
        qv_feature = torch.cat((v_proj, q_proj), dim=-1)

        attn_weight = self.attn_proj(qv_feature) # (B, T, N, h)

        if visual_mask is not None:
            attn_weight.masked_fill_((visual_mask.unsqueeze(-1)) == 0, -float("inf"))

        # (B,T,N,h)
        if self.attn == 'gumbel_softmax':
            hard_attn = F.gumbel_softmax(attn_weight, tau=1, hard=True, dim=2)
        else:
            hard_attn = F.softmax(attn_weight, dim=2)

        if visual_mask is not None:
            hard_attn = hard_attn.masked_fill(hard_attn != hard_attn, 0.)

        # (B,T,N,d,h)
        hard_attn = self.atten_dropout(hard_attn)
        v_distill = visual_feature.unsqueeze(-1) * hard_attn.unsqueeze(-2)
        # (B,T,d,h)
        v_distill = v_distill.sum(2)
        # multi-head
        v_distill = v_distill.sum(-1) # (B,T,d)

        return v_distill

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0.0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim, bias=bias),
                                      dim=None))
            if '' != act and act is not None:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1], bias=bias),
                                  dim=None))
        if '' != act and act is not None:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)