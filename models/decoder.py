import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, input_dim=-1, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        if input_dim < 0:
            input_dim = module_dim

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim + input_dim, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding, *args):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out

class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512, input_dim=-1, num_classes = 1):
        super(OutputUnitCount, self).__init__()

        if input_dim < 0:
            input_dim = module_dim

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim + input_dim, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_classes))

    def forward(self, question_embedding, visual_embedding, *args):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self,module_dim=512, input_dim=-1,num_classes=1):
        super(OutputUnitMultiChoices, self).__init__()

        if input_dim < 0:
            input_dim = module_dim
        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim + input_dim, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_classes))

    def forward(self,ans_candidates_embedding, a_visual_embedding, *args):
        """
        Args:
        :param ans_candidates_embedding:    (B,5,D)
        :param a_visual_embedding:          (B,5,D)
        :return:
            (B,5，1)
        """
        B, N, d = ans_candidates_embedding.size()
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        # out of shape (B,N,4*D)
        out = torch.cat([a_visual_embedding,
                         ans_candidates_embedding], -1)

        out = out.view(B*N,-1)
        # (B*5,1)
        out = self.classifier(out)
        out = out.view(B,N,-1)
        return out

class AnswerDecoder(nn.Module):
    def __init__(self, task, module_dim, num_classes, input_dim=-1):
        super(AnswerDecoder, self).__init__()
        if task == 'action' or task == 'transition':
            self.decoder = OutputUnitMultiChoices(module_dim,input_dim,num_classes)
        elif task == 'count':
            self.decoder = OutputUnitCount(module_dim,input_dim,num_classes)
        elif task == 'frameqa':
            self.decoder = OutputUnitOpenEnded(module_dim,input_dim,num_classes)
        else:
            raise ValueError("Not support task {}".format(task))

    def forward(self, question_embedding, qvisual_embedding,
                answer_embedding = None, ans_visual_embedding = None):
        # (B,num_classes), (B,5,num_classes)
        return self.decoder(question_embedding,qvisual_embedding,
                            answer_embedding,ans_visual_embedding)

if __name__ == '__main__':
    task = 'count'
    B,D = 32, 256
    question_embedding = torch.rand((B,D))
    q_visual_embedding = torch.rand((B,D))

    ans_candidates_embedding = torch.rand((B,5,D))
    a_visual_embedding  = torch.rand((B,5,D))

    output_unit = OutputUnitOpenEnded(256)

    out = output_unit(question_embedding,q_visual_embedding)

    print(out.size())