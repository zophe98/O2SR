
import torch
import torch.nn as nn
import math, copy
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from .simple_gcn import CGCN

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

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, num_head=1):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)

        self.num_head = num_head
        self.d_k = out_features // num_head

        if self.num_head > 1:
            self.mha_output_linear = nn.Linear(out_features, out_features)
            self.act = nn.ReLU()

    def forward(self, input, adj):
        """
        :param input: (*,*,*,N,d)
        :param adj:   (*,*,*,h,N,N)
        :return:
        """
        support = self.weight(input)                # (B,*,N,D)
        # (*,*,*,N,h,dk) -> (*,*,*,h,N,dk)
        support = support.view(*(support.size()[:-1]), self.num_head, self.d_k).transpose(-2, -3)
        # (*,*,*,h,N,dk)
        output = torch.matmul(adj, support)

        # (*,*,*,N,h,dk)
        output = output.transpose(-2, -3).contiguous()
        # (*,*,*,N,d)
        output = output.view(*(output.size()[:-2]), -1)

        if self.num_head > 1:
            output = self.mha_output_linear(self.act(output))

        output = self.layer_norm(output)
        return output

class GCN(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layer=1,dropout=0.1, num_head=1, norm_flag=True):
        super(GCN, self).__init__()
        self.norm_flag=True
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_size, hidden_size, num_head=num_head))
        for i in range(num_layer - 1):
            self.layers.append(GraphConvolution(hidden_size, hidden_size, num_head=num_head))
        if self.norm_flag:
            self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y,adj):
        y_gcn = y
        for i, layer in enumerate(self.layers):
            y_gcn = self.dropout(F.relu(layer(y_gcn, adj)))
        if self.norm_flag:
            return self.layernorm(x + y_gcn)
        else:
            return x + y_gcn

class AdjLearner(nn.Module):
    def __init__(self,
                 module_dim,
                 in_dim=-1,
                 dropout=0.1,
                 temperature=1.0,
                 num_head=1):
        super(AdjLearner, self).__init__()
        if in_dim < 0:
            in_dim = module_dim

        self.num_head = num_head
        self.d_k = module_dim // num_head
        self.temperature = temperature

        self.edge_layer_1 = FCNet([in_dim, module_dim], bias=False)
        self.edge_layer_2 = FCNet([in_dim, module_dim], bias=False)
        self.context_layer = FCNet([in_dim, module_dim], bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query, key,
                object_mask=None,
                is_softmax = True,
                mask_with=-1e+10):

        node_a = self.edge_layer_1(query)
        node_b = self.edge_layer_2(key)

        # transfer for multi-head
        # (B,h,TN,dk)
        node_a = node_a.view(*(node_a.size()[:-1]), self.num_head, self.d_k).transpose(-2, -3)
        node_b = node_b.view(*(node_b.size()[:-1]), self.num_head, self.d_k).transpose(-2, -3)

        # (B,h,TN, TN)
        adj = torch.matmul(node_a, node_b.transpose(-2, -1)) / math.sqrt(self.d_k)

        if object_mask is not None:
            # (B,1,TN) -> (B,1,1,TN)
            object_mask = object_mask.unsqueeze(1)
            adj = adj.masked_fill(object_mask.data == 0.0, mask_with)

        if is_softmax:
            adj = F.softmax(adj , dim=-1)
            if object_mask is not None:
                adj = adj * object_mask
                # drop nan
                adj = adj.masked_fill(adj != adj, 0.)

        adj = self.dropout(adj)
        return adj


class GCN_learner(nn.Module):
    def __init__(self, module_dim,num_layer=1,
                 dropout=0.1, temperature=1.0,
                 fixed = True,
                 graph_mask = None,
                 num_head=1,
                 upper=False,
                 norm_flag=True):
        super(GCN_learner, self).__init__()

        self.upper = upper
        # (1,1,N,N)
        if graph_mask is not None:
            self.graph_mask = graph_mask.unsqueeze(1)
        else:
            self.graph_mask = graph_mask

        self.temperature = temperature
        self.fixed = fixed

        if not self.fixed:
            self.adj_learner = AdjLearner(module_dim, module_dim, dropout, temperature, num_head=num_head)
        self.gcn = GCN(module_dim,module_dim, num_layer=num_layer, dropout=dropout, num_head=num_head, norm_flag=norm_flag)

    def forward(self, query_feature, key_feature,
                predifine_graph = None, node_mask=None, mask_with=-1e+10):
        """
        :param node_feature:        (B,N,d)
        :param context_feature:     (B,d)
        :param predifine_graph:     (1,N,N)
        :param node_mask:           (1,1,N)
        :return:
        """

        if self.fixed:
            graph_adj = predifine_graph.unsqueeze(1) # (1,1,n,n)
        else:
            # not softmax
            graph_adj = self.adj_learner(query_feature,key_feature,node_mask, False)

        if node_mask is not None:
            node_mask = node_mask.unsqueeze(1) # for multi-head
            graph_adj = graph_adj.masked_fill(node_mask.data == 0.0, mask_with)

        if self.graph_mask is not None:
            self.graph_mask = self.graph_mask.to(graph_adj.device)
            graph_adj = graph_adj.masked_fill(self.graph_mask[:,:,:graph_adj.size(2),:graph_adj.size(3)].data == 0.0, mask_with)

        graph_adj = F.softmax(graph_adj, dim=-1)

        if self.graph_mask is not None:
            # drop nan
            graph_adj = graph_adj.masked_fill(graph_adj != graph_adj, 0.0)

        if node_mask is not None:
            graph_adj = graph_adj.masked_fill(graph_adj != graph_adj, 0.0)

        if self.upper:
            graph_adj = graph_adj + graph_adj.transpose(-1,-2)

        node_feature = self.gcn(query_feature,key_feature, graph_adj)
        return node_feature, graph_adj

class ContextSTGCN(nn.Module):
    def __init__(self, module_dim,
                 num_layer=1,
                 dropout=0.1, temperature=100.0,
                 fixed_graph=True,
                 graph_mask=None,
                 num_head=1,
                 q2v_flag=False,
                 question_type='action'):

        super(ContextSTGCN, self).__init__()
        self.num_head = num_head
        self.graph_mask = graph_mask

        self.fixed_graph = fixed_graph
        self.question2visual_flag = q2v_flag
        self.task = question_type

        if self.question2visual_flag:
            self.q2v = CGCN(module_dim, dropout=dropout, temperature=1.0, num_head=num_head, norm=True)

        self.st_gcn = GCN_learner(module_dim,
                                       num_layer=num_layer,
                                       dropout=dropout,
                                       temperature=temperature,
                                       fixed=self.fixed_graph,
                                       graph_mask=self.graph_mask,
                                       num_head=num_head,
                                       upper=False)


    def forward(self, question_feature, glob_question_feature,question_mask,
                object_feature, object_mask=None,
                spatial_graph=None, temporal_graph=None):
        """
        :param question_feature:        (B,L,D)
        :param glob_question_feature    (B,D)
        :param question_mask:           (B,L)
        :param object_feature:
        :param object_mask:             (B,T,N)
        :param spatial_graph:
        :param temporal_graph:
        :return:
        """
        assert len(question_mask.size()) == 2
        assert len(object_mask.size()) == 3
        question_mask = question_mask.unsqueeze(1)
        B, T, N, D = object_feature.size()
        object_feature = object_feature.view(B, T * N, D)
        object_mask = object_mask.view(B, 1, T * N)
        cross_adj,spatial_adj, temporal_adj = None,None,None

        if self.question2visual_flag:
            object_feature, cross_adj = self.q2v(object_feature, question_feature, question_mask)

        object_feature, spatial_adj = self.st_gcn(object_feature, object_feature,spatial_graph,object_mask)

        object_feature = object_feature.view(B,T,N,D)

        return object_feature, cross_adj, spatial_adj, temporal_adj

class FrameGCN(nn.Module):
    def __init__(self, module_dim, dropout,
                 num_head=1, question2visual_flag=True,
                 visual2visual_flag=True,norm=True, question_type='action',
                 num_layer=1):
        super(FrameGCN, self).__init__()

        self.task = question_type
        self.q2v_flag = question2visual_flag
        self.v2v_flag = visual2visual_flag
        self.norm = norm

        if self.q2v_flag :
            self.q2v = CGCN(module_dim,dropout=dropout,temperature=1.0,num_head=num_head)

        if self.v2v_flag:
            self.v2v = GCN_learner(module_dim, num_layer=num_layer,dropout=dropout,temperature=1.0,fixed=False,
                                                  graph_mask=None,num_head=1,upper=False, norm_flag=norm)

        if self.norm:
            self.output_ln = nn.LayerNorm(module_dim, elementwise_affine=False)

    def forward(self, frame_feature, frame_mask,
                question_local_feature, question_mask,
                question_glob_feature):
        # question 2 visual
        question_mask = question_mask.unsqueeze(1) # (B,1,L)
        frame_mask = frame_mask.unsqueeze(1)
        cross_adj = None
        if self.q2v_flag:
            frame_feature, cross_adj = self.q2v(frame_feature, question_local_feature, question_mask)

        # visual 2 visual
        temporal_adj = None
        if self.v2v_flag:
            frame_feature, temporal_adj = self.v2v(frame_feature,frame_feature,None, frame_mask)

        if self.norm:
            frame_feature = self.output_ln(frame_feature)

        return frame_feature, temporal_adj, cross_adj