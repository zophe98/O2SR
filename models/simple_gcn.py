# Graph attention operations without context supervision

import torch
import torch.nn as nn
import math
from torch.nn.utils.weight_norm import weight_norm

from torch.nn import functional as F

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

    def __init__(self, in_features, out_features, num_head=1, norm=True):
        super(GraphConvolution, self).__init__()
        self.norm = norm
        self.weight = nn.Linear(in_features, out_features, bias=False)

        if self.norm:
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
        support = self.weight(input)

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

        if self.norm:
            output = self.layer_norm(output)
        return output

class GCN(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layer=1,dropout=0.1, num_head=1):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_size, hidden_size, num_head=num_head))
        for i in range(num_layer - 1):
            self.layers.append(GraphConvolution(hidden_size, hidden_size, num_head=num_head))

        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y,adj):
        # y_gcn = self.layernorm(y)

        y_gcn = y
        for i, layer in enumerate(self.layers):
            y_gcn = self.dropout(F.relu(layer(y_gcn, adj)))

        return self.layernorm(x + y_gcn)

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


        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query, key,
                object_mask=None,
                is_softmax = True,
                mask_with=-1e+10):
        """
        :param object_feature:              (B, TN, d)
        :param object_mask:                 (B, 1,  TN)
        :return:
            (B, TN, d)
        """
        # query Nxd
        # key   Mxd
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
                 upper=False):
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
        self.gcn = GCN(module_dim,module_dim, num_layer=num_layer, dropout=dropout, num_head=num_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_feature, key_feature,
                predifine_graph = None, node_mask=None, mask_with=-1e+10):
        """
        :param node_feature:        (B,N,d)
        :param predifine_graph:     (1,N,N)
        :param node_mask:           (1,1,N)
        :return:
        """

        if self.fixed:
            graph_adj = predifine_graph.unsqueeze(1) # (1,1,n,n)
        else:
            # not softmax
            graph_adj = self.adj_learner(query_feature,key_feature, node_mask, False)

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

class CAdjLearner(nn.Module):
    def __init__(self,
                 module_dim,
                 in_dim=-1,
                 dropout=0.1,
                 temperature=1.0,
                 num_head=1):
        super(CAdjLearner, self).__init__()
        if in_dim < 0:
            in_dim = module_dim
        self.num_head = num_head
        self.d_k = module_dim // num_head
        self.temperature = temperature

        # 两个顶点的转换矩阵
        self.edge_layer_1 = FCNet([in_dim, module_dim], act='', bias=False)
        self.edge_layer_2 = FCNet([in_dim, module_dim], act='', bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query, key,
                object_mask=None,
                is_softmax = True,
                mask_with=-1e+10):
        """
        :param object_feature:              (B, TN, d)
        :param object_mask:                 (B, 1,  TN)
        :return:
            (B, TN, d)
        """
        node_a = self.edge_layer_1(query)   # (B, N, d)
        node_b = self.edge_layer_2(key)     # (B, M, d)


        # transfer for multi-head
        # (B,h,N,dk)
        node_a = node_a.view(*(node_a.size()[:-1]), self.num_head, self.d_k).transpose(-2, -3)
        node_b = node_b.view(*(node_b.size()[:-1]), self.num_head, self.d_k).transpose(-2, -3)

        # (B,h,N,M)
        adj = torch.matmul(node_a, node_b.transpose(-2, -1)) / math.sqrt(self.d_k)

        if object_mask is not None:
            object_mask = object_mask.unsqueeze(-2)
            adj = adj.masked_fill(object_mask.data == 0.0, mask_with)

        if is_softmax:
            adj = F.softmax(adj , dim=-1)
            if object_mask is not None:
                adj = adj * object_mask
                # drop nan
                adj = adj.masked_fill(adj != adj, 0.)

        adj = self.dropout(adj)
        return adj

class CGCN(nn.Module):
    def __init__(self, module_dim,num_layer=1,
                 dropout=0.1, temperature=1.0, num_head=1,
                 residual=True, norm=True):
        super(CGCN, self).__init__()

        self.norm_flag = norm
        self.temperature = temperature
        self.adj_learner = CAdjLearner(module_dim, module_dim, dropout, temperature, num_head=num_head)
        self.residual = residual

        self.gcn = GraphConvolution(module_dim, module_dim, num_head=num_head, norm=True)

        self.dropout = nn.Dropout(dropout)

        if self.norm_flag:
            self.out_ln = nn.LayerNorm(module_dim, elementwise_affine=False)

    def forward(self, query, key, mask=None, mask_with=-1e+10):
        graph_adj = self.adj_learner(query, key, mask, False)  # not softmax (B,h,N,M)

        if mask is not None:
            mask = mask.unsqueeze(-2)
            graph_adj = graph_adj.masked_fill(mask.data == 0.0, mask_with)

        graph_adj = graph_adj * self.temperature
        graph_adj = F.softmax(graph_adj, dim=-1)

        if mask is not None:
            graph_adj = graph_adj.masked_fill(graph_adj != graph_adj, 0.0)
        node_feature = self.dropout(F.relu(self.gcn(key, graph_adj)))
        if self.residual:
            if self.norm_flag:
                node_feature = self.out_ln(node_feature + query)
            else:
                node_feature = node_feature + query

        return node_feature, graph_adj


