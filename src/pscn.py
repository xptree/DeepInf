#!/usr/bin/env python
# encoding: utf-8
# File Name: pscn.py
# Author: Jiezhong Qiu
# Create Time: 2018/01/28 17:01
# TODO:

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BatchPSCN(nn.Module):
    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature,
            use_vertex_feature, instance_normalization,
            neighbor_size, sequence_size, fine_tune=False):
        super(BatchPSCN, self).__init__()
        assert len(n_units) == 4
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        # input is of shape bs x num_feature x l where l = w*k
        # after conv1, shape=(bs x ? x w)
        # after conv2 shape=(bs x ? x w/2)
        self.conv1 = nn.Conv1d(in_channels=n_units[0],
                    out_channels=n_units[1], kernel_size=neighbor_size,
                    stride=neighbor_size)
        k = 1
        self.conv2 = nn.Conv1d(in_channels=n_units[1],
                    out_channels=n_units[2], kernel_size=k, stride=1)
        self.fc = nn.Linear(in_features=n_units[2] * (sequence_size - k + 1),
                    out_features=n_units[3])

    def forward(self, x, vertices, recep):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        bs, l = recep.size()
        n = x.size()[1] # x is of shape bs x n x num_feature
        offset = torch.ger(torch.arange(0, bs).long(), torch.ones(l).long() * n)
        offset = Variable(offset, requires_grad=False)
        offset = offset.cuda()
        recep = (recep.long() + offset).view(-1)
        x = x.view(bs * n, -1)
        x = x.index_select(dim=0, index=recep)
        x = x.view(bs, l, -1) # x is of shape bs x l x num_feature, l=w*k
        x = x.transpose(1, 2) # x is of shape bs x num_feature x l, l=w*k
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.view(bs, -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
