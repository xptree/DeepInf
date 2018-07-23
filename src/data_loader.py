#!/usr/bin/env python
# encoding: utf-8
# File Name: data_loader.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/13 16:41
# TODO:

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils import load_w2v_feature
import sklearn
import itertools
import logging
import igraph
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InfluenceDataSet(Dataset):
    def __init__(self, file_dir, embedding_dim, seed, shuffle, model):
        self.graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        if model == "gat" or model == "pscn":
            self.graphs = self.graphs.astype(np.dtype('B'))
        elif model == "gcn":
            # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
                graph = (graph.T * d_root_inv).T * d_root_inv
                self.graphs[i] = graph
        else:
            raise NotImplementedError
        logger.info("graphs loaded!")

        # wheather a user has been influenced
        # wheather he/she is the ego user
        self.influence_features = np.load(
                os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        if shuffle:
            self.graphs, self.influence_features, self.labels, self.vertices = \
                    sklearn.utils.shuffle(
                        self.graphs, self.influence_features,
                        self.labels, self.vertices,
                        random_state=seed
                    )

        vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        self.vertex_features = torch.FloatTensor(vertex_features)
        logger.info("global vertex features loaded!")

        embedding_path = os.path.join(file_dir, "deepwalk.emb_%d" % embedding_dim)
        max_vertex_idx = np.max(self.vertices)
        embedding = load_w2v_feature(embedding_path, max_vertex_idx)
        self.embedding = torch.FloatTensor(embedding)
        logger.info("%d-dim embedding loaded!", embedding_dim)

        self.N = self.graphs.shape[0]
        logger.info("%d ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_embedding(self):
        return self.embedding

    def get_vertex_features(self):
        return self.vertex_features

    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]


class PatchySanDataSet(InfluenceDataSet):
    def get_bfs_order(self, g, v, size, key):
        order, indices, _ = g.bfs(v, mode="ALL")
        for j, start in enumerate(indices[:-1]):
            if start >= size:
                break
            end = indices[j + 1]
            order[start:end] = sorted(order[start:end],
                    key=lambda x: key[x][0],
                    reverse=True)

        return order[:size]

    def __init__(self, file_dir, embedding_dim, seed, shuffle,
            model, sequence_size=8, stride=1, neighbor_size=8):
        assert model == "pscn"
        super(PatchySanDataSet, self).__init__(file_dir,
                embedding_dim, seed, shuffle, model)
        n_vertices = self.graphs.shape[1]

        logger.info("generating receptive fields...")
        self.receptive_fields = []
        for i in range(self.graphs.shape[0]):
            adj = self.graphs[i]
            edges = list(zip(*np.where(adj)))
            g = igraph.Graph(edges=edges, directed=False)
            assert(g.vcount() == n_vertices)
            g.simplify()

            sequence = self.get_bfs_order(g, n_vertices - 1,
                    sequence_size, self.influence_features[i])
            neighborhoods = np.zeros((sequence_size, neighbor_size), dtype=np.int32)
            neighborhoods.fill(-1)

            for j, v in enumerate(sequence):
                if v < 0:
                    break
                shortest = list(itertools.islice(g.bfsiter(int(v), mode='ALL'), neighbor_size))
                for k, vtx in enumerate(shortest):
                    neighborhoods[j][k] = vtx.index

            neighborhoods = neighborhoods.reshape(sequence_size * neighbor_size)
            self.receptive_fields.append(neighborhoods)
        self.receptive_fields = np.array(self.receptive_fields, dtype=np.int32)
        logger.info("receptive fields generated!")

    def __getitem__(self, idx):
        return self.receptive_fields[idx], self.influence_features[idx], \
                self.labels[idx], self.vertices[idx]
