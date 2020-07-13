#!/usr/bin/env python
# encoding: utf-8
# File Name: main.py
# Author: Jiezhong Qiu # Create Time: 2017/11/15 06:30
# TODO:

import numpy as np
import argparse
import igraph
import logging
import itertools
import os
import random
logger = logging.getLogger(__name__)


def random_walk_with_restart(g, start, restart_prob):
    current = random.choice(start)
    stop = False
    while not stop:
        stop = yield current
        current = random.choice(start) if random.random() < restart_prob or g.degree(current)==0 \
                else random.choice(g.neighbors(current))

class Data:
    def __init__(self, args):
        try:
            os.makedirs(args.output)
        except OSError as e:
            logger.info(str(e))
        self.args = args
        self.adj_matrices = []
        self.features = []
        self.vertices = []
        self.labels = []
        self.v2id = {}
        self.structural_features = []
        self.diffusion = {}

    def add_action(self, user, action, t):
        if action not in self.diffusion:
            self.diffusion[action] = []
        self.diffusion[action].append((user, t))


    def get_vid(self, u, readonly=False):
        if u not in self.v2id:
            if readonly:
                return -1
            newid = len(self.v2id)
            self.v2id[u] = newid
        return self.v2id[u]

    def extract_edge_list(self):
        raise NotImplementedError

    def load_graph(self):
        edgelist = self.extract_edge_list()
        self.n_vertices = len(self.v2id)
        self.graph = igraph.Graph(len(self.v2id), directed=False)
        self.graph.add_edges(edgelist)
        self.graph.to_undirected()
        self.graph.simplify(multiple=True, loops=True)
        edgelist_path = os.path.join(self.args.output, "graph.edgelist")
        with open(edgelist_path, "w") as f:
            self.graph.write(f, format="edgelist")

        # add some fake vertices
        self.graph.add_vertices(self.args.ego_size)
        self.degree = self.graph.degree()

    def summarize_diffusion(self):
        diffusion_size = [len(v) for v in self.diffusion.values()]
        logger.info("mean diffusion length %.2f", np.mean(diffusion_size))
        logger.info("max diffusion length %.2f", np.max(diffusion_size))
        logger.info("min diffusion length %.2f", np.min(diffusion_size))
        for i in range(1, 10):
            logger.info("%d-th percentile of diffusion length %.2f", i*10,
                    np.percentile(diffusion_size, i*10))
        logger.info("95-th percentile of diffusion length %.2f",
                np.percentile(diffusion_size, 95))

    def summarize_graph(self):
        logger.info("mean degree %.2f", np.mean(self.degree))
        logger.info("max degree %.2f", np.max(self.degree))
        logger.info("min degree %.2f", np.min(self.degree))
        for i in range(1, 10):
            logger.info("%d-th percentile of degree %.2f", i*10,
                    np.percentile(self.degree, i*10))
        logger.info("95-th percentile of degree %.2f",
                    np.percentile(self.degree, 95))

    def compute_structural_features(self):
        logger.info("Computing rarity (reciprocal of degree)")
        degree = np.array(self.graph.degree())
        degree[degree==0] = 1
        rarity = 1. / degree
        logger.info("Computing clustering coefficient..")
        cc = self.graph.transitivity_local_undirected(mode="zero")
        logger.info("Computing pagerank...")
        pagerank = self.graph.pagerank(directed=False)
        logger.info("Computing constraint...")
        """
        constraint = self.graph.constraint()
        logger.info("Computing closeness...")
        closeness = self.graph.closeness(cutoff=3)
        logger.info("Computing betweenness...")
        betweenness = self.graph.betweenness(cutoff=3, directed=False)
        logger.info("Computing authority_score...")
        """
        authority_score = self.graph.authority_score()
        logger.info("Computing hub_score...")
        hub_score = self.graph.hub_score()
        logger.info("Computing evcent...")
        evcent = self.graph.evcent(directed=False)
        logger.info("Computing coreness...")
        coreness = self.graph.coreness()
        logger.info("Structural feature computation done!")
        self.structural_features = np.column_stack(
                (rarity, cc, pagerank,
                 #constraint, closeness, betweenness,
                 authority_score, hub_score, evcent, coreness))

        with open(os.path.join(self.args.output, "vertex_feature.npy"), "wb") as f:
            np.save(f, self.structural_features)


    def create(self, u, p, t, label, user_affected_now):
        graph = self.graph
        args = self.args

        active_neighbor, inactive_neighbor = [], []

        for v in graph.neighbors(u):
            if v in user_affected_now:
                active_neighbor.append(v)
            else:
                inactive_neighbor.append(v)
        if len(active_neighbor) < args.min_active_neighbor:
            return

        n = args.ego_size + 1
        n_active = 0
        ego = []
        if len(active_neighbor) < args.ego_size:
            # we should sample some inactive neighbors
            n_active = len(active_neighbor)
            ego = set(active_neighbor)
            for v in itertools.islice(random_walk_with_restart(graph,
                start=active_neighbor + [u,], restart_prob=args.restart_prob), args.walk_length):
                if v!=u and v not in ego:
                    ego.add(v)
                    if len(ego) == args.ego_size:
                        break
            ego = list(ego)
            if len(ego) < args.ego_size:
                return
                n_fake = args.ego_size - len(ego)
                logger.info("generate %d fake vertices", n_fake)
                ego += list(range(self.n_vertices, self.n_vertices+n_fake))
        else:
            n_active = args.ego_size
            samples = np.random.choice(active_neighbor,
                    size=args.ego_size,
                    replace=False)
            ego += samples.tolist()
        ego.append(u)

        order = np.argsort(ego)
        ranks = np.argsort(order)

        subgraph = graph.subgraph(ego, implementation="create_from_scratch")
        adjacency = np.array(subgraph.get_adjacency().data)
        adjacency = adjacency[ranks][:, ranks]
        self.adj_matrices.append(adjacency)

        feature = np.zeros((n,2))
        for idx, v in enumerate(ego[:-1]):
            if v in user_affected_now:
                feature[idx, 0] = 1
        feature[n-1, 1] = 1
        self.features.append(feature)
        self.vertices.append(np.array(ego, dtype=int))
        self.labels.append(label)

        circle = subgraph.subgraph(ranks[:n_active], implementation="create_from_scratch")

        if len(self.labels) % 10000 == 0:
            logger.info("Collected %d instances", len(self.labels))

    def dump_data(self):
        self.adj_matrices = np.array(self.adj_matrices)
        self.features = np.array(self.features)
        self.vertices = np.array(self.vertices)
        self.labels = np.array(self.labels)

        output_dir = self.args.output
        with open(os.path.join(output_dir, "adjacency_matrix.npy"), "wb") as f:
            np.save(f, self.adj_matrices)
        with open(os.path.join(output_dir, "influence_feature.npy"), "wb") as f:
            np.save(f, self.features)
        with open(os.path.join(output_dir, "vertex_id.npy"), "wb") as f:
            np.save(f, self.vertices)
        with open(os.path.join(output_dir, "label.npy"), "wb") as f:
            np.save(f, self.labels)

        logger.info("Dump %d instances in total" % (len(self.labels)))

        self.adj_matrices = []
        self.features = []
        self.vertices = []
        self.labels = []


    def dump(self):
        logger.info("Dump data ...")
        graph = self.graph
        diffusion = self.diffusion
        args = self.args

        nu = 0
        for cascade_idx, cascade in diffusion.items():
            nu += 1
            if nu % 10000 == 0:
                logger.info("%d (%.2f percent) diffusion processed" % (nu, 100.*nu/len(diffusion)))

            if len(cascade)<args.min_inf or len(cascade)>=args.max_inf:
                continue
            user_affected_all = set([item[0] for item in cascade])
            user_affected_now = set()
            last = 0
            #infected = set((cas[0][0],))
            for item in cascade[1:]:
                u, t = item
                while last < len(cascade) and cascade[last][1] < t:
                    user_affected_now.add(cascade[last][0])
                    last += 1
                if len(user_affected_now) == 0:
                    continue
                if u in user_affected_now:
                    continue
                if self.degree[u]>=args.min_degree and self.degree[u]<args.max_degree:
                    # create positive case for user u, photo p, time t
                    self.create(u, cascade_idx, t, 1, user_affected_now)

                negative = list(set(graph.neighbors(u)) - user_affected_all)
                negative = [v for v in negative \
                        if self.degree[v]>=args.min_degree \
                        and self.degree[v]<args.max_degree]
                if len(negative) == 0:
                    continue
                negative_sample = np.random.choice(negative,
                        size=min(args.negative, len(negative)), replace=False)
                for v in negative_sample:
                    # create negative case for user v photo p, time t
                    self.create(v, cascade_idx, t, 0, user_affected_now)
        if len(self.labels) > 0:
            self.dump_data()

    def sort_diffusion(self):
        for k in self.diffusion:
            self.diffusion[k] = sorted(self.diffusion[k],
                                        key=lambda item: item[1])

