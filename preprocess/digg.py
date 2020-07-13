#!/usr/bin/env python
# encoding: utf-8
# File Name: digg.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/27 15:26
# TODO:

import os
import logging
import argparse
from data import Data
logger = logging.getLogger(__name__)

class Digg(Data):
    def __init__(self, args):
        super(Digg, self).__init__(args)

    def extract_edge_list(self):
        edgelist = []
        logger.info("Extrack edges from %s", self.args.graph_file)
        with open(self.args.graph_file, "r") as f:
            nu = 0
            for line in f:
                nu += 1
                if (nu+1) % 100000 == 0:
                    logger.info("%d user profile proccessed" % (nu+1))
                content = line.strip().split(',')
                u, v = int(content[-2][1:-1]), int(content[-1][1:-1])
                u, v = self.get_vid(u, readonly=False), self.get_vid(v, readonly=False)
                edgelist.append((u,v))

        return edgelist

    def load_vote(self):
        logger.info("Load digg vote ...")

        with open(self.args.vote_file, "r") as f:
            for line in f:
                content = line.strip().split(',')
                t, u, vote = [int(x[1:-1]) for x in content]
                u = self.get_vid(u, readonly=True)
                if u > -1:
                    self.add_action(u, vote, t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    DIGG_BASE_DIR="./digg/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-active-neighbor", type=int, default=3,
            help="Minimum #active neighbors (inclusive)")
    parser.add_argument("--min-inf", type=int, default=100,
            help="Minimum influence (inclusive)")
    parser.add_argument("--max-inf", type=int, default=1500,
            help="Maximum influence (exclusive)")
    parser.add_argument("--min-degree", type=int, default=3,
            help="Minimum degree (inclusive)")
    parser.add_argument("--max-degree", type=int, default=31,
            help="Maximum degree (exclusive)")
    parser.add_argument("--ego-size", type=int, default=49,
            help="Size of ego network, not including center vertex")
    parser.add_argument("--graph-file", type=str,
            default=os.path.join(DIGG_BASE_DIR,
                "digg_friends/digg_friends.csv"))
    parser.add_argument("--vote-file", type=str,
            default=os.path.join(DIGG_BASE_DIR,
                "digg_votes/digg_votes1.csv"))
    parser.add_argument("--negative", type=int, default=1,
            help="Number of negative samples")
    parser.add_argument("--output", type=str,
            default=os.path.join(DIGG_BASE_DIR,
                "digg_processed/rw_sample_inf_100_1k_degree_3_31_ego_50_neg_1_restart_20"),
            help="output directory")
    parser.add_argument("--restart-prob", type=float, default=0.2)
    parser.add_argument("--walk-length", type=int, default=1000)
    args = parser.parse_args()
    digg_data = Digg(args)
    digg_data.load_graph()
    digg_data.load_vote()
    digg_data.sort_diffusion()
    digg_data.summarize_diffusion()
    digg_data.summarize_graph()
    digg_data.compute_structural_features()
    digg_data.dump()
    logger.info("Done.")

