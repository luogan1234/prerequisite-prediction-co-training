import numpy as np
import argparse
import json
from collections import Counter
import math
import random
import tqdm
import os

def add_edge(graph, c1s, c2s, d):
    if c1s and c2s:
        d /= max(len(c1s), len(c2s))
        for c1 in c1s:
            for c2 in c2s:
                if c1 != c2:
                    graph[c1][c2] += d

def build_concept_graph(dataset, alpha, no_weight):
    prefix = 'dataset/{}/'.format(dataset)
    concepts = []
    with open(prefix+'concepts.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                concepts.append(line)
    videos = []
    with open(prefix+'captions.txt', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                videos.append(line.split(',')[0])
    cn, vn = len(concepts), len(videos)
    video_to_concept = {}
    with open(prefix+'video-concepts.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                v = videos.index(obj['video'])
                cs = [concepts.index(c) for c in obj['concepts']]
                video_to_concept[v] = cs
    cgraph, vgraph = np.zeros((cn, cn)), np.zeros((vn, vn))
    with open(prefix+'course-videos.json', 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            if line:
                obj = json.loads(line)
                c = obj['course']
                vs = [videos.index(v) for v in obj['videos']]
                n = len(vs)
                for i in range(n):
                    for j in range(i+1, n):
                        vgraph[vs[i]][vs[j]] += alpha ** (j-i)
    for i in range(vn):
        for j in range(vn):
            if vgraph[i][j] > 0:
                add_edge(cgraph, video_to_concept[i], video_to_concept[j], vgraph[i][j])
    print('video graph covered edge proportion: {:.3f}, total edge weight: {:.3f}'.format(len(vgraph[vgraph>0]) / (vn*vn), np.sum(vgraph[vgraph>0])))
    print('concept graph covered edge proportion: {:.3f}, total edge weight: {:.3f}'.format(len(cgraph[cgraph>0]) / (cn*cn), np.sum(cgraph)))
    if no_weight:
        cgraph[cgraph>0] = 1
    np.save(prefix+'graph.npy', np.array(cgraph))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-alpha', type=float, default=None)
    parser.add_argument('-no_weight', action='store_true')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    assert os.path.exists('dataset/{}/'.format(args.dataset))
    set_seed(args.seed)
    if not args.alpha:
        if args.dataset == 'moocen':
            args.alpha = 0.1
        if args.dataset == 'mooczh':
            args.alpha = 0.3
    assert 0.0 <= args.alpha <= 1.0
    build_concept_graph(args.dataset, args.alpha, args.no_weight)

if __name__ == '__main__':
    main()
   