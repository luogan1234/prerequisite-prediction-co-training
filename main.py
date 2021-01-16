import argparse
from config import Config
from data_loader import PreqDataLoader
from processor import Processor
import pickle
import os
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def main():
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/model_states/'):
        os.mkdir('result/model_states/')
    if not os.path.exists('result/predictions/'):
        os.mkdir('result/predictions/')
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, required=True, choices=['cs', 'math', 'psy', 'phy', 'chem'])
    parser.add_argument('-text_encoder', type=str, required=True, choices=['lstm', 'bert', 'bert-freeze'])
    parser.add_argument('-graph_layer', type=str, required=True, choices=['gcn', 'gat'])
    parser.add_argument('-init_num', type=int, default=-1)
    parser.add_argument('-max_change_num', type=int, default=36)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    config = Config(args.dataset, args.text_encoder, args.graph_layer, args.init_num, args.max_change_num, args.seed, args.cpu)
    model_path = 'result/model_states/{}.pth'.format(config.store_name())
    if os.path.exists(model_path):
        print('experiment done.')
        return
    data_loader = PreqDataLoader(config)
    processor = Processor(config, data_loader)
    processor.train()
    processor.predict()

if __name__ == '__main__':
    main()