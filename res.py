import numpy as np
import json
import numpy as np
import argparse

def get_info(score):
    max_f1 = np.max([obj['f1'] for obj in score])
    min_f1 = np.min([obj['f1'] for obj in score])
    mean_p = np.mean([obj['p'] for obj in score])
    mean_r = np.mean([obj['r'] for obj in score])
    mean_f1 = np.mean([obj['f1'] for obj in score])
    return max_f1, min_f1, mean_p, mean_r, mean_f1

def stat(args):
    with open('result/result.txt', 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if args.dataset not in ['', obj['dataset']] or args.text_encoder not in ['',  obj['text_encoder']] or args.graph_layer not in ['', obj['graph_layer']]:
                continue
            print('dataset: {}, text_encoder: {}, graph_layer: {}, init_num: {}, max_change_num: {}'.format(obj['dataset'], obj['text_encoder'], obj['graph_layer'], obj['init_num'], obj['max_change_num']))
            '''
            for i, [score_text, score_graph] in enumerate(zip(obj['score_texts'], obj['score_graphs'])):
                print('Iteration {}'.format(i))
                max_f1, min_f1, mean_p, mean_r, mean_f1 = get_info(score_text)
                print('score_text: max_f1 {:.3f}, min_f1 {:.3f}, mean_p {:.3f} mean_r {:.3f} mean_f1 {:.3f}'.format(max_f1, min_f1, mean_p, mean_r, mean_f1))
                max_f1, min_f1, mean_p, mean_r, mean_f1 = get_info(score_graph)
                print('score_graph: max_f1 {:.3f}, min_f1 {:.3f}, mean_p {:.3f} mean_r {:.3f} mean_f1 {:.3f}'.format(max_f1, min_f1, mean_p, mean_r, mean_f1))
            '''
            for score_text, score_graph in zip([obj['score_texts'][0],obj['score_texts'][-1]], [obj['score_graphs'][0],obj['score_graphs'][-1]]):
                max_f1, min_f1, mean_p, mean_r, mean_f1 = get_info(score_text)
                print(' & {:.3f} & {:.3f} & {:.3f}'.format(mean_p, mean_r, mean_f1))
                max_f1, min_f1, mean_p, mean_r, mean_f1 = get_info(score_graph)
                print(' & {:.3f} & {:.3f} & {:.3f}'.format(mean_p, mean_r, mean_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prerequisite prediction')
    parser.add_argument('-dataset', type=str, default='', choices=['', 'cs', 'math', 'psy', 'phy', 'chem'])
    parser.add_argument('-text_encoder', type=str, default='', choices=['', 'lstm', 'bert', 'bert-freeze'])
    parser.add_argument('-graph_layer', type=str, default='', choices=['', 'gcn', 'gat'])
    args = parser.parse_args()
    stat(args)