import os
import json
import random

def load(file, is_json=None):
    if not os.path.exists(file):
        return []
    if is_json is None:
        is_json = file.endswith('json')
    with open(file, 'r', encoding='utf-8') as f:
        res = [line.strip() for line in f.read().split('\n') if line.strip()]
    if is_json:
        res = [json.loads(line) for line in res]
    return res

def save(file, data, is_json=None):
    if is_json is None:
        is_json = file.endswith('json')
    with open(file, 'w', encoding='utf-8') as f:
        for obj in data:
            string = json.dumps(obj, ensure_ascii=False) if is_json else obj
            f.write(string+'\n')

def work(field, num):
    print('Create candidate pairs for field {}.'.format(field))
    concepts = load('dataset/{}/concepts.txt'.format(field))
    predictions = load('result/predictions/{}_bert_gcn_-1_36_0.json'.format(field))
    filtered_list = []
    for obj in predictions:
        if obj['ground_truth']==-1 and obj['c1'] in concepts and obj['c2'] in concepts and obj['text_predict'][1]+obj['graph_predict'][1] > 1.7:
            filtered_list.append(obj)
    filtered_list.sort(key=lambda x: (x['text_predict'][1]+x['graph_predict'][1]), reverse=True)
    res = []
    for obj in random.sample(filtered_list, min(len(filtered_list), num)):
        res.append({'label': 0, 'c1': obj['c1'], 'c2': obj['c2']})
    save('label/{}.json'.format(field), res)

if __name__ == '__main__':
    work('cs', 3000)
    work('psy', 1000)
    work('math', 1000)
    #work('phy')
    #work('chem')