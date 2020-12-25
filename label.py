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

def work(field):
    print('Create candidate pairs for field {}.'.format(field))
    concepts = load('dataset/{}/final_concepts.txt'.format(field))
    predictions = load('result/predictions/{}_bert-freeze_gcn_-1_36_0.json'.format(field))
    predictions = [obj for obj in predictions if obj['ground_truth']==-1 and obj['c1'] in concepts and obj['c2'] in concepts]
    predictions = [obj for obj in predictions if obj['text_predict'][1]+obj['graph_predict'][1] > 1.8]
    predictions.sort(key=lambda x: (x['text_predict'][1]+x['graph_predict'][1]), reverse=True)
    res = []
    for obj in random.sample(predictions, min(len(predictions), 4000)):
        res.append({'label': 0, 'c1': obj['c1'], 'c2': obj['c2']})
    save('label/{}.json'.format(field), res)

if __name__ == '__main__':
    work('cs')
    work('psy')
    work('math')
    work('phy')
    work('chem')