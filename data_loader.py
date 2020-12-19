import numpy as np
import os
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.decomposition import PCA
import json
import tqdm
import random
import torch
import pickle

class PreqDataset:
    def __init__(self, config):
        super().__init__()
        self.dataset_path = 'dataset/{}/'.format(config.dataset)
        self.config = config
        # load concepts
        with open(os.path.join(self.dataset_path, 'concepts.txt'), 'r', encoding='utf-8') as f:
            self.concepts = [c.strip() for c in f.read().split('\n') if c.strip()]
        # convert concepts into token ids by BertTokenizer
        file = os.path.join(self.dataset_path, 'concept_tokens.pkl')
        if not os.path.exists(file):
            if config.language == 'en':
                tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')
            if config.language == 'zh':
                tokenzier = BertTokenizer.from_pretrained('bert-base-chinese')
            self.tokens = []
            for concept in self.concepts:
                token = tokenzier.encode(concept, truncation=True, max_length=config.max_term_length)
                self.tokens.append(token)
            with open(file, 'wb') as f:
                pickle.dump(self.tokens, f)
        else:
            with open(file, 'rb') as f:
                self.tokens = pickle.load(f)
        # get concept embeddings by BertModel
        file = os.path.join(self.dataset_path, 'embeddings.pth')
        if not os.path.exists(file):
            if config.language == 'en':
                bert = BertModel.from_pretrained('bert-base-uncased')
            if config.language == 'zh':
                bert = BertModel.from_pretrained('bert-base-chinese')
            for p in bert.parameters():
                p.requires_grad = False
            bert.eval()
            bert.to(config.device)
            concept_embedding = []
            for token in tqdm.tqdm(self.tokens):
                token = torch.tensor(token, dtype=torch.long).to(config.device)
                with torch.no_grad():
                    h, _ = bert(token.unsqueeze(0))
                    h = h.squeeze(0)[1:-1]
                    ce = torch.mean(h, 0)
                concept_embedding.append(ce.cpu())
            pca = PCA()
            X = torch.stack(concept_embedding).numpy()
            X = pca.fit_transform(X)
            self.concept_embedding = torch.from_numpy(X)
            with open(file, 'wb') as f:
                torch.save(self.concept_embedding, f)
        else:
            with open(file, 'rb') as f:
                self.concept_embedding = torch.load(f)
        # load labeled data
        self.labeled_data, pairs = [], set()
        with open(os.path.join(self.dataset_path, 'pairs.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip().split('\t')
                c1, c2, label = s[0], s[1], int(s[2])
                i1, i2 = self.concepts.index(s[0]), self.concepts.index(s[1])
                t1, t2 = self.tokens[i1], self.tokens[i2]
                self.labeled_data.append({'c1': c1, 'c2': c2, 'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'ground_truth': label, 'label': label})
                pairs.add('{}-{}'.format(i1, i2))
        random.shuffle(self.labeled_data)
        # create unlabeled data
        self.unlabeled_data = []
        n = len(self.concepts)
        for i1 in range(n):
            for i2 in range(n):
                if i1 != i2 and '{}-{}'.format(i1, i2) not in pairs:
                    c1, c2 = self.concepts[i1], self.concepts[i2]
                    t1, t2 = self.tokens[i1], self.tokens[i2]
                    self.unlabeled_data.append({'c1': c1, 'c2': c2, 'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'ground_truth': -1, 'label': -1})
        # load graph data
        graph_path = os.path.join(self.dataset_path, 'graph.npy')
        if os.path.exists(graph_path):
            self.graph = np.load(graph_path)
        else:
            self.graph = np.eye(n)
        config.set_gcn_parameters(self.graph)
        print('data loader init finished.')

class MyPreqBatch:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        max_len_t1, max_len_t2 = 0, 0
        for datum in data:
            max_len_t1 = max(max_len_t1, len(datum['t1']))
            max_len_t2 = max(max_len_t2, len(datum['t2']))
        c1, c2, i1, i2, t1, t2, ground_truths, labels, origins = [], [], [], [], [], [], [], [], []
        for datum in data:
            c1.append(datum['c1'])
            c2.append(datum['c2'])
            i1.append(datum['i1'])
            i2.append(datum['i2'])
            t1.append(datum['t1']+[0]*(max_len_t1-len(datum['t1'])))
            t2.append(datum['t2']+[0]*(max_len_t2-len(datum['t2'])))
            ground_truths.append(datum['ground_truth'])
            labels.append(datum['label'])
            origins.append(datum)
        i1 = torch.tensor(i1, dtype=torch.long).to(self.config.device)
        i2 = torch.tensor(i2, dtype=torch.long).to(self.config.device)
        t1 = torch.tensor(t1, dtype=torch.long).to(self.config.device)
        t2 = torch.tensor(t2, dtype=torch.long).to(self.config.device)
        obj = {'c1': c1, 'c2': c2, 'i1': i1, 'i2': i2, 't1': t1, 't2': t2, 'ground_truths': ground_truths, 'labels': labels, 'origins': origins}
        return obj

class MyConceptBatch:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        max_len = max([len(datum) for datum in data])
        concepts = []
        for datum in data:
            concepts.append(datum+[0]*(max_len-len(datum)))
        concepts = torch.tensor(concepts, dtype=torch.long).to(self.config.device)
        obj = {'concepts': concepts}
        return obj

class PreqDataLoader:
    def __init__(self, config):
        self.dataset = PreqDataset(config)
        self.config = config
        self.fn = MyPreqBatch(config)
        self.fn2 = MyConceptBatch(config)
    
    def get_train(self, mode):
        labeled, unlabeled = self.dataset.labeled_data, self.dataset.unlabeled_data
        train, eval, test = [], [], []
        for i in range(self.config.num_classes):
            class_labeled = [datum for datum in labeled if datum['ground_truth']==i]
            n, d = len(class_labeled), self.config.init_num
            splits = random_split(class_labeled, [d, (n-d)//2, (n-d+1)//2])
            train += splits[0]
            eval += splits[1]
            test += splits[2]
        '''
        n = len(labeled)
        d = self.config.init_num
        train, eval, test = random_split(labeled, [d*2, (n-d*2)//2, (n-d*2+1)//2])
        '''
        train += [datum for datum in unlabeled if datum['label'] >= 0]
        train = DataLoader(train, self.config.batch_size('train'), shuffle=True, collate_fn=self.fn)
        unlabeled = DataLoader(unlabeled, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        eval = DataLoader(eval, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        test = DataLoader(test, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        return train, unlabeled, eval, test
    
    def get_concepts(self):
        data = self.dataset.tokens
        data = DataLoader(data, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn2)
        return data
    
    def get_predict(self):
        data = self.dataset.labeled_data+self.dataset.unlabeled_data
        data = DataLoader(data, self.config.batch_size('eval'), shuffle=False, collate_fn=self.fn)
        return data