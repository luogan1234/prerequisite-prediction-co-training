import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import numpy as np
import tqdm
import random
from model.co_training_models import CoTrainingModels
import pickle
import copy

class Processor:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
    
    def loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def train_one_step(self, data, model, optimizer):
        outputs = model(data)
        loss = self.loss(outputs, data['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    
    def eval_one_step(self, data, model):
        with torch.no_grad():
            outputs = model(data)
            loss = self.loss(outputs, data['labels'])
            outputs = outputs.cpu().numpy()
        return outputs, loss.item()
    
    def evaluate(self, data, model):
        # evaluate single model
        model.eval()
        trues, preds = [], []
        eval_loss = 0.0
        for batch in data:
            trues.extend(batch['labels'])
            outputs, loss = self.eval_one_step(batch, model)
            preds.extend(np.argmax(outputs, axis=1).tolist())
            eval_loss += loss
        eval_loss /= len(data)
        model.train()
        acc = accuracy_score(trues, preds)
        p = precision_score(trues, preds, average='binary')
        r = recall_score(trues, preds, average='binary')
        f1 = f1_score(trues, preds, average='binary')
        score = {'acc': acc, 'p': p, 'r': r, 'f1': f1}
        return eval_loss, score
    
    def new_label(self, mode):
        # cache embeddings first, especially for text models (the unlabeled data is O(n^2))
        data = self.data_loader.get_unlabeled_data()
        print('Batch data number: unlabeled {}'.format(len(data)))
        print('Cache concept embeddings.')
        concepts = self.data_loader.get_concepts()
        self.model.eval(mode)
        for batch in tqdm.tqdm(concepts):
            self.model.set_cached_embeds(batch)
        # generate new labels from ensemble predictions by selected models
        candidates = []
        print('Generate new labels from ensemble predictions.')
        for batch in tqdm.tqdm(data):
            outputs = self.model.predict(batch, mode).detach().cpu().numpy()
            pred_classes, pred_probs = np.argmax(outputs, axis=1), np.max(outputs, axis=1)
            for i, (pred_class, pred_prob) in enumerate(zip(pred_classes, pred_probs)):
                if pred_prob > self.config.threshold:
                    candidates.append([pred_class, pred_prob, batch['origins'][i]])
        self.model.train(mode)
        candidates.sort(key=lambda x: x[1], reverse=True)
        change_num = [self.config.max_change_num]*self.config.num_classes
        n1, n2 = 0, 0
        for item in candidates:
            if item[2]['label'] != item[0] and change_num[item[0]] > 0:
                n1 += 1
                n2 += int(item[2]['label'] != -1)
                item[2]['label'] = item[0]
                change_num[item[0]] -= 1
        print('Total {} pairs reached the threshold, where we select {} unlabeled pairs change their labels, including {} pairs labeled in previous iterations.'.format(len(candidates), n1, n2))
        return n1 > 0
    
    def score_to_str(self, score):
        s = 'acc: {:.4f}, p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(score['acc'], score['p'], score['r'], score['f1'])
        return s
    
    def train_model(self, mode):
        # train each selected model seperately
        self.model.clear_models(mode)
        scores = []
        for i in range(self.config.ensemble_num(mode)):
            print('Ensemble {} model id {} train start.'.format(mode, i))
            train, valid, test = self.data_loader.get_training_data(mode)
            print('Batch data number ({} samples for train batch, {} samples for eval batch): train {}, valid {}, test {}'.format(self.config.batch_size('train'), self.config.batch_size('eval'), len(train), len(valid), len(test)))
            if mode == 'text':
                model = self.model.new_text_model()
            if mode == 'graph':
                model = self.model.new_graph_model()
            model.to(self.config.device)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=self.config.lr(mode), eps=1e-8)
            train_tqdm = tqdm.tqdm(range(self.config.max_epochs))
            train_tqdm.set_description('Epoch {} | train_loss: {:.4f} valid_loss: {:.4f}'.format(0, 0, 0))
            best_para, min_loss, patience = copy.deepcopy(model.state_dict()), 1e16, 0
            for epoch in train_tqdm:
                train_loss = 0.0
                for batch in train:
                    loss = self.train_one_step(batch, model, optimizer)
                    train_loss += loss
                train_loss /= len(train)
                valid_loss, score = self.evaluate(valid, model)
                train_tqdm.set_description('Epoch {} | train_loss: {:.4f} valid_loss: {:.4f}'.format(epoch, train_loss, valid_loss))
                if valid_loss < min_loss:
                    patience = 0
                    min_loss = valid_loss
                    best_para = copy.deepcopy(model.state_dict())
                patience += 1
                if patience > self.config.early_stop_time:
                    train_tqdm.close()
                    break
            model.load_state_dict(best_para)
            print('Train model {} finished, stop at {} epochs, min valid_loss {:.4f}'.format(i, epoch, min_loss))
            test_loss, score = self.evaluate(test, model)
            scores.append(score)
            print('Test model {} finished, test loss {:.4f},'.format(i, test_loss), self.score_to_str(score))
        average_score = {}
        for key in scores[0]:
            average_score[key] = sum([score[key] for score in scores])/len(scores)
        print('Train {} models finished, ensemble'.format(mode), self.score_to_str(average_score))
        return scores, average_score
    
    def get_concept_embeddings(self):
        #self.config.embeddings = self.data_loader.dataset.concept_embedding[:, :self.config.graph_embedding_dim].to(self.config.device)
        data = self.data_loader.get_concepts()
        self.model.eval('text')
        embeds = []
        for batch in data:
            embeds.append(self.model.get_text_embeddings(batch))
        embeds = torch.cat(embeds, 0)
        self.model.train('text')
        self.config.set_embeddings(embeds)
    
    def train(self):
        self.model = CoTrainingModels(self.config)
        score_text_list, score_graph_list = [], []
        for iteration in range(self.config.max_cotraining_iterations):
            print('Iteration {} start.'.format(iteration))
            print('Train text models.')
            score_texts, _ = self.train_model('text')
            score_text_list.append(score_texts)
            print('Get concept embeddings for graph models.')
            self.get_concept_embeddings()
            print('Train graph models.')
            score_graphs, _ = self.train_model('graph')
            score_graph_list.append(score_graphs)
            print('Generate new labels from text models.')
            flag_text = self.new_label('text')
            print('Generate new labels from graph models.')
            flag_graph = self.new_label('graph')
            if not flag_text and not flag_graph:
                print('Stop early, since no unlabeled samples are newly labeled after iteration {}'.format(iteration))
                break
        if flag_text | flag_graph:
            print('Stop normally after reaching the max_cotraining_iterations.')
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'wb') as f:
            para_text = self.model.state_dict('text')
            para_graph = self.model.state_dict('graph')
            torch.save([para_text, para_graph], f)
        with open('result/result.txt', 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj['score_texts'] = score_text_list
            obj['score_graphs'] = score_graph_list
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        self.model = CoTrainingModels(self.config)
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'rb') as f:
            para_text, para_graph = torch.load(f)
            self.model.load_state_dict(para_text, 'text')
            self.model.load_state_dict(para_graph, 'graph')
        self.model.to(self.config.device)
        print('Cache concept embeddings.')
        concepts = self.data_loader.get_concepts()
        self.model.eval()
        for mode in ['text', 'graph']:
            for batch in tqdm.tqdm(concepts):
                self.model.set_cached_embeds(batch)
        data = self.data_loader.get_prediction_data()
        self.get_concept_embeddings()
        print('Predict prerequisite relations between concepts.')
        self.model.eval()
        res = []
        for batch in tqdm.tqdm(data):
            with torch.no_grad():
                preds_text = self.model.predict(batch, 'text')
                preds_graph = self.model.predict(batch, 'graph')
            preds_text = preds_text.cpu().tolist()
            preds_graph = preds_graph.cpu().tolist()
            for i in range(len(batch['origins'])):
                datum = batch['origins'][i]
                obj = {'c1': datum['c1'], 'c2': datum['c2'], 'ground_truth': datum['ground_truth'], 'text_predict': preds_text[i], 'graph_predict': preds_graph[i]}
                res.append(obj)
        with open('result/predictions/{}.json'.format(self.config.store_name()), 'w', encoding='utf-8') as f:
            for obj in res:
                f.write(json.dumps(obj, ensure_ascii=False)+'\n')