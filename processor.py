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
    
    def new_label(self, data, mode):
        # generate new labels from ensemble predictions by selected models
        self.model.eval(mode)
        candidates = []
        print('Generate new labels from ensemble predictions.')
        for batch in tqdm.tqdm(data):
            outputs = self.model(batch, mode).detach().cpu().numpy()
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
        s = 'acc: {:.3f}, p: {:.3f}, r: {:.3f}, f1: {:.3f}'.format(score['acc'], score['p'], score['r'], score['f1'])
        return s
    
    def train_model(self, mode):
        # train each selected model seperately
        self.model.clear_models(mode)
        average_test_loss, scores = 0.0, []
        for i in range(self.config.ensemble_num):
            print('Ensemble {} model id {} train start.'.format(mode, i))
            train, unlabeled, eval, test = self.data_loader.get_train(mode)
            print('Batch data number ({} samples for train batch, {} samples for eval batch): train {}, (original) unlabeled {}, eval {}, test {}'.format(self.config.batch_size('train'), self.config.batch_size('eval'), len(train), len(unlabeled), len(eval), len(test)))
            if mode == 'text':
                model = self.model.new_text_model()
            if mode == 'graph':
                model = self.model.new_graph_model()
            model.to(self.config.device)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=self.config.lr(mode))
            train_tqdm = tqdm.tqdm(range(self.config.max_epochs))
            train_tqdm.set_description('Epoch {} | train_loss: {:.3f} eval_loss: {:.3f}'.format(0, 0, 0))
            best_para, min_loss, patience = model.state_dict(), 1e16, 0
            for epoch in train_tqdm:
                train_loss = 0.0
                for batch in train:
                    loss = self.train_one_step(batch, model, optimizer)
                    train_loss += loss
                train_loss /= len(train)
                eval_loss, score = self.evaluate(eval, model)
                train_tqdm.set_description('Epoch {} | train_loss: {:.3f} eval_loss: {:.3f}'.format(epoch, train_loss, eval_loss))
                if eval_loss < min_loss:
                    patience = 0
                    min_loss = eval_loss
                    best_para = model.state_dict()
                patience += 1
                if patience > self.config.early_stop_time:
                    train_tqdm.close()
                    break
            print('Train finished, stop at {} epochs, min eval_loss {:.3f}'.format(epoch, min_loss))
            test_loss, score = self.evaluate(test, model)
            average_test_loss += test_loss
            scores.append(score)
            print('Test finished, test loss {:.3f},'.format(test_loss), self.score_to_str(score))
        flag = self.new_label(unlabeled, mode)
        average_test_loss /= self.config.ensemble_num
        average_score = {}
        for key in scores[0]:
            average_score[key] = round(sum([score[key] for score in scores])/len(scores), 3)
        print('Train {} models finished, average test loss {:.3f},'.format(mode, average_test_loss), self.score_to_str(average_score))
        return average_test_loss, average_score, flag
    
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
        min_loss_text, min_loss_graph = 1e16, 1e16
        for iteration in range(self.config.max_cotraining_iterations):
            print('Iteration {} start.'.format(iteration))
            print('Train text models.')
            loss_text, score_text, flag_text = self.train_model('text')
            print('Get concept embeddings.')
            self.get_concept_embeddings()
            print('Train graph models.')
            loss_graph, score_graph, flag_graph = self.train_model('graph')
            if loss_text < min_loss_text:
                best_para_text = self.model.state_dict('text')
                min_loss_text = loss_text
                best_score_text = score_text
            if loss_graph < min_loss_graph:
                best_para_graph = self.model.state_dict('graph')
                min_loss_graph = loss_graph
                best_score_graph = score_graph
            if not flag_text and not flag_graph:
                print('Stop early, since no samples in train_unlabeled are new labeled after iteration {}'.format(iteration))
                break
        if flag_text | flag_graph:
            print('Stop normally after reaching the max_cotraining_iterations.')
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'wb') as f:
            torch.save([best_para_text, best_para_graph], f)
        with open('result/result.txt', 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj['score_text'] = best_score_text
            obj['score_graph'] = best_score_graph
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        self.model = CoTrainingModels(self.config)
        data = self.data_loader.get_predict()
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'rb') as f:
            best_para_text, best_para_graph = torch.load(f)
            self.model.load_state_dict(best_para_text, 'text')
            self.model.load_state_dict(best_para_graph, 'graph')
        self.model.to(self.config.device)
        self.get_concept_embeddings()
        self.model.eval()
        res = []
        for batch in tqdm.tqdm(data):
            with torch.no_grad():
                preds_text = self.model(batch, 'text')
                preds_graph = self.model(batch, 'graph')
            preds_text = preds_text.cpu().tolist()
            preds_graph = preds_graph.cpu().tolist()
            for i in range(len(batch['origins'])):
                datum = batch['origins'][i]
                obj = {'c1': datum['c1'], 'c2': datum['c2'], 'ground_truth': datum['ground_truth'], 'text_predict': preds_text[i], 'graph_predict': preds_graph[i]}
                res.append(obj)
        with open('result/predictions/{}.json'.format(self.config.store_name()), 'w', encoding='utf-8') as f:
            for obj in res:
                f.write(json.dumps(obj, ensure_ascii=False)+'\n')