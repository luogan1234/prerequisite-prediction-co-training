import numpy as np
import torch
from sklearn.decomposition import PCA

class Config:
    def __init__(self, dataset, text_encoder, graph_layer, init_num, max_change_num, seed, cpu):
        self.dataset = dataset
        self.text_encoder = text_encoder
        self.graph_layer = graph_layer
        self.init_num = init_num
        self.max_change_num = max_change_num
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        if dataset in ['moocen']:
            self.language = 'en'
            self.vocab_num = 30522  # bert-base-uncased
        if dataset in ['mooczh', 'cs', 'psy', 'math', 'phy', 'chem']:
            self.language = 'zh'
            self.vocab_num = 21128  # bert-base-chinese
        assert self.language, 'Need to provide the language information for new datasets'
        
        self.max_term_length = 10
        self.word_embedding_dim = 32
        self.attention_dim = 32
        self.text_embedding_dim = 768 if text_encoder in ['bert', 'bert-freeze'] else 128
        self.graph_embedding_dim = 128
        self.encoding_dim = 64
        self.ensemble_num = 8
        self.max_cotraining_iterations = 10
        self.max_epochs = 64
        self.early_stop_time = 8
        self.num_classes = 2
        self.threshold = 0.75
    
    def lr(self, mode):
        if mode == 'text' and self.text_encoder == 'bert':
            lr = 4e-5
        else:
            lr = 1e-3
        return lr
    
    def batch_size(self, mode):
        if mode == 'train':
            batch_size = 16
        else:
            batch_size = 64
            if self.text_encoder in ['lstm', 'bert-freeze']:
                batch_size *= 4
        return batch_size
    
    def set_concepts_parameters(self, concepts):
        self.concept_num = len(concepts)
    
    def set_gcn_parameters(self, graph):
        self.laplacian1 = self.to_laplacian_matrix(graph.T)
        self.laplacian2 = self.to_laplacian_matrix(graph)
    
    def set_embeddings(self, embeds):
        pca = PCA()
        X = embeds.detach().cpu().numpy()
        X = pca.fit_transform(X)
        embeds = torch.from_numpy(X)
        embeds = embeds[:, :self.graph_embedding_dim]
        size = embeds.size()
        padding = torch.zeros(size[0], self.graph_embedding_dim-size[1])
        self.embeddings = torch.cat([embeds, padding], 1).to(self.device)
    
    def to_laplacian_matrix(self, graph):
        a = np.eye(graph.shape[0]) + graph
        d = np.power(np.sum(np.abs(a), 1), -1)
        d[np.isinf(d)] = 0
        d = np.diag(d)
        laplacian = np.array(np.matmul(d, a), dtype=np.float32)
        laplacian = torch.from_numpy(laplacian).to(self.device)
        return laplacian
    
    def store_name(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.dataset, self.text_encoder, self.graph_layer, self.init_num, self.max_change_num, self.seed)
    
    def parameter_info(self):
        obj = {'dataset': self.dataset, 'text_encoder': self.text_encoder, 'graph_layer': self.graph_layer, 'init_num': self.init_num, 'max_change_num': self.max_change_num, 'seed': self.seed}
        return obj