# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:35:07 2023

@author: xingc
"""

from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np


#from layers import GraphConvolutionLayer

#from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP

import argparse

############copied from utils

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # print_log(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'px', 'py', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, px, py, allx, ally, adj = tuple(objects)
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx, px)).tolil()
    labels = np.vstack((ally, ty, py))
    # print(len(labels))

    train_idx_orig = parse_index_file(
        "./data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]
    pred_size = px.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    idx_pred = range(allx.shape[0]+ test_size, allx.shape[0] + test_size + pred_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    pred_mask = sample_mask(idx_pred, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_pred = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    y_pred[pred_mask, :] = labels[pred_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, y_pred, train_mask, val_mask, test_mask, pred_mask, train_size, test_size


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.A


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

'''
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print_log("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    # t_k.append(sp.eye(adj.shape[0]))
    # t_k.append(scaled_laplacian)
    t_k.append(sp.eye(adj.shape[0]).A)
    t_k.append(scaled_laplacian.A)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    # return sparse_to_tuple(t_k)
    return t_k
'''

def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print_log('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


import datetime
def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
      + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]
        
    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else: 
            print('[' + t + '] ' + str(line))
            
    
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class BERTGCN(nn.Module):
    def __init__(self, bert, opt):
        super(BERTGCN, self).__init__()
        self.opt = opt
        self.bert = bert
        self.gc1 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        self.gc4 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc5 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc6 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc7 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        #self.gc8 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)
        
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        
        text_bert_indices, bert_segments_ids, dependency_graph, affective_graph = inputs

        text_out, _ = self.bert(text_bert_indices, token_type_ids = bert_segments_ids, output_all_encoded_layers=False)

        x = F.relu(self.gc1(text_out, dependency_graph))
        x = F.relu(self.gc2(x, affective_graph))
        x = F.relu(self.gc3(text_out, dependency_graph))
        x = F.relu(self.gc4(x, affective_graph))
        #x = F.relu(self.gc5(text_out, dependency_graph))
        #x = F.relu(self.gc6(x, affective_graph))
        #x = F.relu(self.gc7(text_out, dependency_graph))
        #x = F.relu(self.gc8(x, affective_graph))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


##############

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GAT",
                        choices=["GCN", "SAGE", "GAT"],
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='require early stopping.')
    parser.add_argument('--dataset', type=str, default='twcvd2',
                        choices = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','tw', 'twcvd2','twcvd3'],
                        help='dataset to train')

    args, _ = parser.parse_known_args()
    #args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

args = get_citation_args()

# if len(sys.argv) != 2:
# 	sys.exit("Use: python train.py <dataset>")


#dataset = sys.argv[1]

# Set random seed
# seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device('cuda:0')

# Load data

adj, features, y_train, y_val, y_test, y_pred, train_mask, val_mask, test_mask, pred_mask, train_size, test_size = load_corpus('twcvd2')

features = sp.identity(features.shape[0])
features = preprocess_features(features)

def pre_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

adjdense = torch.from_numpy(pre_adj(adj).A.astype(np.float32))

def construct_graph(adjacency):
    g = DGLGraph()
    adj = pre_adj(adjacency)
    g.add_nodes(adj.shape[0])
    g.add_edges(adj.row,adj.col)
    adjdense = adj.A
    adjd = np.ones((adj.shape[0]))
    for i in range(adj.shape[0]):
        adjd[i] = adjd[i] * np.sum(adjdense[i,:])
    weight = torch.from_numpy(adj.data.astype(np.float32))
    g.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
    g.edata['w'] = weight

    if args.cuda:
        g.to(torch.device('cuda:0'))
    
    return g

class SimpleConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(SimpleConv, self).__init__()
        self.graph = g
        self.activation = activation
        #self.reset_parameters()
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        #self.b = nn.Parameter(torch.zeros(1, out_feats))
        #self.linear = nn.Linear(in_feats,out_feats)
        self.feat_drop = feat_drop
    
    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     nn.init.xavier_uniform_(self.linear.weight,gain=gain)
    
    def forward(self, feat):
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m',out='h'))
        rst = g.ndata['h']
        #rst = self.linear(rst)
        rst = self.activation(rst)
        return rst

class SAGEMeanConv(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation):
        super(SAGEMeanConv, self).__init__()
        self.graph = g
        self.feat_drop = nn.Dropout(0.5)
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        #self.linear = nn.Linear(in_feats, out_feats, bias=True)
        setattr(self, 'Wn', nn.Parameter(torch.randn(out_feats,out_feats)))
        self.activation = activation
        #self.neigh_linear = nn.Linear(out_feats, out_feats, bias=True)
        # self.reset_parameters()
    
    '''
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight,gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight,gain=gain)
    '''
    
    def forward(self,feat):
        g = self.graph.local_var()
        #feat = self.feat_drop(feat)
        h_self = feat.mm(getattr(self, 'W'))
        g.ndata['h'] = h_self
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
        h_neigh = g.ndata['neigh']
        degs = g.in_degrees().float()
        degs = degs.to(torch.device('cuda:0'))
        g.ndata['h'] = (h_neigh + h_self) / (degs.unsqueeze(-1) + 1)
        rst = g.ndata['h']
        rst = self.activation(rst)
        # rst = th.norm(rst)
        return rst

class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.graph = g
        setattr(self, 'W', nn.Parameter(torch.randn(in_feats,out_feats)))
        setattr(self, 'al', nn.Parameter(torch.randn(in_feats,1)))
        setattr(self, 'ar', nn.Parameter(torch.randn(in_feats,1)))

    def forward(self, feat):
        # equation (1)
        g = self.graph.local_var()
        g.ndata['h'] = feat.mm(getattr(self, 'W'))
        g.ndata['el'] = feat.mm(getattr(self, 'al'))
        g.ndata['er'] = feat.mm(getattr(self, 'ar'))
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        # message passing
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        e = F.leaky_relu(g.edata['e'])
        # compute softmax
        g.edata['w'] = F.softmax(e)
        rst = g.ndata['h']
        #rst = self.linear(rst)
        #rst = self.activation(rst)
        return rst

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, activation, num_heads=2, merge=None):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
        self.activation = activation

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            x = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            x = torch.mean(torch.stack(head_outs),dim=0)
        
        return self.activation(x)

class MultiLayer(nn.Module):
    def __init__(self,g,in_feats,out_feats,activation,feat_drop=True):
        super(MultiLayer, self).__init__()
        self.graph = g
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.reset_parameters()
        self.feat_drop = feat_drop
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight,gain=gain)

    def forward(self,feat):
        g = self.graph.local_var()
        if self.feat_drop:
            drop = nn.Dropout(0.5)
            feat = drop(feat)

        rst = self.linear(feat)
        rst = self.activation(rst)
        return rst

class Classifer(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super(Classifer, self).__init__()
        self.GCN = conv
        self.gcn1 = self.GCN(g,input_dim, 800, F.relu)
        self.gcn2 = self.GCN(g, 800, num_classes, F.relu)
    
    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)
        return x
    
class Classifer_Bert(nn.Module):
    def __init__(self,g,input_dim,num_classes,conv):
        super(Classifer, self).__init__()
        self.BERTGCN = conv
        self.gcn1 = self.BERTGCN(g,input_dim, 100, F.relu)
        self.gcn2 = self.BERTGCN(g, 100, num_classes, F.relu)
    
    def forward(self, features):
        x = self.gcn1(features)
        self.embedding = x
        x = self.gcn2(x)
        
        return x

g = construct_graph(adj)

# Define placeholders
t_features = torch.from_numpy(features.astype(np.float32))
t_features_tr = torch.from_numpy(features.astype(np.float32))
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
support = [preprocess_adj(adj)]
num_supports = 1
t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]))

if args.model == 'GCN':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SimpleConv)
elif args.model == 'SAGE':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=SAGEMeanConv)
elif args.model == 'GAT':
    model = Classifer(g,input_dim=features.shape[0], num_classes=y_train.shape[1],conv=MultiHeadGATLayer)
else:
    raise NotImplemented
# support has only one element, support[0] is adjacency
if args.cuda and torch.cuda.is_available():
    t_features = t_features.cuda()
    t_features_tr = t_features_tr.cuda()
    t_y_train = t_y_train.cuda()
    #t_y_val = t_y_val.cuda()
    #t_y_test = t_y_test.cuda()
    t_train_mask = t_train_mask.cuda()
    tm_train_mask = tm_train_mask.cuda()
    # for i in range(len(support)):
    #     t_support = [t.cuda() for t in t_support if True]
    model = model.cuda()

print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def evaluate(features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features).cpu()
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
        
    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

def prediction(features, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features).cpu()
        t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
        pred = torch.max(logits, 1)[1]
        
    return pred.numpy(), (time.time() - t_test)

val_losses = []

# Train model
for epoch in range(args.epochs):

    t = time.time()
    
    # Forward pass
    logits = model(t_features)
    loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])    
    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
    val_losses.append(val_loss)

    print_log("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

    if epoch > args.early_stopping and val_losses[-1] > np.mean(val_losses[-(args.early_stopping+1):-1]):
        print_log("Early stopping...")
        break


print_log("Optimization Finished!")


# Testing
test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))

test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(np.argmax(labels[i]))


print_log("Test Precision, Recall and F1-Score...")
print_log(metrics.classification_report(test_labels, test_pred, digits=4))
print_log("Macro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print_log("Micro average Test Precision, Recall and F1-Score...")
print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


# Transfer labelling
pred_lbl, pred_duration_tr = prediction(t_features, pred_mask)
print_log("Labelling: time= {:.5f}".format(pred_duration_tr))

pred_pred = []
for i in range(len(pred_mask)):
    if pred_mask[i]:
        pred_pred.append(pred_lbl[i])
       

import csv
with open('./data/corpus/' + args.dataset + '_pred.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(pred_pred)

# doc and word embeddings
tmp = model.embedding.cpu().numpy()
word_embeddings = tmp[train_size: adj.shape[0] - test_size]
train_doc_embeddings = tmp[:train_size]  # include val docs
test_doc_embeddings = tmp[adj.shape[0] - test_size:]

print_log('Embeddings:')
print_log('\rWord_embeddings:'+str(len(word_embeddings)))
print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings))) 
print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings))) 
print_log('\rWord_embeddings:') 
print(word_embeddings)

with open('./data/corpus/' + args.dataset + '_vocab.txt', 'r') as f:
    words = f.readlines()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
with open('./data/' + args.dataset + '_word_vectors.txt', 'w') as f:
    f.write(word_embeddings_str)

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
with open('./data/' + args.dataset + '_doc_vectors.txt', 'w') as f:
    f.write(doc_embeddings_str)

