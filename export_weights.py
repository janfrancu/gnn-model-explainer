import argparse
import os
import pickle
import random
import shutil
import time

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models

from train import train_node_classifier

def export_weights_and_prediction_npy(model, filename, G, labels):
    W1 = model.conv_first._parameters['weight'].detach().numpy()
    b1 = model.conv_first._parameters['bias'].detach().numpy()
    W2 = model.conv_block[0]._parameters['weight'].detach().numpy()
    b2 = model.conv_block[0]._parameters['bias'].detach().numpy()
    W3 = model.conv_last._parameters['weight'].detach().numpy()
    b3 = model.conv_last._parameters['bias'].detach().numpy()

    Wp = model.pred_model._parameters['weight'].detach().numpy()
    bp = model.pred_model._parameters['bias'].detach().numpy()


    data = gengraph.preprocess_input_graph(G, labels)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], dtype=torch.float)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    model.zero_grad()
    ypred, adj_att = model(x, adj)

    loss = model.loss(ypred, labels)
    loss.backward()

    gW1 = model.conv_first._parameters['weight'].grad.numpy()
    gb1 = model.conv_first._parameters['bias'].grad.numpy()
    gW2 = model.conv_block[0]._parameters['weight'].grad.numpy()
    gb2 = model.conv_block[0]._parameters['bias'].grad.numpy()
    gW3 = model.conv_last._parameters['weight'].grad.numpy()
    gb3 = model.conv_last._parameters['bias'].grad.numpy()

    gWp = model.pred_model._parameters['weight'].grad.numpy()
    gbp = model.pred_model._parameters['bias'].grad.numpy()
    

    loss = loss.detach().numpy()

    ypred = ypred.detach().numpy()
    adj_att = adj_att.detach().numpy()
    labels = labels.detach().numpy()

    np.savez(filename, W1=W1, W2=W2, W3=W3, Wp=Wp, 
                       b1=b1, b2=b2, b3=b3, bp=bp,
                       gW1=gW1, gW2=gW2, gW3=gW3, gWp=gWp, 
                       gb1=gb1, gb2=gb2, gb3=gb3, gbp=gbp,  
                       adj=adj, x=x, ypred=ypred, 
                       labels=labels, loss=loss)


def syn_task1(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)), seed=args.seed
    )
    num_classes = max(labels) + 1

    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    export_weights_and_prediction_npy(model, "./log/model_export/syn1:{}_init".format(args.seed), G, labels)

    train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy(model, "./log/model_export/syn1:{}_trained".format(args.seed), G, labels)


def syn_task2(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn2(seed=args.seed)
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1

    model = models.GcnEncoderNode(
        input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    export_weights_and_prediction_npy(model, "./log/model_export/syn2:{}_init".format(args.seed), G, labels)

    train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy(model, "./log/model_export/syn2:{}_trained".format(args.seed), G, labels)


def syn_task3(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn3(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)), seed=args.seed
    )
    print(labels)
    num_classes = max(labels) + 1
  
    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    export_weights_and_prediction_npy(model, "./log/model_export/syn3:{}_init".format(args.seed), G, labels)

    train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy(model, "./log/model_export/syn3:{}_trained".format(args.seed), G, labels)


def syn_task4(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn4(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)), seed=args.seed
    )
    print(labels)
    num_classes = max(labels) + 1

    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    export_weights_and_prediction_npy(model, "./log/model_export/syn4:{}_init".format(args.seed), G, labels)

    train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy(model, "./log/model_export/syn4:{}_trained".format(args.seed), G, labels)


def syn_task5(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn5(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)), seed=args.seed
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    model = models.GcnEncoderNode(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        num_classes,
        args.num_gc_layers,
        bn=args.bn,
        args=args,
    )

    export_weights_and_prediction_npy(model, "./log/model_export/syn5:{}_init".format(args.seed), G, labels)

    train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy(model, "./log/model_export/syn5:{}_trained".format(args.seed), G, labels)



# the data should be the same - export the adjacency matrix and others
args = prog_args = configs.arg_parse()

for seed in range(1):
    prog_args.seed = seed
    for i in range(5):
       fun = 'syn_task' + str(i+1)
       eval(fun)(prog_args)