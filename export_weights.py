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

### explainer specific
import explainer_main
from explainer import explain

###

def _get_model_weights(model):
    W1 = model.conv_first._parameters['weight'].detach().numpy()
    b1 = model.conv_first._parameters['bias'].detach().numpy()
    W2 = model.conv_block[0]._parameters['weight'].detach().numpy()
    b2 = model.conv_block[0]._parameters['bias'].detach().numpy()
    W3 = model.conv_last._parameters['weight'].detach().numpy()
    b3 = model.conv_last._parameters['bias'].detach().numpy()

    Wp = model.pred_model._parameters['weight'].detach().numpy()
    bp = model.pred_model._parameters['bias'].detach().numpy()

    return W1, b1, W2, b2, W3, b3, Wp, bp


def export_weights_and_prediction_npy(filename, model, G, labels, train_idx=None):
    W1, b1, W2, b2, W3, b3, Wp, bp = _get_model_weights(model)

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

    if train_idx:
        np.savez(filename, W1=W1, W2=W2, W3=W3, Wp=Wp, 
                           b1=b1, b2=b2, b3=b3, bp=bp,
                           gW1=gW1, gW2=gW2, gW3=gW3, gWp=gWp, 
                           gb1=gb1, gb2=gb2, gb3=gb3, gbp=gbp,  
                           adj=adj, x=x, ypred=ypred, 
                           labels=labels, loss=loss, train_idx=np.array(train_idx))
    else:
        np.savez(filename, W1=W1, W2=W2, W3=W3, Wp=Wp, 
                           b1=b1, b2=b2, b3=b3, bp=bp,
                           gW1=gW1, gW2=gW2, gW3=gW3, gWp=gWp, 
                           gb1=gb1, gb2=gb2, gb3=gb3, gbp=gbp,  
                           adj=adj, x=x, ypred=ypred, 
                           labels=labels, loss=loss)


def export_explainer_weights(filename, seed, node_idx, cg_dict, model):
    exp_args = explainer_main.arg_parse()
    writer = None
    graph_mode = False
    graph_idx = 0
    unconstrained = False

    explainer = explain.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=exp_args,
        writer=writer,
        print_training=False,
        graph_mode=graph_mode,
        graph_idx=graph_idx,
    )

    # every saved file should have these
    W1, b1, W2, b2, W3, b3, Wp, bp = _get_model_weights(model)

    # mimicking Explainer.explain method to construct ExplainModule
    node_idx_new, sub_adj, sub_feat, sub_label, neighbors = explainer.extract_neighborhood(node_idx, graph_idx)
    sub_label = np.expand_dims(sub_label, axis=0)
    sub_adj = np.expand_dims(sub_adj, axis=0)
    sub_feat = np.expand_dims(sub_feat, axis=0)

    adj   = torch.tensor(sub_adj, dtype=torch.float)
    x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
    label = torch.tensor(sub_label, dtype=torch.long)

    pred_label = np.argmax(explainer.pred[graph_idx][neighbors], axis=1)

    torch.random.manual_seed(seed) # fix seed for reproducibility
    explain_module = explain.ExplainModule(
            adj=adj,
            x=x,
            model=model,
            label=label,
            args=exp_args,
            writer=writer,
            graph_idx=graph_idx,
            graph_mode=graph_mode,
        )

    print("node label: ", explainer.label[graph_idx][node_idx])
    print("neigh graph idx: ", node_idx, node_idx_new)
    print("Node predicted label: ", pred_label[node_idx_new])

    ### grad method mask
    explain_module.zero_grad()
    adj_grad, feat_grad = explain_module.adj_feat_grad(node_idx_new, pred_label[node_idx_new])
    adj_grad = torch.abs(adj_grad)[graph_idx]
    edge_mask = adj_grad + adj_grad.t()
    edge_mask = nn.functional.sigmoid(edge_mask)
    edge_mask = edge_mask.cpu().detach().numpy() * sub_adj.squeeze()
    feat_mask = feat_grad.detach().numpy()

    ypred, _ = model(x, adj) # prediction on the subgraph, used for further checks
    ypred = ypred.detach().numpy()

    np.savez(filename + "_grad", node_idx=node_idx, node_idx_new=node_idx_new, neighbors=neighbors,
        pred_label=pred_label, ypred=ypred, sub_label=sub_label, sub_feat=sub_feat, sub_adj=sub_adj,
        edge_mask=edge_mask, feat_mask=feat_mask,
        W1=W1, W2=W2, W3=W3, Wp=Wp, b1=b1, b2=b2, b3=b3, bp=bp)

    ### explainer's loss and gradient (first iteration output only)
    # what do we need for the loss computation
    # ypred, pred_label, label, edge_mask, feat_mask, features and original adjacency
    explain_module.zero_grad()
    explain_module.optimizer.zero_grad()
    ypred, _ = explain_module(node_idx_new, unconstrained=unconstrained)
    

    loss = explain_module.loss(ypred, pred_label, node_idx_new, 0)
    loss.backward()

    edge_mask_init = explain_module._parameters['mask'].detach().numpy().copy()
    feat_mask_init = explain_module._parameters['feat_mask'].detach().numpy().copy()
    loss = loss.detach().numpy().copy()

    grad_edge_mask_init = explain_module._parameters['mask'].grad.numpy().copy()
    grad_feat_mask_init = explain_module._parameters['feat_mask'].grad.numpy().copy()

    ypred = ypred.detach().numpy().copy()
   
    masked_adj_init = explain_module._masked_adj().detach().numpy().copy()

    # optimize the weights for 100 iters
    for epoch in range(100):
        explain_module.zero_grad()
        explain_module.optimizer.zero_grad()
        _ypred, _adj_atts = explain_module(node_idx_new, unconstrained=unconstrained)
        _loss = explain_module.loss(_ypred, pred_label, node_idx_new, epoch)
        _loss.backward()
        explain_module.optimizer.step()

    masked_adj_trained = explain_module._masked_adj().detach().numpy().copy()

    np.savez(filename + "_exp", node_idx=node_idx, node_idx_new=node_idx_new, neighbors=neighbors,
        loss=loss, ypred=ypred, pred_label=pred_label,
        sub_label=sub_label, sub_feat=sub_feat, sub_adj=sub_adj,
        masked_adj_trained=masked_adj_trained, masked_adj_init=masked_adj_init, 
        edge_mask_init=edge_mask_init, feat_mask_init=feat_mask_init,
        grad_edge_mask_init=grad_edge_mask_init, grad_feat_mask_init=grad_feat_mask_init,
        W1=W1, W2=W2, W3=W3, Wp=Wp, b1=b1, b2=b2, b3=b3, bp=bp)


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

    export_weights_and_prediction_npy("./log/model_export/syn1:{}_init".format(args.seed), model, G, labels)

    cg_dict = train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy("./log/model_export/syn1:{}_trained".format(args.seed), model, G, labels, cg_dict["train_idx"])

    node_idx = 300
    export_explainer_weights("./log/model_export/syn1:{}_explain".format(args.seed), args.seed, node_idx, cg_dict, model)


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

    export_weights_and_prediction_npy("./log/model_export/syn2:{}_init".format(args.seed), model, G, labels)

    cg_dict = train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy("./log/model_export/syn2:{}_trained".format(args.seed), model, G, labels, cg_dict["train_idx"])

    node_idx = 300
    export_explainer_weights("./log/model_export/syn2:{}_explain".format(args.seed), args.seed, node_idx, cg_dict, model)


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

    export_weights_and_prediction_npy("./log/model_export/syn3:{}_init".format(args.seed), model, G, labels)

    cg_dict = train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy("./log/model_export/syn3:{}_trained".format(args.seed), model, G, labels, cg_dict["train_idx"])

    node_idx = 301
    export_explainer_weights("./log/model_export/syn3:{}_explain".format(args.seed), args.seed, node_idx, cg_dict, model)


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

    export_weights_and_prediction_npy("./log/model_export/syn4:{}_init".format(args.seed), model, G, labels)

    cg_dict = train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy("./log/model_export/syn4:{}_trained".format(args.seed), model, G, labels, cg_dict["train_idx"])

    node_idx = 511
    export_explainer_weights("./log/model_export/syn4:{}_explain".format(args.seed), args.seed, node_idx, cg_dict, model)



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

    export_weights_and_prediction_npy("./log/model_export/syn5:{}_init".format(args.seed), model, G, labels)

    cg_dict = train_node_classifier(G, labels, model, args, writer=writer)
    export_weights_and_prediction_npy("./log/model_export/syn5:{}_trained".format(args.seed), model, G, labels, cg_dict["train_idx"])

    node_idx = 512
    export_explainer_weights("./log/model_export/syn5:{}_explain".format(args.seed), args.seed, node_idx, cg_dict, model)


# the data should be the same - export the adjacency matrix and others
args = prog_args = configs.arg_parse()

for seed in range(5):
    prog_args.seed = seed
    for i in range(5):
       fun = 'syn_task' + str(i+1)
       eval(fun)(prog_args)