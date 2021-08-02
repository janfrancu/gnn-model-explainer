import os
import numpy as np
import gengraph
import utils.featgen as featgen
import json

from networkx.readwrite import json_graph, write_gexf


def syn_task1(input_dim=10, seed=0):
   return gengraph.gen_syn1(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float)), seed=seed 
   )

def syn_task2(seed=0):
   return gengraph.gen_syn2(seed=seed)

def syn_task3(input_dim=10, seed=0):
   return gengraph.gen_syn3(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float)), seed=seed
   )

def syn_task4(input_dim=10, seed=0):
   return gengraph.gen_syn4(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float)), seed=seed
   )

def syn_task5(input_dim=10, seed=0):
   return gengraph.gen_syn5(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float)), seed=seed
   )

def convert_to_json(G, labels, name):
   # convert numpy arrays to float and add label information
   # part of the label list is filled with np.int64
   for i, l in enumerate(labels):
       G.nodes[i]['feat'] = G.nodes[i]['feat'].tolist()
       G.nodes[i]['label'] = int(l)

   data = json_graph.node_link_data(G)

   # convert all node ids from int64 to int
   for i in range(len(data['nodes'])):
       data['nodes'][i]['id'] = int(data['nodes'][i]['id'])

   # convert all links ids from int64 to int
   for i in range(len(data['links'])):
       data['links'][i]['source'] = int(data['links'][i]['source'])
       data['links'][i]['target'] = int(data['links'][i]['target'])

   # node link JSON
   with open(name + '.json', 'w') as file:
       file.write(json.dumps(data))

# comes from `train_node_classifier` method
def store_traintest_split(G, train_ratio, seed):
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    # print(train_idx[:10])

    for n in train_idx:
        G.nodes[n]['split'] = "train"

    return G


outdir = "./synth_graphs"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

train_ratio = 0.8
for seed in range(20):
   for i in range(5):
       fun = 'syn_task' + str(i+1)
       G, labels, name = eval(fun)(seed=seed)
       store_traintest_split(G, train_ratio, seed)
       l = np.array(labels)
       # print(np.nonzero(l > 0))
       convert_to_json(G, labels, 'synth_graphs/' + fun + '_' + name + '_' + str(seed))


### check reproducibility
# import networkx as nx

# for i in range(5):
#   fun = 'syn_task' + str(i+1)

#   G1, l1, _ = eval(fun)(seed=1)
#   G2, l2, _ = eval(fun)(seed=1)

#   d1, d2 = gengraph.preprocess_input_graph(G1, l1), gengraph.preprocess_input_graph(G2, l2)

#   print(np.all(d1["adj"] == d2["adj"]))
#   print(np.all(d1["feat"] == d2["feat"]))
#   print(np.all(d1["labels"] == d2["labels"]))

# for i in range(5):
#   fun = 'syn_task' + str(i+1)

#   G1, l1, _ = eval(fun)(seed=1)
#   G2, l2, _ = eval(fun)(seed=2)

#   d1, d2 = gengraph.preprocess_input_graph(G1, l1), gengraph.preprocess_input_graph(G2, l2)

#   print(np.all(d1["adj"] == d2["adj"]))
#   print(np.all(d1["feat"] == d2["feat"]))
#   print(np.all(d1["labels"] == d2["labels"]))