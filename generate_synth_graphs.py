import numpy as np
import gengraph
import utils.featgen as featgen
import json

from networkx.readwrite import json_graph, write_gexf


def syn_task1(input_dim=10):
   return gengraph.gen_syn1(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
   )

def syn_task2():
   return gengraph.gen_syn2()

def syn_task3(input_dim=10):
   return gengraph.gen_syn3(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
   )

def syn_task4(input_dim=10):
   return gengraph.gen_syn4(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
   )

def syn_task5(input_dim=10):
   return gengraph.gen_syn5(
       feature_generator=featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
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


fun = "syn_task1"

for sample in range(10):
   for i in range(5):
       fun = 'syn_task' + str(i+1)
       G, labels, name = eval(fun)()
       convert_to_json(G, labels, 'synth_graphs/' + fun + '_' + name + '_' + str(sample+1))
