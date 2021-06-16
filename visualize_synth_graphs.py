import numpy as np
import gengraph
import utils.synthetic_structsim as synthetic_structsim
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

def export_gexf(G, labels, name):
    for i, l in enumerate(labels):
      del G.nodes[i]['feat']
      G.nodes[i]['ytrue'] = int(l)

    write_gexf(G, './dump/sample_' + fun + '_' + name + '.gexf')

for i in range(5):
  fun = 'syn_task' + str(i+1)
  G, labels, name = eval(fun)()
  export_gexf(G, labels, name)

