import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import torch

REAL_GRAPH = 'cora.pickle'
GRAPHS = ['1000_ts_1000_gs.pickle', '1000_ts_2000_gs.pickle', '2000_ts_1000_gs.pickle', '2000_ts_2000_gs.pickle']
FOCAL = ['1000_ts_1000_gs_focal.pickle', '1000_ts_2000_gs_focal.pickle']
FIFTY = ['1000(50)_ts_1000_gs.pickle', '1000(50)_ts_2000_gs.pickle', '500(50)_ts_1000_gs.pickle', '500(50)_ts_2000_gs.pickle']
TWENTY = ['1000(20)_ts_1000_gs.pickle', '1000(20)_ts_2000_gs.pickle', '1000(20)_ts_1000_gs_fair.pickle', '1000(20)_ts_2000_gs_fair.pickle']

def read_graph(file):
    G = pickle.load(open(file, 'rb'))
    return G

def create_graph(file, num_samples):
    G = nx.Graph()
    with open(file, "r") as f:
        for i in range(num_samples):
            s = f.readline()
            N = int(s[2:])
            s = f.readline()
            s = f.readline()
            nodes = s.split(' ')[:-1]
            nodes = [int(v) for v in nodes]
            s = f.readline()
            for i in range(N):
                l = f.readline().split(' ')[:-1]
                for j in range(N):
                    if(l[j]) == '1':
                        G.add_edge(nodes[i], nodes[j])

            s = f.readline()
    pickle.dump(G, open(f'1000(20)_ts_{num_samples}_gs_fair.pickle', 'wb'))
    return G

def eval(G):
    degrees = list(dict(G.degree()).values())
    avg_deg = sum(degrees)/len(degrees)
    triangles = nx.triangles(G)
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    rel_edge_dist = -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())
    plaw = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    gini = avg_deg_diff / (np.mean(degrees)*2)
    print(f"| |{G.number_of_nodes()}|{G.number_of_edges()}|{round(avg_deg,2)}|{int(sum(triangles.values())/3)}|{max(scc_size)}|{round(rel_edge_dist,4)}|{round(plaw.alpha, 4)}|{round(gini,4)}| |")



def IoU(G_gen, G_real):
    union = G_real.number_of_edges()
    intersection = 0
    for u,v in G_gen.edges():
        if G_real.has_edge(u,v):
            intersection += 1
        else:
            union += 1
    return intersection/union

# create_graph("generated_subgraphs/generated_samples19.txt", 1000)
# create_graph("generated_subgraphs/generated_samples19.txt", 2000)

# G = read_graph('latent4.pickle')
# eval(G)
#2 35 | |7817|16876|4.32|6566|7043|0.9442|1.8864|0.4335| | not sure what is this
# 2 35 |6493|15265|4.7|8615|6086|0.9258|1.8776|0.4812| | 35 seems to be best for this

samples = pickle.load(open('sample_list4.pickle', 'rb'))
print(len(samples))
G = nx.Graph()
nodes = []
for id,sample in enumerate(samples):
    print(f"{id}/1000")
    if id >= 1000:
        break
    embeddings = sample[0]
    adj = sample[1]
    nodeids = []
    for embedding in embeddings: 
        id = None
        for idx, node in enumerate(nodes):
            if torch.norm(embedding - node) < 0.35:
                id = idx
                break
        if not id:
            id = len(nodes)
            nodes.append(embedding)
        nodeids.append(id)

    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if adj[i][j] == 1:
                G.add_edge(nodeids[i], nodeids[j])


eval(G)

# G_real = read_graph(REAL_GRAPH)
# for graph in FIFTY + TWENTY:
#     print(graph)
#     G = read_graph(graph)
#     eval(G)
#     print(round(IoU(G, G_real),4))



# from torch_geometric.datasets import Planetoid
# import networkx as nx
# import numpy as np

# graph = Planetoid("../data","Cora").get(0)
# sensitive_attribute = graph.y.detach().clone()

# print(graph.y.unique(return_counts=True))

# from torch_geometric.utils import to_networkx

# G = to_networkx(graph, to_undirected=True)

# x = nx.eigenvector_centrality(G)
# x_by_class = np.zeros(7)

# for i in range(2708):
#     x_by_class[sensitive_attribute[i]] += x[i]**2

# print(x_by_class)


## 17 - size 50 500 subgraphs
## 18 - size 20 1000 subgraphs
## 19 - size 20 1000 subgraphs fair
## 20 - size 20 2000 subgraphs 500 epochs