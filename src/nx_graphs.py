import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw

REAL_GRAPH = 'cora.pickle'
GRAPHS = ['1000_ts_1000_gs.pickle', '1000_ts_2000_gs.pickle', '2000_ts_1000_gs.pickle', '2000_ts_2000_gs.pickle']

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
    pickle.dump(G, open(f'2000_ts_{num_samples}_gs.pickle', 'wb'))
    return G

def eval(G):
    degrees = list(dict(G.degree()).values())
    print("Avg degree: ", sum(degrees)/len(degrees))
    triangles = nx.triangles(G)
    print("Triangles: ", sum(triangles.values())/3)
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    print("Largest CC", max(scc_size))
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    print("Relative edge distribution entropy", -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes()))
    plaw = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)  #from Sagess paper it is this
    print("Power Law Exponent", plaw.alpha)
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    gini = avg_deg_diff / (np.mean(degrees)*2)
    print("Gini coefficient", gini)
    print(G)

def IoU(G_gen, G_real):
    union = G_real.number_of_edges()
    print(union)
    intersection = 0
    for u,v in G_gen.edges():
        if G_real.has_edge(u,v):
            intersection += 1
        else:
            union += 1
    return intersection/union







G_real = read_graph(REAL_GRAPH)
for graph in GRAPHS:
    print(graph)
    G = read_graph(graph)
    print(IoU(G, G_real))