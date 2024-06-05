import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw

# REAL_GRAPH = 'cora.pickle'
# GRAPHS = ['1000_ts_1000_gs.pickle', '1000_ts_2000_gs.pickle', '2000_ts_1000_gs.pickle', '2000_ts_2000_gs.pickle']
# FOCAL = ['1000_ts_1000_gs_focal.pickle', '1000_ts_2000_gs_focal.pickle']
# FIFTY = ['1000(50)_ts_1000_gs.pickle', '1000(50)_ts_2000_gs.pickle', '500(50)_ts_1000_gs.pickle', '500(50)_ts_2000_gs.pickle']
# TWENTY = ['1000(20)_ts_1000_gs.pickle', '1000(20)_ts_2000_gs.pickle', '1000(20)_ts_1000_gs_fair.pickle', '1000(20)_ts_2000_gs_fair.pickle']
# FIVEHUNDRED = ['1000(20)_ts_1000_gs_500.pickle', '1000(20)_ts_2000_gs_500.pickle']
# PERNODE = ['1_per_node(20)_ts_1000_gs.pickle', '1_per_node(20)_ts_2000_gs.pickle', '1_per_node(20)_ts_1000_gs_200.pickle', '1_per_node(20)_ts_2000_gs_200.pickle']
# SAGESS = ['sagess_real_for_real.pickle']
DATASETS = ['Cora', 'Facebook', 'NBA', 'Oklahoma97', 'UNC28']
MODELS = ['', 'fair', 'fair_focal', 'focal', 'graphrnn', 'cell', 'netgan']
MODELS = ['cell', 'graphrnn']


def read(file):
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
            if G.number_of_edges() >= 5278:
                break
    pickle.dump(G, open(f'sagess_real_for_real.pickle', 'wb'))
    return G

def eval(G, G_real):
    degrees = list(dict(G.degree()).values())
    avg_deg = sum(degrees)/len(degrees)
    triangles = nx.triangles(G)
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    degrees_np[degrees_np == 0] = np.finfo(np.float32).eps
    rel_edge_dist = -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())
    plaw = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    gini = avg_deg_diff / (np.mean(degrees)*2)
    print(f" & {G.number_of_nodes()} & {G.number_of_edges()} & {round(avg_deg,4)} & {int(sum(triangles.values())/3)} & {max(scc_size)} & {round(rel_edge_dist,4)} & {round(plaw.alpha, 4)} & {round(gini,4)} & {round(IoU(G, G_real),4)} \\\\")


def avg_deg(G):
    degrees = list(dict(G.degree()).values())
    return sum(degrees)/len(degrees)

def triangles(G):
    triangles = nx.triangles(G)
    return int(sum(triangles.values())/3)


def max_scc(G):
    scc = nx.connected_components(G)
    scc_size = [len(c) for c in scc]
    return max(scc_size)

def edge_dist_ent(G):
    degrees = list(dict(G.degree()).values())
    degrees_np = np.array(degrees, dtype=float)
    degrees_np /= 2*G.number_of_edges()
    degrees_np[degrees_np == 0] = np.finfo(np.float32).eps
    return -np.sum(degrees_np*np.log(degrees_np))/np.log(G.number_of_nodes())

def plaw(G):
    degrees = list(dict(G.degree()).values())
    pl = powerlaw.Fit(degrees, xmin=min(degrees), verbose=False)
    return pl.alpha

def gini(G):
    degrees = list(dict(G.degree()).values())
    avg_deg_diff = np.abs(np.subtract.outer(degrees, degrees)).mean()
    return avg_deg_diff / (np.mean(degrees)*2)


def fair_metric(Gp,Gp_bar, G_realp, G_realp_bar, metric):
    return np.abs(np.abs(metric(G_realp_bar) - metric(Gp_bar))/ metric(G_realp_bar) - np.abs(metric(G_realp) - metric(Gp))/ metric(G_realp))



def eval_fair(G, G_real, sensitive_attr):
    nodes = np.array(list(G))
    p = nodes[sensitive_attr[nodes] == 1]
    p_bar = nodes[sensitive_attr[nodes] == 0]
    Gp = G.subgraph(p)
    Gp_bar = G.subgraph(p_bar)

    nodes = np.array(list(G_real))
    p = nodes[sensitive_attr[nodes] == 1]
    p_bar = nodes[sensitive_attr[nodes] == 0]
    G_realp = G_real.subgraph(p)
    G_realp_bar = G_real.subgraph(p_bar)

    fair = lambda x : fair_metric(Gp, Gp_bar, G_realp, G_realp_bar, x)

    print(f"& {round(fair(avg_deg),4)} & {round(fair(triangles),4)} & {round(fair(max_scc),4)} & {round(fair(edge_dist_ent),4)} & {round(fair(plaw),4)} & {round(fair(gini),4)} \\\\")

    return



def IoU(G_gen, G_real):
    union = G_real.number_of_edges()
    intersection = 0
    for u,v in G_gen.edges():
        if G_real.has_edge(u,v):
            intersection += 1
        else:
            union += 1
    return intersection/union


print("\\begin{tabular}{ccccccccccc} \n \\hline")
print(" & \multicolumn{1}{c}{Method} & \
      \multicolumn{1}{c}{Nodes} & \
      \multicolumn{1}{c}{Edges} & \
      \multicolumn{1}{c}{Avg Deg} & \
      \multicolumn{1}{c}{TC} & \
      \multicolumn{1}{c}{LCC} & \
      \multicolumn{1}{c}{EDE} & \
      \multicolumn{1}{c}{PLE} & \
      \multicolumn{1}{c}{Gini} &\
      \multicolumn{1}{c}{IoU}\
      \\\\ \n \hline")

for dataset in DATASETS:
    path = "eval/real"
    G_real = read(f"{path}/{dataset}.pickle")
    sens_attr = read(f"{path}/{dataset}_sa.pickle")
    print("\\parbox[t]{2mm}{\\multirow{3}{*}{\\rotatebox[origin=c]{90}{", dataset, "}}} & Real ")
    eval(G_real, G_real)
    for model in MODELS:
        path = "eval/gen"
        G = read(f"{path}/{dataset}_{model}.pickle")
        print(f"&{model}", end="")
        eval(G, G_real)
    print("\\hline")

print("\nFAIRNESS\n")
for dataset in DATASETS:
    path = "eval/real"
    G_real = read(f"{path}/{dataset}.pickle")
    sens_attr = read(f"{path}/{dataset}_sa.pickle")
    for model in MODELS:
        path = "eval/gen"
        G = read(f"{path}/{dataset}_{model}.pickle")
        eval_fair(G, G_real, sens_attr)
    print("\\hline")



# G_real = read_graph(REAL_GRAPH)
# for graph in SAGESS:
#     print(graph)
#     G = read_graph(graph)
#     eval(G)
#     print(round(IoU(G, G_real),4))