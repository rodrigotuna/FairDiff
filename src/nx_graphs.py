import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
import powerlaw

DATASETS = ['Cora', 'Facebook', 'NBA', 'Oklahoma97', 'UNC28']
MODELS = ['fair', 'fair_focal','', 'focal', 'graphrnn', 'cell', 'netgan']
PRESENT_NAME = {
    'fair': "Fair-SaGess", 
    'fair_focal':"Fair-SaGess FL",
    '': "SaGess", 
    'focal': "SaGess FL", 
    'graphrnn': "GraphRNN", 
    'cell': "CELL", 
    'netgan': "NetGAN"
}


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
    max_deg = max(degrees)
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
    print(f" & {G.number_of_nodes()} & {G.number_of_edges()} & {round(max_deg,4)} & {int(sum(triangles.values())/3)} & {round(rel_edge_dist,4)}  & {round(gini,4)} & {round(clusterCoeff(G),4)} & {round(assortativity(G),4)} & {round(IoU(G, G_real),4)} \\\\")


def max_deg(G):
    degrees = list(dict(G.degree()).values())
    return max(degrees)

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

def clusterCoeff(G):
    return nx.average_clustering(G)

def assortativity(G):
    return nx.degree_assortativity_coefficient(G)


def fair_metric(Gp,Gp_bar, G_realp, G_realp_bar, metric):
    return np.abs(np.abs((metric(G_realp_bar) - metric(Gp_bar))/ metric(G_realp_bar)) - np.abs((metric(G_realp) - metric(Gp))/ metric(G_realp)))



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

    print(f"& {round(fair(avg_deg),4)} & {round(fair(max_deg),4)} & {round(fair(max_scc),4)} & {round(fair(edge_dist_ent),4)} & {round(fair(plaw),4)} & {round(fair(gini),4)} \\\\")

    return [round(fair(avg_deg),4), round(fair(max_deg),4) , round(fair(max_scc),4), round(fair(edge_dist_ent),4), round(fair(plaw),4), round(fair(gini),4), round(fair(triangles),4), round(fair(clusterCoeff),4), round(fair(assortativity),4)]


#see node level metrics in graph and then split. and then compare avg with cluster, degeee connectivity, compare mmd? centrality. if this is better than sagess i gues.s 

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist = np.linalg.norm(x - y) ** 2
    return np.exp(-beta * dist)

def compute_mmd(X, Y, kernel=gaussian_kernel, sigma=1.0):
    # Number of samples
    m = len(X)
    n = len(Y)
    
    # Compute kernel values
    K_XX = np.zeros((m, m))
    K_YY = np.zeros((n, n))
    K_XY = np.zeros((m, n))
    
    for i in range(m):
        for j in range(m):
            K_XX[i, j] = kernel(X[i], X[j], sigma)
    
    for i in range(n):
        for j in range(n):
            K_YY[i, j] = kernel(Y[i], Y[j], sigma)
    
    for i in range(m):
        for j in range(n):
            K_XY[i, j] = kernel(X[i], Y[j], sigma)
    
    # Calculate MMD
    mmd = (1.0 / (m * (m - 1)) * np.sum(K_XX - np.diag(np.diagonal(K_XX))) +
           1.0 / (n * (n - 1)) * np.sum(K_YY - np.diag(np.diagonal(K_YY))) -
           2.0 / (m * n) * np.sum(K_XY))
    
    return mmd

def eval_fair2(G, G_real, sensitive_attr):
    nodes = np.array(list(G))
    nodes_real = np.array(list(G_real))
    deg_real = dict(G_real.degree())
    deg = dict(G.degree())
    coeff_real = nx.clustering(G_real)
    coeff = nx.clustering(G)
    centr_real = nx.eigenvector_centrality(G_real)
    centr = nx.eigenvector_centrality(G)

    p_nodes_real = nodes_real[sensitive_attr[nodes_real] == 1]
    p_bar_nodes_real = nodes_real[sensitive_attr[nodes_real] == 0]

    p_nodes = nodes[sensitive_attr[nodes] == 1]
    p_bar_nodes = nodes[sensitive_attr[nodes] == 0]

    deg_real_p = [deg_real[i] for i in p_nodes_real]
    deg_real_pbar = [deg_real[i] for i in p_bar_nodes_real]

    deg_p = [deg[i] for i in p_nodes]
    deg_pbar = [deg[i] for i in p_bar_nodes]

    coeff_real_p = [coeff_real[i] for i in p_nodes_real]
    coeff_real_pbar = [coeff_real[i] for i in p_bar_nodes_real]

    coeff_p = [coeff[i] for i in p_nodes]
    coeff_pbar = [coeff[i] for i in p_bar_nodes]

    centr_real_p = [centr_real[i] for i in p_nodes_real]
    centr_real_pbar = [centr_real[i] for i in p_bar_nodes_real]

    centr_p = [centr[i] for i in p_nodes]
    centr_pbar = [centr[i] for i in p_bar_nodes]

    deg_mes = np.abs(compute_mmd(deg_real_p, deg_p) - compute_mmd(deg_real_pbar, deg_pbar))
    coeff_mes = np.abs(compute_mmd(coeff_real_p, coeff_p) - compute_mmd(coeff_real_pbar, coeff_pbar))
    centr_mes = np.abs(compute_mmd(centr_real_p, centr_p) - compute_mmd(centr_real_pbar, centr_pbar))

    return [deg_mes, coeff_mes, centr_mes]




def IoU(G_gen, G_real):
    union = G_real.number_of_edges()
    intersection = 0
    for u,v in G_gen.edges():
        if G_real.has_edge(u,v):
            intersection += 1
        else:
            union += 1
    return intersection/union

real = []
metric = []

# print("\\begin{tabular}{cccc|ccccccc} \n \\hline")
# print(" & \multicolumn{1}{c}{Method} & \
#       \multicolumn{1}{c}{Nodes} & \
#       \multicolumn{1}{c}{Edges} & \
#       \multicolumn{1}{c}{Max Deg} & \
#       \multicolumn{1}{c}{TC} & \
#       \multicolumn{1}{c}{EDE} & \
#       \multicolumn{1}{c}{Gini} &\
#       \multicolumn{1}{c}{CC} &\
#       \multicolumn{1}{c}{Assort.} &\
#       \multicolumn{1}{c}{IoU}\
#       \\\\ \n \hline")
# for dataset in DATASETS:
#     path = "eval/real"
#     G_real = read(f"{path}/{dataset}.pickle")
#     sens_attr = read(f"{path}/{dataset}_sa.pickle")
#     print("\\parbox[t]{2mm}{\\multirow{8}{*}{\\rotatebox[origin=c]{90}{", dataset, "}}} & Real ")
#     eval(G_real, G_real)
#     for model in MODELS:
#         path = "eval/gen"
#         G = read(f"{path}/{dataset}_{model}.pickle")
#         print(f"&{PRESENT_NAME[model]}", end="")
#         eval(G, G_real)
#     print("\\hline")


# metric = np.array(metric)
# metric = np.transpose(metric, (2, 1, 0))
# print(metric.shape)

# N = 6
# #TRIANGLE COUNT IS OUT 1.
# #max DEGREE IS IN 6.
# #

# for i in range(7):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     w = 0.10
#     ind = np.arange(5)
#     for j in range(metric[i].shape[0]):
#         ax.bar(ind + j * w, metric[i][j],w)

#     plt.savefig(f"{i}.pdf")
#     plt.cla


print("\nFAIRNESS\n")


# print("\\begin{tabular}{ccccccc} \n \\hline")
# print(" & \multicolumn{1}{c}{Method} & \
#       \multicolumn{1}{c}{Max Deg} & \
#       \multicolumn{1}{c}{TC} & \
#       \multicolumn{1}{c}{LCC} & \
#       \multicolumn{1}{c}{EDE} & \
#       \multicolumn{1}{c}{PLE} & \
#       \multicolumn{1}{c}{Gini} \
#       \\\\ \n \hline")


metric = []

for dataset in DATASETS:
    dataset_m = []
    path = "eval/real"
    G_real = read(f"{path}/{dataset}.pickle")
    sens_attr = read(f"{path}/{dataset}_sa.pickle")
    # print("\\parbox[t]{2mm}{\\multirow{7}{*}{\\rotatebox[origin=c]{90}{", dataset, "}}}")
    for model in MODELS:
        print(f"& {model} ")
        path = "eval/gen"
        G = read(f"{path}/{dataset}_{model}.pickle")
        dataset_m.append(eval_fair2(G, G_real, sens_attr))
    # print("\\hline")

    metric.append(dataset_m)
print("\end{tabular}")

metric = np.array(metric)
metric = np.transpose(metric, (2, 1, 0))
print(metric.shape)

N = 6
#TRIANGLE COUNT IS OUT 1.
#max DEGREE IS IN 6.
#

for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    w = 0.10
    ind = np.arange(5)
    for j in range(4):
        ax.bar(ind + j * w, metric[i][j],w)

    plt.savefig(f"mmd{i}.pdf")
    plt.cla


# G_real = read_graph(REAL_GRAPH)
# for graph in SAGESS:
#     print(graph)
#     G = read_graph(graph)
#     eval(G)
#     print(round(IoU(G, G_real),4))