import numpy as np
import numpy.random as random

class FairRW:
    def sample(self, G, length, sensitive_attribute = None, k=None, starting_node=None):
        if k is not None:
            nodes_with_degree_k = [node for node, degree in dict(G.degree()).items() if degree == k]
            G_ = G.subgraph(nodes_with_degree_k)
        else:
            G_ = G

        node = random.choice(G_.nodes()) if starting_node is None else starting_node

        sampled_nodes = []
        for i in range(length):
            sampled_nodes.append(node)
            neighbours = G_[node]
            if not neighbours:
                break
            num_neigh = len(G_[node])
            prob = np.ones(num_neigh)/num_neigh

            if sensitive_attribute != None:
                diff_attr = (sensitive_attribute != sensitive_attribute[node]).numpy()
                diff_neigh = diff_attr[neighbours]
                num_diff = diff_neigh.sum()
                if num_diff != 0 and num_diff != num_neigh:
                    prob = 0.5/num_diff * diff_neigh + 0.5/(num_neigh - num_diff) * (~diff_neigh)

            next_node = random.choice(G_[node], p=prob)
            node = next_node

        return sampled_nodes
    

class BiasedRW:
    def __init__(self, N, alpha):
        self.N = N
        self.prev_samples = np.zeros(N)
        self.alpha = alpha

    def sample(self, G, length, starting_node, beta, sensitive_attribute = None):
        already_sampled = np.zeros(self.N)
        sampled_nodes = []
        node = starting_node
        for i in range(length):
            sampled_nodes.append(node)
            neighbours = G[node]
            if not neighbours:
                break
            num_neigh = len(G[node])
            prob = np.ones(num_neigh)/num_neigh
            factor = self.alpha**self.prev_samples
            prob *= factor
            ##normalize 
            if sensitive_attribute != None:
                diff_attr = (sensitive_attribute != sensitive_attribute[node]).numpy()
                diff_neigh = diff_attr[neighbours]
                num_diff = diff_neigh.sum()
                if num_diff != 0 and num_diff != num_neigh:
                    prob = 0.5/num_diff * diff_neigh + 0.5/(num_neigh - num_diff) * (~diff_neigh)

            next_node = random.choice(G[node], p=prob)
            node = next_node

        return sampled_nodes

'''
class KHop:
    def sample(self, G, radius, K, sensitive_attribute = None, starting_node=None):
        if k is not None:
            nodes_with_degree_k = [node for node, degree in dict(G.degree()).items() if degree == k]
            G_ = G.subgraph(nodes_with_degree_k)
        else:
            G_ = G

        node = random.choice(G_.nodes()) if starting_node is None else starting_node

        sampled_nodes = []
        for i in range(radius):
            sampled_nodes.append(node)
            neighbours = G_[node]
            if not neighbours:
                break
            num_neigh = len(G_[node])
            prob = np.ones(num_neigh)/num_neigh

            if sensitive_attribute != None:
                diff_attr = (sensitive_attribute != sensitive_attribute[node]).numpy()
                diff_neigh = diff_attr[neighbours]
                num_diff = diff_neigh.sum()
                if num_diff != 0 and num_diff != num_neigh:
                    prob = 0.5/num_diff * diff_neigh + 0.5/(num_neigh - num_diff) * (~diff_neigh)

            next_node = random.choice(G_[node], p=prob)
            node = next_node

        return sampled_nodes
'''

    