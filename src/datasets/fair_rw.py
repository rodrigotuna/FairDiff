import numpy as np
import numpy.random as random

class FairRW:
    def sample(self, G, length, sensitive_attribute = None, k=None):
        if k is not None:
            nodes_with_degree_k = [node for node, degree in dict(G.degree()).items() if degree == k]
            G_ = G.subgraph(nodes_with_degree_k)
        else:
            G_ = G
        node = None
        while node is None or not G[node]:
            node = random.randint(0, G_.number_of_nodes())
        sampled_nodes = []
        for i in range(length):
            sampled_nodes.append(node)
            neighbours = G_[node]
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

    