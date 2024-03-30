import numpy as np
import numpy.random as random

class FairRW:
    def sample(self, G, length, sensitive_attribute = None):
        node = random.randint(0, G.number_of_nodes())
        sampled_nodes = []
        for i in range(length):
            sampled_nodes.append(node)
            neighbours = G[node]
            num_neigh = len(G[node])
            prob = np.ones(num_neigh)/num_neigh

            if sensitive_attribute != None:
                diff_attr = (sensitive_attribute != sensitive_attribute[node]).numpy()
                diff_neigh = diff_attr[neighbours]
                num_diff = diff_neigh.sum()
                if num_diff != 0 and num_diff != num_neigh:
                    prob = 0.5/num_diff * diff_neigh + 0.5/(num_neigh - num_diff) * (~diff_neigh)

            next_node = random.choice(G[node], p=prob)
            node = next_node

        return sampled_nodes

    