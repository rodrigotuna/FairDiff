import numpy.random as random

class FairRW:
    def sample(self, G, length): #sensitive_attribute, theta):
        node = random.randint(0, G.number_of_nodes())
        sampled_nodes = []

        for i in range(length):
            sampled_nodes.append(node)
            next_node = random.choice(G[node])
            node = next_node

        return list(dict.fromkeys(sampled_nodes))
    