import random


def connect_nodes(n1, n2):
    n1.add_neighbour(n2)
    n2.add_neighbour(n1)

def sample_t_local_faulty_set(adj: dict[int, set[int]], t, p_try=0.3, seed=None):

    rng = random.Random(seed)
    nodes = list(adj.keys())
    B = set()
    def violates(candidate, Bset):
        Bp = Bset | {candidate}
        for u, Nu in adj.items():
            if len(Bp & Nu) > t:
                return True
        return False

    for v in rng.sample(nodes, k=len(nodes)):
        if rng.random() < p_try and not violates(v, B):
            B.add(v)
    return B

