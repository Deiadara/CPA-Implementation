from node import Node, connect_nodes
import networkx as nx

def nodes_from_networkx(G):
    nodes = {i: Node(i, behavior=None) for i in G.nodes()}
    for u, v in G.edges():
        connect_nodes(nodes[u], nodes[v])
    return nodes

def build_line_graph(n, dealer_id=0):
    nodes = {}
    for i in range(n):
        nodes[i] = Node(i, behavior=None)
    for i in range(n-1):
        connect_nodes(nodes[i], nodes[i+1])

    return nodes

def build_complete_graph(n, dealer_id=0):
    G_complete = nx.complete_graph(n)
    nodes = nodes_from_networkx(G_complete)
    return nodes

def build_complete_multipartite_graph(n, dealer_id=0, subset_sizes=(4, 3, 3)):
    # subset_sizes can be like (4, 3, 3) or (range(0,4), range(4,7), range(7,10))
    G = nx.complete_multipartite_graph(*subset_sizes)
    return nodes_from_networkx(G)

def build_complete_bipartite_graph(n, dealer_id=0, subset_sizes=(3, 3)):
    # Expect subset_sizes to be a 2-tuple (a, b). If not provided, split n roughly in half.
    if subset_sizes is None or len(subset_sizes) != 2:
        a = n // 2
        b = n - a
    else:
        a, b = subset_sizes
    G = nx.complete_bipartite_graph(a, b)
    G = nx.convert_node_labels_to_integers(G)
    return nodes_from_networkx(G)

def build_star_graph(n, dealer_id=0):
    # networkx.star_graph(k) returns k+1 nodes. To get n nodes total, pass n-1.
    G = nx.star_graph(max(0, n - 1))
    G = nx.convert_node_labels_to_integers(G)
    return nodes_from_networkx(G)

def build_hypercube_graph(n, dealer_id=0):
    # n is the requested number of nodes. Hypercube requires n = 2^d.
    # Use the largest d such that 2^d <= n (round down). If n < 2, fall back to a single node.
    import math
    if n <= 1:
        G = nx.empty_graph(1)
    else:
        d = int(math.floor(math.log2(n)))
        d = max(1, d)
        G = nx.hypercube_graph(d)
    G = nx.convert_node_labels_to_integers(G)
    return nodes_from_networkx(G)
