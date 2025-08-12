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

    connect_nodes(nodes[0], nodes[2])
    connect_nodes(nodes[1], nodes[5])
    connect_nodes(nodes[3], nodes[6])
    connect_nodes(nodes[4], nodes[9])
    connect_nodes(nodes[6], nodes[8])
    connect_nodes(nodes[2], nodes[4])
    connect_nodes(nodes[3], nodes[7])
    connect_nodes(nodes[4], nodes[8])
    connect_nodes(nodes[6], nodes[9])

    return nodes

def build_complete_graph(n, dealer_id=0):
    G_complete = nx.complete_graph(n)
    nodes = nodes_from_networkx(G_complete)
    return nodes

def build_complete_multipartite_graph(n, dealer_id=0, subset_sizes=(4, 3, 3)):
    # subset_sizes can be like (4, 3, 3) or (range(0,4), range(4,7), range(7,10))
    G = nx.complete_multipartite_graph(*subset_sizes)
    print(sorted(G.edges()))  # optional: show edges
    return nodes_from_networkx(G)
