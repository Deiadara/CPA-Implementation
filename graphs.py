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
    print(sorted(G.edges()))  
    return nodes_from_networkx(G)
