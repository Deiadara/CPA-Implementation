from node import Node, connect_nodes


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
