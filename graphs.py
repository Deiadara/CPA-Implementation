from node import Node, connect_nodes
import networkx as nx
import json

def nodes_from_networkx(G):
    nodes = {i: Node(i, behavior=None) for i in G.nodes()}
    for u, v in G.edges():
        connect_nodes(nodes[u], nodes[v])
    return nodes

def build_custom_graph_from_json(json_path: str, dealer_id: int = 0):
    """
    Load a custom graph from a JSON file.
    
    Supported JSON formats:
    
    1. Simple format (adjacency list):
       {
         "adjacency": {
           "0": [1, 2],
           "1": [0, 2],
           "2": [0, 1, 3],
           "3": [2]
         }
       }
    
    2. Node-link format (similar to NetworkX/D3.js):
       {
         "nodes": [0, 1, 2, 3],
         "edges": [[0, 1], [1, 2], [2, 3]]
       }
    
    3. Detailed node-link format:
       {
         "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
         "links": [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
       }
    
    Args:
        json_path: Path to the JSON file containing graph data
        dealer_id: ID of the dealer node (not used for graph construction but kept for API consistency)
    
    Returns:
        Dictionary of Node objects indexed by node ID
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    # Format 1: Adjacency list
    if "adjacency" in data:
        adj_dict = data["adjacency"]
        # Add all nodes first
        for node_id in adj_dict.keys():
            G.add_node(int(node_id))
        # Add edges
        for node_id, neighbors in adj_dict.items():
            node_id = int(node_id)
            for neighbor in neighbors:
                G.add_edge(node_id, int(neighbor))
    
    # Format 2 & 3: Node-link format
    elif "nodes" in data:
        # Parse nodes
        nodes_data = data["nodes"]
        if isinstance(nodes_data, list):
            if len(nodes_data) > 0 and isinstance(nodes_data[0], dict):
                # Format 3: nodes are objects with "id" field
                for node in nodes_data:
                    G.add_node(node["id"])
            else:
                # Format 2: nodes are just IDs
                for node_id in nodes_data:
                    G.add_node(node_id)
        
        # Parse edges/links
        edges_data = data.get("edges", data.get("links", []))
        for edge in edges_data:
            if isinstance(edge, dict):
                # Format 3: edges are objects with "source" and "target"
                source = edge.get("source", edge.get("from"))
                target = edge.get("target", edge.get("to"))
                G.add_edge(source, target)
            else:
                # Format 2: edges are tuples/lists
                G.add_edge(edge[0], edge[1])
    
    else:
        raise ValueError(
            "Invalid JSON format. Expected either 'adjacency' or 'nodes' key. "
            "See documentation for supported formats."
        )
    
    # Convert to integer node labels if not already
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    return nodes_from_networkx(G)

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
