from adversary import ByzantineEquivocator
from graphs import (
    build_line_graph,
    build_complete_graph,
    build_complete_multipartite_graph,
    build_complete_bipartite_graph,
    build_star_graph,
    build_hypercube_graph,
    build_custom_graph_from_json,
)
from network import Message, Network
from utils import sample_t_local_faulty_set
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import random
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature
from node import Behavior
import collections

ROUNDS = 100  # legacy constant no longer used; kept for compatibility if referenced elsewhere

class HonestCPA(Behavior):
    def __init__(self, dealer_id, t):
        self.dealer_id = dealer_id
        self.t = t

    def on_receive(self, node, msg):
        if msg.mtype != "PROPOSE":
            return
        node.received_from[msg.value].add(msg.sender)

        if msg.sender == self.dealer_id:
            node.seen_from_dealer.add(msg.value)

    def on_round(self, node, rnd):
        out = []
        if not node.decided:
            for v in node.seen_from_dealer:
                node.decide(v)

            if not node.decided:
                for v, senders in node.received_from.items():
                    if len(senders) >= self.t + 1:
                        node.decide(v)
                        break

        if node.decided and not node.already_broadcast:
            for nid in node.neighbors:
                out.append((nid, Message("PROPOSE", node.id, node.value, rnd)))
            node.already_broadcast = True
        return out

def _build_graph(graph: str, n: int, dealer_id: int, subset_sizes: Optional[tuple[int, ...]] = None, custom_graph_path: Optional[str] = None):
    if graph == "custom":
        if custom_graph_path is None:
            raise ValueError("--custom-graph path must be provided when using --graph custom")
        return build_custom_graph_from_json(custom_graph_path, dealer_id)
    if graph == "line":
        return build_line_graph(n, dealer_id)
    if graph == "complete":
        return build_complete_graph(n, dealer_id)
    if graph in {"complete_multipartite", "multipartite", "cmp"}:
        sizes = subset_sizes if subset_sizes is not None else (3, 3, 3)
        return build_complete_multipartite_graph(n, dealer_id, sizes)
    if graph in {"complete_bipartite", "bipartite", "kbipartite"}:
        sizes2 = subset_sizes if subset_sizes is not None else (n // 2, n - (n // 2))
        # Ensure it is a 2-tuple
        if isinstance(sizes2, tuple) and len(sizes2) >= 2:
            sizes2 = (sizes2[0], sizes2[1])
        else:
            sizes2 = (n // 2, n - (n // 2))
        return build_complete_bipartite_graph(n, dealer_id, sizes2)
    if graph == "star":
        return build_star_graph(n, dealer_id)
    if graph == "hypercube":
        return build_hypercube_graph(n, dealer_id)
    # default fallback
    return build_complete_multipartite_graph(n, dealer_id, (3, 3, 3))


def _graph_diameter_from_dealer(adj: dict[int, set[int]], dealer_id: int) -> int:
    # BFS from dealer to compute eccentricity; diameter lower bound is max distance from dealer
    dist = {dealer_id: 0}
    q = collections.deque([dealer_id])
    while q:
        u = q.popleft()
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return max(dist.values()) if dist else 0


def minimum_m_level_ordering_exists(adj: dict[int, set[int]], dealer_id: int, m: int) -> bool:
    # Algorithm from CPAjournal.pdf: existence check of a minimum m-level ordering
    counters = {v: 0 for v in adj}
    q = collections.deque()
    enqueued = set()

    # Step 2: enqueue dealer and all its neighbors
    q.append(dealer_id)
    enqueued.add(dealer_id)
    for nb in adj.get(dealer_id, set()):
        if nb not in enqueued:
            q.append(nb)
            enqueued.add(nb)

    # Steps 3-4: process
    while q:
        u = q.popleft()
        for v in adj.get(u, set()):
            counters[v] += 1
            if v not in enqueued and counters[v] >= m:
                q.append(v)
                enqueued.add(v)

    # Step 5: all nodes must have been enqueued
    return len(enqueued) == len(adj)


def compute_K(adj: dict[int, set[int]], dealer_id: int) -> int:
    # K(G,D) = max m in N such that a minimum m-level ordering exists
    # Monotone in m, so increase until it fails
    n = len(adj)
    last_ok = 0
    for m in range(0, n + 1):
        if minimum_m_level_ordering_exists(adj, dealer_id, m):
            last_ok = m
        else:
            break
    return last_ok


def predict_cpa_outcome_for_constant_t(adj: dict[int, set[int]], dealer_id: int, t: int) -> tuple[int, str]:
    # Heuristic based on literature: CPA succeeds if t < K(G,D)
    K = compute_K(adj, dealer_id)
    verdict = "succeeds" if 2*t < K else ("fails" if t >= K else "unknown")
    return K, verdict


def check_t_local_cut_exists(adj: dict[int, set[int]], v: int, t: int) -> bool:
    """
    Check if there exists a t-local cut with respect to node v.
    
    A t-local cut with respect to v exists if:
    - We can remove at most t neighbors of v (forming set B with |B ∩ N(v)| ≤ t)
    - This separates v from some other nodes (creates a graph partition)
    - AND importantly: the remaining graph (excluding v and B) is NOT empty and NOT fully connected to v
    
    In other words: removing ≤ t of v's neighbors creates TWO+ non-trivial components
    where at least one component doesn't contain v and has ≥ 2 nodes.
    
    Returns True if a t-local cut exists, False otherwise.
    """
    n = len(adj)
    if n <= 2:
        # With ≤ 2 nodes, can't have a meaningful separation
        return False
    
    neighbors_v = adj.get(v, set())
    
    # Special case: if v has <= t neighbors, check if removing them creates a non-trivial separation
    if len(neighbors_v) <= t:
        # Remove v and all its neighbors, check if remaining graph is non-empty and connected
        removed = {v} | neighbors_v
        remaining = set(adj.keys()) - removed
        
        if len(remaining) <= 1:
            # No non-trivial separation (remaining part has ≤ 1 node)
            return False
        
        # Check if remaining nodes form a connected component
        # If they do, then removing v + its neighbors doesn't create a separation
        # (the rest of the graph stays connected)
        if _is_connected_subgraph(adj, remaining):
            # Remaining graph is connected, so this is NOT a t-local cut
            # (we just isolated v, but didn't separate the graph)
            return False
        else:
            # Remaining graph is disconnected, so we created a separation
            return True
    
    # General case: try removing each subset of size ≤ t from v's neighbors
    # Check if any such removal creates a separation
    from itertools import combinations
    
    for subset_size in range(1, min(t + 1, len(neighbors_v)) + 1):
        for removed_neighbors in combinations(neighbors_v, subset_size):
            removed_set = set(removed_neighbors)
            
            # Check if removing these neighbors separates the graph
            # The remaining graph should have v in one component and other nodes in another
            remaining_neighbors = neighbors_v - removed_set
            
            if not remaining_neighbors:
                # v becomes isolated, check if rest is non-trivial
                remaining_nodes = set(adj.keys()) - {v} - removed_set
                if len(remaining_nodes) >= 2 and not _is_connected_subgraph(adj, remaining_nodes):
                    return True
                continue
            
            # Check if v can reach all other nodes through remaining neighbors
            reachable_from_v = _get_reachable(adj, v, removed_set)
            unreachable = set(adj.keys()) - reachable_from_v - removed_set
            
            if len(unreachable) >= 2:
                # Found nodes unreachable from v after removing the subset
                # This is a t-local cut
                return True
    
    return False


def _can_reach_with_more_than_t_neighbors(adj: dict[int, set[int]], v: int, target: int, t: int) -> bool:
    """
    Check if we can reach target from v through more than t of v's neighbors.
    Returns True if target is reachable through > t different neighbors of v.
    """
    neighbors_v = adj.get(v, set())
    
    # If target is a direct neighbor, it's reachable through itself
    if target in neighbors_v:
        return True
    
    # Count how many of v's neighbors can reach target
    reachable_count = 0
    for neighbor in neighbors_v:
        # BFS from this neighbor to target (without going back through v)
        if _bfs_without_node(adj, neighbor, target, v):
            reachable_count += 1
            if reachable_count > t:
                return True
    
    return False


def _bfs_without_node(adj: dict[int, set[int]], start: int, target: int, blocked: int) -> bool:
    """BFS from start to target, without passing through blocked node."""
    if start == target:
        return True
    
    visited = {start, blocked}
    queue = collections.deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, set()):
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False


def _is_connected_subgraph(adj: dict[int, set[int]], nodes: set[int]) -> bool:
    """Check if the given set of nodes forms a connected subgraph."""
    if not nodes:
        return True
    if len(nodes) == 1:
        return True
    
    # BFS from arbitrary node in the set
    start = next(iter(nodes))
    visited = {start}
    queue = collections.deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, set()):
            if neighbor in nodes and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(nodes)


def _get_reachable(adj: dict[int, set[int]], start: int, blocked: set[int]) -> set[int]:
    """Get all nodes reachable from start without passing through blocked nodes."""
    reachable = {start}
    queue = collections.deque([start])
    
    while queue:
        node = queue.popleft()
        for neighbor in adj.get(node, set()):
            if neighbor not in blocked and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    
    return reachable


def _compute_vertex_connectivity(adj: dict[int, set[int]]) -> int:
    """
    Compute the vertex connectivity of the graph.
    
    Vertex connectivity = minimum number of vertices whose removal disconnects the graph.
    
    Uses NetworkX for efficient computation.
    """
    import networkx as nx
    
    n = len(adj)
    if n <= 1:
        return n
    
    # Build NetworkX graph from adjacency dict
    G = nx.Graph()
    G.add_nodes_from(adj.keys())
    for node, neighbors in adj.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Compute vertex connectivity
    # This is the size of the minimum vertex cut
    return nx.node_connectivity(G)


def predict_ds_cpa_outcome_for_any_dealer(adj: dict[int, set[int]], t: int) -> tuple[bool, str]:
    """
    Predict if DS-CPA will succeed for ANY dealer choice.
    
    According to Theorem 4.29: DS-CPA achieves Byzantine Broadcast if for every v ∈ V
    there does not exist a t-local cut.
    
    A t-local node cut (separator) S exists if:
    1. Removing S disconnects the graph
    2. |S ∩ N(v)| ≤ t for every node v (t-locality constraint)
    3. S does not contain the sender/dealer (dealer is always honest)
    
    This checks if for ANY possible dealer, the graph remains connected after removing
    any valid t-local Byzantine set.
    
    Returns:
        (has_cut, verdict) where:
        - has_cut: True if a t-local separator exists for some dealer
        - verdict: "succeeds" or "fails"
    """
    n = len(adj)
    
    if n <= 2:
        return False, "succeeds"
    
    # For each possible dealer, check if there exists a t-local cut
    # (excluding the dealer from Byzantine set)
    for dealer in adj:
        if _has_t_local_cut_excluding_dealer(adj, dealer, t):
            # Found a dealer for which a t-local cut exists
            return True, "fails"
    
    # No t-local cut exists for any dealer choice
    return False, "succeeds"


def predict_ds_cpa_outcome(adj: dict[int, set[int]], sender_id: int, t: int) -> tuple[bool, str]:
    """
    Predict if DS-CPA will succeed for a SPECIFIC sender/dealer.
    
    According to Theorem 4.29: DS-CPA achieves Byzantine Broadcast if there does not
    exist a t-local cut (where dealer is always honest).
    
    A t-local node cut (separator) S exists if:
    1. Removing S disconnects the graph
    2. |S ∩ N(v)| ≤ t for every node v (t-locality constraint)
    3. S does not contain the sender/dealer (dealer is always honest)
    
    Returns:
        (has_cut, verdict) where:
        - has_cut: True if a t-local separator exists (excluding dealer)
        - verdict: "succeeds" or "fails"
    """
    n = len(adj)
    
    if n <= 2:
        return False, "succeeds"
    
    # Check if there exists a t-local cut that excludes the specific dealer
    has_cut = _has_t_local_cut_excluding_dealer(adj, sender_id, t)
    
    if has_cut:
        return True, "fails"
    else:
        return False, "succeeds"


def _has_t_local_cut_excluding_dealer(adj: dict[int, set[int]], dealer: int, t: int) -> bool:
    """
    Check if there exists a t-local separator S where dealer ∉ S.
    
    Algorithm:
    1. Find all minimal vertex cuts of size ≤ t
    2. Check if any such cut is t-local AND doesn't contain dealer
    """
    import networkx as nx
    from itertools import combinations
    
    n = len(adj)
    all_nodes = set(adj.keys())
    non_dealer_nodes = all_nodes - {dealer}
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(adj.keys())
    for node, neighbors in adj.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Try all subsets of non-dealer nodes of size 1 to t
    for size in range(1, min(t + 1, len(non_dealer_nodes)) + 1):
        for separator_tuple in combinations(non_dealer_nodes, size):
            separator = set(separator_tuple)
            
            # Check if this separator disconnects the graph
            remaining = all_nodes - separator
            if len(remaining) <= 1:
                continue
            
            # Check if graph is disconnected after removing separator
            G_remaining = G.subgraph(remaining)
            if not nx.is_connected(G_remaining):
                # This is a separator! Now check if it's t-local
                if _is_t_local_set(adj, separator, t):
                    # Found a t-local separator that doesn't include dealer
                    return True
    
    return False


def _is_t_local_set(adj: dict[int, set[int]], node_set: set[int], t: int) -> bool:
    """
    Check if a set of nodes satisfies the t-locality constraint.
    
    A set S is t-local if |S ∩ N(v)| ≤ t for every node v.
    """
    for v in adj:
        neighbors_v = adj[v]
        intersection = node_set & neighbors_v
        if len(intersection) > t:
            return False
    return True

def predict_cpa_outcome_for_constant_t_and_signatures(adj: dict[int, set[int]], dealer_id: int, t: int) -> tuple[int, str]:
    # Heuristic based on literature: CPA succeeds if t < K(G,D)
    K = compute_K(adj, dealer_id)
    verdict = "succeeds" if t < K else "fails"
    return K, verdict

def evaluate_execution(decided: dict[int, tuple[bool, int | None]], B: set[int], dealer_value: int) -> tuple[bool, list[int]]:
    """Return (success, corrupted).

    success = True iff every honest node (not in B) decided on dealer_value.
    undesired lists honest node ids that either did not decide or decided on a different value.
    """
    undesired: list[int] = []
    for nid, (decided, val) in decided.items():
        if nid in B:
            continue
        # Dealer is honest by construction and should have dealer_value
        expected = dealer_value
        if not decided or val != expected:
            undesired.append(nid)
    return (len(undesired) == 0, undesired)


def run_cpa_with_adversary(
    n: int = 10,
    dealer_id: int = 0,
    dealer_value: int = 1,
    t: int = 0,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
    custom_graph_path: Optional[str] = None,
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, custom_graph_path)
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t-local): {B}")

    # assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPA(dealer_id, t)
            nodes[i].decide(dealer_value)
        elif i in B:
            # choose a Byzantine strategy
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),  # equivocate 0/1 forever
                withhold_prob=0.2,                # sometimes stay silent
                spam=True
            )
        else:
            nodes[i].behavior = HonestCPA(dealer_id, t)

    net = Network(nodes)

    # dealer broadcasts in round 0
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, Message("PROPOSE", dealer_id, dealer_value, 0)))
        print(f"Dealer {dealer_id} sending to neighbor {nid}: value={dealer_value}")
    net.deliver(initial_out)

    max_dist = _graph_diameter_from_dealer(adj, dealer_id)
    rounds = max_dist + 1
    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        net.run_round(r)

        # Debug: Show what each node received
        for i in sorted(nodes.keys()):
            node = nodes[i]
            if node.received_from:
                print(f"Node {i} received: {dict(node.received_from)}")
            if node.decided:
                print(f"Node {i} decided on: {node.value}")

    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}


    #nodes = build_complete_graph(n, dealer_id)
    #print(nodes)

    return decided, B


# ---------------- Signed CPA variant ----------------

def _encode_value_bytes(value: int) -> bytes:
    # Deterministic, fixed-length encoding of the integer value
    return int(value).to_bytes(8, byteorder="big", signed=True)


@dataclass
class SignedMessage:
    mtype: str
    sender: int
    value: int
    rnd: int
    signature: bytes


class HonestCPAWithDealerSignature(Behavior):
    def __init__(self, dealer_public_key: Ed25519PublicKey):
        self.dealer_public_key = dealer_public_key

    def on_receive(self, node, msg):
        if getattr(msg, "mtype", None) != "PROPOSE":
            return
        # Only accept properly signed dealer value
        signature = getattr(msg, "signature", None)
        if signature is None:
            return
        try:
            self.dealer_public_key.verify(signature, _encode_value_bytes(msg.value))
        except InvalidSignature:
            return
        if not node.decided:
            node.decide(msg.value)
            setattr(node, "dealer_signature", signature)

    def on_round(self, node, rnd):
        out = []
        if node.decided and not node.already_broadcast:
            sig: Optional[bytes] = getattr(node, "dealer_signature", None)
            for nid in node.neighbors:
                out.append((nid, SignedMessage("PROPOSE", node.id, node.value, rnd, sig)))
            node.already_broadcast = True
        return out


def run_cpa_with_dealer_signature(
    n: int = 10,
    dealer_id: int = 0,
    dealer_value: int = 1,
    t: int = 0,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
    custom_graph_path: Optional[str] = None,
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, custom_graph_path)
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    # sample a t-local faulty set (t only for sampling here) and ensure dealer is never faulty
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t-local): {B}")

    # Dealer keypair generation and public key distribution
    dealer_private_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()
    dealer_public_key: Ed25519PublicKey = dealer_private_key.public_key()

    # assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
            nodes[i].decide(dealer_value)
            setattr(nodes[i], "dealer_signature", dealer_private_key.sign(_encode_value_bytes(dealer_value)))
        elif i in B:
            # Byzantine nodes reuse existing strategies; their messages won't verify
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True,
            )
        else:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)

    net = Network(nodes)

    initial_out = []
    dealer_sig = getattr(nodes[dealer_id], "dealer_signature")
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, SignedMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_sig)))
        print(f"Dealer {dealer_id} sending to neighbor {nid}: value={dealer_value}")
    net.deliver(initial_out)

    max_dist = _graph_diameter_from_dealer(adj, dealer_id)
    rounds = max_dist + 1
    for r in range(1, rounds + 1):  
        print(f"\n--- Round {r} ---")
        net.run_round(r)

        for i in sorted(nodes.keys()):
            node = nodes[i]
            if node.decided:
                print(f"Node {i} decided on: {node.value}")

    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    return decided, B


# ---------------- Per-node threshold CPA variant ----------------

def _u_index(node_id: int) -> int:
    # Node ids are assumed 0..n-1; u is 1-based index
    return node_id + 1


if TYPE_CHECKING:
    from node import Node


def build_tu_map(nodes: dict[int, "Node"], func_id: int, n: int, seed: Optional[int]) -> dict[int, int]:
    rng = random.Random(seed)
    t_map: dict[int, int] = {}
    for nid in nodes:
        u = _u_index(nid)
        if func_id == 1:
            t_val = 1
        elif func_id == 2:
            t_val = u
        elif func_id == 3:
            t_val = u * u
        elif func_id == 4:
            t_val = u % 2
        elif func_id == 5:
            t_val = u % 5
        elif func_id == 6:
            t_val = rng.randint(0, n)
        else:
            t_val = 1
        t_map[nid] = t_val
    return t_map


def sample_tu_local_faulty_set(adj: dict[int, set[int]], t_map: dict[int, int], p_try=0.3, seed: Optional[int] = None) -> set[int]:
    rng = random.Random(seed)
    nodes = list(adj.keys())
    B: set[int] = set()

    def violates(candidate: int, Bset: set[int]) -> bool:
        Bp = Bset | {candidate}
        for u, Nu in adj.items():
            if len(Bp & Nu) > t_map[u]:
                return True
        return False

    for v in rng.sample(nodes, k=len(nodes)):
        if rng.random() < p_try and not violates(v, B):
            B.add(v)
    return B


class HonestCPAWithPerNodeT(Behavior):
    def __init__(self, dealer_id: int, t_map: dict[int, int]):
        self.dealer_id = dealer_id
        self.t_map = t_map

    def on_receive(self, node, msg):
        if msg.mtype != "PROPOSE":
            return
        node.received_from[msg.value].add(msg.sender)
        if msg.sender == self.dealer_id:
            node.seen_from_dealer.add(msg.value)

    def on_round(self, node, rnd):
        out = []
        if not node.decided:
            for v in node.seen_from_dealer:
                node.decide(v)

            if not node.decided:
                threshold = self.t_map.get(node.id, 1) + 1
                for v, senders in node.received_from.items():
                    if len(senders) >= threshold:
                        node.decide(v)
                        break

        if node.decided and not node.already_broadcast:
            for nid in node.neighbors:
                out.append((nid, Message("PROPOSE", node.id, node.value, rnd)))
            node.already_broadcast = True
        return out


def run_cpa_with_per_node_threshold(
    n: int = 10,
    dealer_id: int = 0,
    dealer_value: int = 1,
    t_func_id: int = 1,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
    custom_graph_path: Optional[str] = None,
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, custom_graph_path)
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    # build per-node thresholds and sample faulty set under t(u)
    t_map = build_tu_map(nodes, t_func_id, n, seed)
    B = sample_tu_local_faulty_set(adj, t_map, seed=seed)
    B.discard(dealer_id)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t(u)-local): {B}")
    print(f"t(u) map: { {nid: t_map[nid] for nid in sorted(t_map)} }")

    # assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithPerNodeT(dealer_id, t_map)
            nodes[i].decide(dealer_value)
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True,
            )
        else:
            nodes[i].behavior = HonestCPAWithPerNodeT(dealer_id, t_map)

    net = Network(nodes)

    # dealer broadcasts in round 0
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, Message("PROPOSE", dealer_id, dealer_value, 0)))
        print(f"Dealer {dealer_id} sending to neighbor {nid}: value={dealer_value}")
    net.deliver(initial_out)

    max_dist = _graph_diameter_from_dealer(adj, dealer_id)
    rounds = max_dist + 1
    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        net.run_round(r)
        for i in sorted(nodes.keys()):
            node = nodes[i]
            if node.decided:
                print(f"Node {i} decided on: {node.value}")

    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    return decided, B


# ---------------- Combined: dealer signature + per-node threshold t(u) ---------------

class HonestCPAWithDealerSignatureAndPerNodeT(Behavior):
    def __init__(self, dealer_public_key: Ed25519PublicKey, t_map: dict[int, int], dealer_id: int):
        self.dealer_public_key = dealer_public_key
        self.t_map = t_map
        self.dealer_id = dealer_id

    def on_receive(self, node, msg):
        # Accept only messages that carry a valid dealer signature for the value
        if getattr(msg, "mtype", None) != "PROPOSE":
            return
        signature = getattr(msg, "signature", None)
        if signature is None:
            return
        try:
            self.dealer_public_key.verify(signature, _encode_value_bytes(msg.value))
        except InvalidSignature:
            return
        # Track receipt for thresholding and note dealer-origin value visibility
        node.received_from[msg.value].add(msg.sender)
        if msg.sender == self.dealer_id:
            node.seen_from_dealer.add(msg.value)
        # Store signature so we can forward it if/when we decide
        if not getattr(node, "dealer_signature", None):
            setattr(node, "dealer_signature", signature)

    def on_round(self, node, rnd):
        out = []
        if not node.decided:
            # Immediate decide if saw dealer value
            for v in node.seen_from_dealer:
                node.decide(v)

            # Otherwise threshold-based decide using t(u)
            if not node.decided:
                threshold = self.t_map.get(node.id, 1) + 1
                for v, senders in node.received_from.items():
                    if len(senders) >= threshold:
                        node.decide(v)
                        break

        if node.decided and not node.already_broadcast:
            sig: Optional[bytes] = getattr(node, "dealer_signature", None)
            for nid in node.neighbors:
                out.append((nid, SignedMessage("PROPOSE", node.id, node.value, rnd, sig)))
            node.already_broadcast = True
        return out


def run_cpa_with_dealer_signature_and_per_node_threshold(
    n: int = 10,
    dealer_id: int = 0,
    dealer_value: int = 1,
    t_func_id: int = 1,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
    custom_graph_path: Optional[str] = None,
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, custom_graph_path)
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    # Build per-node thresholds t(u) and sample faulty set accordingly
    t_map = build_tu_map(nodes, t_func_id, n, seed)
    B = sample_tu_local_faulty_set(adj, t_map, seed=seed)
    B.discard(dealer_id)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t(u)-local): {B}")
    print(f"t(u) map: { {nid: t_map[nid] for nid in sorted(t_map)} }")

    # Dealer keypair
    dealer_private_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()
    dealer_public_key: Ed25519PublicKey = dealer_private_key.public_key()

    # Assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithDealerSignatureAndPerNodeT(dealer_public_key, t_map, dealer_id)
            nodes[i].decide(dealer_value)
            setattr(nodes[i], "dealer_signature", dealer_private_key.sign(_encode_value_bytes(dealer_value)))
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True,
            )
        else:
            nodes[i].behavior = HonestCPAWithDealerSignatureAndPerNodeT(dealer_public_key, t_map, dealer_id)

    net = Network(nodes)

    # Dealer broadcasts in round 0 with signature
    initial_out = []
    dealer_sig = getattr(nodes[dealer_id], "dealer_signature")
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, SignedMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_sig)))
        print(f"Dealer {dealer_id} sending to neighbor {nid}: value={dealer_value}")
    net.deliver(initial_out)

    max_dist = _graph_diameter_from_dealer(adj, dealer_id)
    rounds = max_dist + 1
    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        net.run_round(r)
        for i in sorted(nodes.keys()):
            node = nodes[i]
            if node.decided:
                print(f"Node {i} decided on: {node.value}")

    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    return decided, B


# ---------------- DS-CPA (Dolev-Strong with CPA) variant ----------------

@dataclass
class DSCPAMessage:
    """Message for DS-CPA with signature chain."""
    mtype: str
    value: int
    rnd: int
    signature_chain: list[tuple[int, bytes]]  # [(node_id, signature), ...]
    
    def get_signers(self) -> set[int]:
        """Return set of node IDs that signed this message."""
        return {node_id for node_id, _ in self.signature_chain}


class HonestDSCPA(Behavior):
    """
    DS-CPA (Dolev-Strong combined with CPA) behavior.
    
    In DS-CPA, instead of direct broadcast, each node uses CPA with signatures (σ-CPA)
    to propagate values. This combines the robustness of Dolev-Strong with the 
    efficiency of CPA under t-local corruption.
    """
    def __init__(self, sender_id: int, all_public_keys: dict[int, Ed25519PublicKey], 
                 my_private_key: Optional[Ed25519PrivateKey], f_hat: int):
        self.sender_id = sender_id
        self.all_public_keys = all_public_keys  # Maps node_id -> public_key
        self.my_private_key = my_private_key
        self.f_hat = f_hat  # f_hat = n - 2
        self.extracted_set = set()  # V_i in the protocol
        self.messages_to_relay = []  # Messages to send in next round
        self.sent_count = 0  # Track how many messages we've sent
        
    def on_receive(self, node, msg):
        if not isinstance(msg, DSCPAMessage):
            return
            
        # Verify the signature chain
        if not self._verify_signature_chain(msg):
            return
            
        signers = msg.get_signers()
        
        # Check if sender is in the signature chain
        if self.sender_id not in signers:
            return
            
        value = msg.value
        num_signatures = len(signers)
        
        # Check if we should accept this message based on round
        # In round r, we expect r signatures (including sender)
        if num_signatures != msg.rnd:
            return
            
        # If value not in extracted set, add it and prepare to relay
        if value not in self.extracted_set:
            self.extracted_set.add(value)
            
            # Prepare to relay this message with our signature added
            if self.my_private_key and node.id not in signers:
                new_chain = list(msg.signature_chain)
                my_sig = self.my_private_key.sign(_encode_value_bytes(value))
                new_chain.append((node.id, my_sig))
                
                self.messages_to_relay.append({
                    'value': value,
                    'chain': new_chain,
                    'round': msg.rnd + 1
                })
    
    def on_round(self, node, rnd):
        out = []
        
        # Round 0: Only the sender broadcasts initial value
        if rnd == 0 and node.id == self.sender_id:
            if self.my_private_key and node.value is not None:
                sig = self.my_private_key.sign(_encode_value_bytes(node.value))
                chain = [(node.id, sig)]
                self.extracted_set.add(node.value)
                
                # Send to all neighbors using σ-CPA
                for nid in node.neighbors:
                    out.append((nid, DSCPAMessage("DS-CPA", node.value, 1, chain)))
                self.sent_count += 1
        
        # Rounds 1 to f_hat + 1: Relay messages
        elif rnd > 0 and rnd <= self.f_hat + 1:
            # Send prepared messages from previous round
            # Limit to avoid spamming (send at most 2 messages per round as per protocol)
            messages_to_send = [m for m in self.messages_to_relay 
                               if m['round'] == rnd and self.sent_count < 2]
            
            for msg_data in messages_to_send:
                for nid in node.neighbors:
                    out.append((nid, DSCPAMessage("DS-CPA", msg_data['value'], 
                                                 msg_data['round'], msg_data['chain'])))
                self.sent_count += 1
            
            # Clear sent messages
            self.messages_to_relay = [m for m in self.messages_to_relay 
                                     if m['round'] != rnd]
        
        # At the end of f_hat + 1 rounds, decide
        if rnd == self.f_hat + 1:
            if len(self.extracted_set) == 1:
                # Output the single value
                node.decide(list(self.extracted_set)[0])
            else:
                # Output 0 (default/bottom value)
                node.decide(0)
        
        return out
    
    def _verify_signature_chain(self, msg: DSCPAMessage) -> bool:
        """Verify that all signatures in the chain are valid."""
        if not msg.signature_chain:
            return False
            
        for node_id, signature in msg.signature_chain:
            if node_id not in self.all_public_keys:
                return False
            
            pub_key = self.all_public_keys[node_id]
            try:
                pub_key.verify(signature, _encode_value_bytes(msg.value))
            except InvalidSignature:
                return False
        
        return True


def run_ds_cpa(
    n: int = 10,
    sender_id: int = 0,
    sender_value: int = 1,
    t: int = 0,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
    custom_graph_path: Optional[str] = None,
):
    """
    Run DS-CPA (Dolev-Strong combined with CPA) protocol.
    
    DS-CPA replaces each broadcast in Dolev-Strong with a CPA execution with signatures.
    It runs for f_hat + 1 rounds where f_hat = n - 2 (since we only know t-local corruption).
    
    Args:
        n: Number of nodes
        sender_id: ID of the sender/dealer node
        sender_value: Value to broadcast
        t: Parameter for t-local fault sampling
        seed: Random seed for fault sampling
        graph: Graph type
        subset_sizes: Subset sizes for multipartite graphs
        custom_graph_path: Path to custom graph JSON
    
    Returns:
        (decided, B) where decided maps node_id -> (decided, value) and B is Byzantine set
    """
    nodes = _build_graph(graph, n, sender_id, subset_sizes, custom_graph_path)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    # Sample Byzantine nodes
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(sender_id)  # Sender is always honest
    
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t-local): {B}")
    print(f"Running DS-CPA with f_hat = {n - 2} rounds")
    
    # Generate keypairs for all nodes
    private_keys = {}
    public_keys = {}
    for i in nodes:
        priv_key = Ed25519PrivateKey.generate()
        private_keys[i] = priv_key
        public_keys[i] = priv_key.public_key()
    
    f_hat = n - 2
    
    # Assign behaviors
    for i in nodes:
        if i == sender_id:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat)
            nodes[i].value = sender_value
        elif i in B:
            # Byzantine nodes - use equivocator
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat)
    
    net = Network(nodes)
    
    # Run for f_hat + 1 rounds
    rounds = f_hat + 2  # +2 to include round 0 and final decision round
    for r in range(rounds):
        print(f"\n--- DS-CPA Round {r} ---")
        net.run_round(r)
        
        # Debug: Show extracted sets
        for i in sorted(nodes.keys()):
            if i not in B:
                behavior = nodes[i].behavior
                if isinstance(behavior, HonestDSCPA):
                    if behavior.extracted_set:
                        print(f"Node {i} extracted set: {behavior.extracted_set}")
        
        # Show decisions at the end
        if r >= f_hat + 1:
            for i in sorted(nodes.keys()):
                node = nodes[i]
                if node.decided:
                    print(f"Node {i} decided on: {node.value}")
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    return decided, B
