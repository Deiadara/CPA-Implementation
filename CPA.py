from adversary import ByzantineEquivocator
from graphs import (
    build_line_graph,
    build_complete_graph,
    build_complete_multipartite_graph,
    build_complete_bipartite_graph,
    build_star_graph,
    build_hypercube_graph,
    build_custom_graph_from_json,
    build_cycle_graph,
    build_dense_random_graph,
    build_sparse_random_graph,
    build_random_regular_graph,
    build_grid_2d_graph,
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

def _build_graph(graph: str, n: int, dealer_id: int, subset_sizes: Optional[tuple[int, ...]] = None, custom_graph_path: Optional[str] = None, seed: Optional[int] = None):
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
    if graph == "cycle":
        return build_cycle_graph(n, dealer_id)
    if graph in {"dense_random", "dense"}:
        return build_dense_random_graph(n, dealer_id, seed=seed)
    if graph in {"sparse_random", "sparse"}:
        return build_sparse_random_graph(n, dealer_id, seed=seed)
    if graph in {"random_regular", "regular"}:
        return build_random_regular_graph(n, dealer_id, degree=3, seed=seed)
    if graph in {"grid", "grid_2d"}:
        return build_grid_2d_graph(n, dealer_id)
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
    sender: int  # Who sent this message (for CPA threshold logic)
    value: int
    ds_round: int  # DS-CPA round number (not message-passing round)
    signature_chain: list[tuple[int, bytes]]  # [(node_id, signature), ...]
    
    def get_signers(self) -> set[int]:
        """Return set of node IDs that signed this message."""
        return {node_id for node_id, _ in self.signature_chain}


class HonestDSCPA(Behavior):
    """
    DS-CPA (Dolev-Strong combined with CPA) behavior.
    
    DS-CPA runs for f̂ + 1 rounds where f̂ = n - 2. Each "broadcast" in the 
    Dolev-Strong protocol is replaced by a σ-CPA execution. A DS-CPA round is 
    defined as the time needed for a message to traverse through the diameter d.
    
    IMPORTANT: Each DS-CPA round lasts for d message-passing rounds, where d is
    the diameter of the graph. This allows σ-CPA to propagate messages throughout
    the entire network before moving to the next DS-CPA round.
    
    Protocol:
    - DS-Round 0 (msg rounds 0 to d-1): Sender invokes σ-CPA with < b, sig(b) >
    - DS-Round r (msg rounds r*d to (r+1)*d-1): For each message b̃ with r signatures:
        - If b̃ ∉ Vi: add to Vi, sign it, invoke σ-CPA with r+1 signatures
    - After f̂+1 DS-rounds: output single value in Vi, or 0 if |Vi| ≠ 1
    """
    def __init__(self, sender_id: int, all_public_keys: dict[int, Ed25519PublicKey], 
                 my_private_key: Optional[Ed25519PrivateKey], f_hat: int, t: int, diameter: int):
        self.sender_id = sender_id
        self.all_public_keys = all_public_keys
        self.my_private_key = my_private_key
        self.f_hat = f_hat  # f̂ = n - 2
        self.t = t  # t-local corruption bound
        self.diameter = diameter  # Graph diameter (each DS-CPA round lasts d msg rounds)
        self.extracted_set = set()  # Vi in the protocol
        
        # Track messages for CPA threshold logic - BY VALUE
        # CPA threshold is about getting t+1 confirmations for a VALUE,
        # regardless of signature chains (multiple nodes can have different chains)
        # Format: {value: {'senders': set(), 'best_chain': signature_chain}}
        self.message_tracker_by_value = collections.defaultdict(lambda: {'senders': set(), 'best_chain': None})
        
        # Messages accepted and ready to broadcast in next DS-CPA round
        self.accepted_messages = []
        
        # Track which messages we've already broadcast to avoid duplicates
        # Track by (value, our_signature) to avoid broadcasting same value multiple times
        self.broadcasted_messages = set()  # Set of (value, frozenset(signers))
        
        # Track when we've already broadcast our messages for current DS-CPA round
        self.current_ds_round_broadcasted = False
        
    def on_receive(self, node, msg):
        if not isinstance(msg, DSCPAMessage):
            return
            
        # Verify the signature chain
        if not self._verify_signature_chain(msg):
            return
            
        signers = msg.get_signers()
        
        # Check if sender (original dealer) is in the signature chain
        if self.sender_id not in signers:
            return
        
        value = msg.value
        
        # Track this message for CPA threshold logic - BY VALUE
        # Multiple signature chains for same value count toward threshold
        tracker = self.message_tracker_by_value[value]
        tracker['senders'].add(msg.sender)
        # Keep the longest/best chain we've seen
        if tracker['best_chain'] is None or len(msg.signature_chain) > len(tracker['best_chain']):
            tracker['best_chain'] = msg.signature_chain
        
        senders = tracker['senders']
        
        # CPA logic (σ-CPA): Accept message if received from t+1 distinct neighbors
        # OR if it's directly from the sender/dealer (first hop special case)
        if msg.sender == self.sender_id or len(senders) >= self.t + 1:
            # If value not in extracted set, add it and prepare to relay
            if value not in self.extracted_set:
                self.extracted_set.add(value)
                
                # Prepare to broadcast with our signature added
                # Use the best chain we've seen for this value
                best_chain = tracker['best_chain'] if tracker['best_chain'] else msg.signature_chain
                
                if self.my_private_key and node.id not in signers:
                    new_chain = list(best_chain)
                    my_sig = self.my_private_key.sign(_encode_value_bytes(value))
                    new_chain.append((node.id, my_sig))
                    
                    self.accepted_messages.append({
                        'value': value,
                        'chain': new_chain,
                        'ds_round': len(new_chain)  # DS-CPA round = number of signatures
                    })
    
    def on_round(self, node, rnd):
        """
        Called each message-passing round.
        
        Each DS-CPA round lasts for `diameter` message-passing rounds.
        DS-round r spans message rounds [r*diameter, (r+1)*diameter - 1].
        
        σ-CPA works like normal CPA: nodes broadcast immediately after accepting
        messages (when t+1 threshold is met), allowing continuous propagation.
        
        The DS-CPA round structure is about which signature counts are being processed,
        but within each DS-round, σ-CPA propagates messages continuously.
        """
        out = []
        
        # Calculate current DS-CPA round from message-passing round
        ds_round = rnd // self.diameter
        round_phase = rnd % self.diameter  # Position within current DS-CPA round
        
        # DS-Round 0, Phase 0: Only the sender initiates σ-CPA
        if rnd == 0 and node.id == self.sender_id:
            if self.my_private_key and node.value is not None:
                sig = self.my_private_key.sign(_encode_value_bytes(node.value))
                chain = [(node.id, sig)]
                self.extracted_set.add(node.value)
                
                msg_key = (node.value, frozenset([node.id]))
                self.broadcasted_messages.add(msg_key)
                
                # Broadcast to all neighbors (σ-CPA)
                for nid in node.neighbors:
                    out.append((nid, DSCPAMessage("DS-CPA", node.id, node.value, 1, chain)))
        
        # Every round: broadcast any accepted messages that haven't been sent yet
        # This implements σ-CPA: immediate relay after acceptance (CPA behavior)
        else:
            # Broadcast all accepted messages that haven't been broadcast yet
            messages_to_send = []
            for msg_data in self.accepted_messages[:]:  # Copy list to avoid modification during iteration
                msg_key = (msg_data['value'], frozenset(s for s, _ in msg_data['chain']))
                if msg_key not in self.broadcasted_messages:
                    messages_to_send.append(msg_data)
                    self.broadcasted_messages.add(msg_key)
            
            # Broadcast to all neighbors
            for msg_data in messages_to_send:
                for nid in node.neighbors:
                    out.append((nid, DSCPAMessage("DS-CPA", node.id, msg_data['value'], 
                                                 msg_data['ds_round'], msg_data['chain'])))
            
            # Remove broadcasted messages from accepted list
            self.accepted_messages = [m for m in self.accepted_messages 
                                     if (m['value'], frozenset(s for s, _ in m['chain'])) not in self.broadcasted_messages]
        
        # Decision at the end of f_hat DS-CPA rounds (last message-passing round)
        if ds_round == self.f_hat and round_phase == self.diameter - 1:
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
    
    DS-CPA replaces each broadcast in Dolev-Strong with a σ-CPA (CPA with signatures) 
    execution. It runs for f̂ + 1 DS-CPA rounds where f̂ = n - 2 (maximum possible 
    corrupted nodes under t-local corruption when exact count is unknown).
    
    Each DS-CPA round consists of:
    1. Nodes receive messages with r signatures
    2. Apply CPA threshold logic (t+1 distinct senders or from original sender)
    3. Accept value into extracted set Vi
    4. Sign and prepare to broadcast with r+1 signatures
    
    Each "broadcast" uses σ-CPA: messages propagate through neighbors with CPA's
    t+1 threshold, ensuring Byzantine resilience under t-local corruption.
    
    Args:
        n: Number of nodes
        sender_id: ID of the sender/dealer node
        sender_value: Value to broadcast
        t: Parameter for t-local corruption bound (used in CPA threshold)
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
    print(f"t-local corruption bound: {t}")
    
    # Calculate graph diameter (critical for DS-CPA round duration)
    diameter = _graph_diameter_from_dealer(adj, sender_id)
    print(f"Graph diameter from sender: {diameter}")
    
    # Generate keypairs for all nodes
    private_keys = {}
    public_keys = {}
    for i in nodes:
        priv_key = Ed25519PrivateKey.generate()
        private_keys[i] = priv_key
        public_keys[i] = priv_key.public_key()
    
    # f̂ = n - 2 (maximum corrupted nodes when exact count unknown)
    f_hat = n - 2
    
    # Calculate total message-passing rounds:
    # DS-CPA runs for f̂ + 1 DS-rounds, each lasting `diameter` message-passing rounds
    ds_cpa_rounds = f_hat + 1  # DS-CPA rounds (0 to f̂)
    total_msg_rounds = ds_cpa_rounds * diameter  # Total message-passing rounds
    
    print(f"Running DS-CPA:")
    print(f"  - DS-CPA rounds: {ds_cpa_rounds} (rounds 0 to {f_hat})")
    print(f"  - Each DS-round lasts: {diameter} message-passing rounds")
    print(f"  - Total message-passing rounds: {total_msg_rounds}")
    
    # Assign behaviors
    for i in nodes:
        if i == sender_id:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, t, diameter)
            nodes[i].value = sender_value
        elif i in B:
            # Byzantine nodes - use equivocator
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, t, diameter)
    
    net = Network(nodes)
    
    # Run for all message-passing rounds
    for r in range(total_msg_rounds):
        ds_round = r // diameter
        msg_phase = r % diameter
        print(f"\n--- Message Round {r} (DS-Round {ds_round}, phase {msg_phase}/{diameter-1}) ---")
        net.run_round(r)
        
        # Debug: Show extracted sets (only at DS-round boundaries to reduce output)
        if msg_phase == 0 or msg_phase == diameter - 1:
            for i in sorted(nodes.keys()):
                if i not in B:
                    behavior = nodes[i].behavior
                    if isinstance(behavior, HonestDSCPA):
                        if behavior.extracted_set:
                            print(f"Node {i} extracted set: {behavior.extracted_set}")
        
        # Show decisions (happens at end of last DS-CPA round)
        if ds_round == f_hat and msg_phase == diameter - 1:
            print(f"\n--- DECISION TIME (end of DS-Round {f_hat}) ---")
            for i in sorted(nodes.keys()):
                node = nodes[i]
                if node.decided:
                    print(f"Node {i} decided on: {node.value}")
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    return decided, B
