from adversary import ByzantineEquivocator
from graphs import build_line_graph, build_complete_graph, build_complete_multipartite_graph
from network import Message, Network
from utils import sample_t_local_faulty_set
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import random
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
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

def _build_graph(graph: str, n: int, dealer_id: int, subset_sizes: Optional[tuple[int, ...]] = None):
    if graph == "line":
        return build_line_graph(n, dealer_id)
    if graph == "complete":
        return build_complete_graph(n, dealer_id)
    if graph in {"complete_multipartite", "multipartite", "cmp"}:
        sizes = subset_sizes if subset_sizes is not None else (3, 3, 3)
        return build_complete_multipartite_graph(n, dealer_id, sizes)
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


def predict_cpa_outcome(adj: dict[int, set[int]], dealer_id: int, t: int) -> tuple[int, str]:
    # Heuristic based on literature: CPA succeeds if t < K(G,D)
    K = compute_K(adj, dealer_id)
    verdict = "succeeds" if t < K else ("fails" if t >= K else "unknown")
    return K, verdict


def evaluate_execution(decided: dict[int, tuple[bool, int | None]], B: set[int], dealer_value: int, dealer_id: int) -> tuple[bool, list[int]]:
    """Return (success, bad_honest_nodes).

    success = True iff every honest node (not in B) decided on dealer_value.
    bad_honest_nodes lists honest node ids that either did not decide or decided on a different value.
    """
    bad: list[int] = []
    for nid, (did_decide, val) in decided.items():
        if nid in B:
            continue
        # Dealer is honest by construction and should have dealer_value
        expected = dealer_value
        if not did_decide or val != expected:
            bad.append(nid)
    return (len(bad) == 0, bad)


def run_cpa_with_adversary(
    n: int = 10,
    dealer_id: int = 0,
    dealer_value: int = 1,
    t: int = 0,
    seed: Optional[int] = None,
    graph: str = "complete_multipartite",
    subset_sizes: Optional[tuple[int, ...]] = (3, 3, 3),
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes)
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
        except Exception:
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
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes)
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
):
    nodes = _build_graph(graph, n, dealer_id, subset_sizes)
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    # build per-node thresholds and sample faulty set under t(u)
    t_map = build_tu_map(nodes, t_func_id, n, seed)
    B = sample_tu_local_faulty_set(adj, t_map, seed=seed)
    B.discard(dealer_id)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t(u)-local): {B}")
    print(f"t(u) map: {{nid: t_map[nid] for nid in sorted(t_map)}}")

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
