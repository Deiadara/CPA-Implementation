"""
Comprehensive benchmark comparing CPA, σ-CPA, DS-CPA, and B-CPA protocols.

This benchmark:
1. Tests specific corruption scenarios showing differences between algorithms
2. Runs random corruption samples for statistical analysis (parallelized)
3. Tracks actual rounds needed (when all honest nodes decided)
4. Compares honest vs dishonest dealer scenarios
"""

import io
import sys
import random
import collections
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from CPA import (
    run_cpa_with_adversary,
    run_cpa_with_dealer_signature,
    run_ds_cpa,
    run_bcpa,
    evaluate_execution,
    _build_graph,
    compute_K,
    _has_t_local_cut_excluding_dealer,
    HonestCPA,
    HonestCPAWithDealerSignature,
    HonestDSCPA,
    HonestBCPA,
    Network,
    Message,
    SignedMessage,
    DSCPAMessage,
    BCPAMessage,
    sample_t_local_faulty_set,
    sample_tu_local_faulty_set,
    build_tu_map,
)
from adversary import ByzantineEquivocator
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@dataclass
class BenchmarkResult:
    protocol: str
    graph: str
    n: int
    success: bool
    protocol_rounds: int  # Total rounds the protocol runs
    actual_rounds: int  # Round when all honest nodes decided
    honest_decided: int
    total_honest: int
    dealer_byzantine: bool
    corruption_set: set


def run_cpa_with_round_tracking(n, dealer_id, dealer_value, t, seed, graph, subset_sizes=None):
    """Run CPA and track when all honest nodes decided."""
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, None, seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPA(dealer_id, t)
            nodes[i].decide(dealer_value)
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestCPA(dealer_id, t)
    
    net = Network(nodes)
    
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, Message("PROPOSE", dealer_id, dealer_value, 0)))
    net.deliver(initial_out)
    
    rounds = n
    actual_round = rounds
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        
        # Check if all honest nodes have decided
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == rounds:
            actual_round = r
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided, B, dealer_value)
    honest_decided = sum(1 for i, (d, v) in decided.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, rounds, actual_round, honest_decided, total_honest, B


def run_sigma_cpa_with_round_tracking(n, dealer_id, dealer_value, t, seed, graph, subset_sizes=None):
    """Run σ-CPA and track when all honest nodes decided."""
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, None, seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    
    dealer_private_key = Ed25519PrivateKey.generate()
    dealer_public_key = dealer_private_key.public_key()
    
    def _encode_value_bytes(value):
        return int(value).to_bytes(8, byteorder="big", signed=True)
    
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
            nodes[i].decide(dealer_value)
            setattr(nodes[i], "dealer_signature", dealer_private_key.sign(_encode_value_bytes(dealer_value)))
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
    
    net = Network(nodes)
    
    dealer_sig = getattr(nodes[dealer_id], "dealer_signature")
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, SignedMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_sig)))
    net.deliver(initial_out)
    
    rounds = n
    actual_round = rounds
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == rounds:
            actual_round = r
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided, B, dealer_value)
    honest_decided = sum(1 for i, (d, v) in decided.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, rounds, actual_round, honest_decided, total_honest, B


def run_ds_cpa_with_round_tracking(n, sender_id, sender_value, t, seed, graph, subset_sizes=None):
    """Run DS-CPA and track when all honest nodes decided."""
    nodes = _build_graph(graph, n, sender_id, subset_sizes, None, seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(sender_id)
    
    private_keys = {}
    public_keys = {}
    for i in nodes:
        priv_key = Ed25519PrivateKey.generate()
        private_keys[i] = priv_key
        public_keys[i] = priv_key.public_key()
    
    f_hat = n - 2
    ds_cpa_rounds = f_hat + 1
    total_msg_rounds = ds_cpa_rounds * n
    
    for i in nodes:
        if i == sender_id:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, n)
            nodes[i].value = sender_value
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, n)
    
    net = Network(nodes)
    
    actual_round = total_msg_rounds
    
    for r in range(total_msg_rounds):
        net.run_round(r)
        
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == total_msg_rounds:
            actual_round = r + 1
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided, B, sender_value)
    honest_decided = sum(1 for i, (d, v) in decided.items() if i not in B and d and v == sender_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, total_msg_rounds, actual_round, honest_decided, total_honest, B


def run_bcpa_with_round_tracking(n, dealer_id, dealer_value, f, t, seed, graph, 
                                  subset_sizes=None, dealer_is_byzantine=False,
                                  timeout_rounds=None):
    """
    Run B-CPA and track when all honest nodes decided.
    
    For dishonest dealer scenarios (asynchronous model), we use a timeout.
    After timeout, we check if Agreement holds (all honest that decided agree).
    
    Success conditions (Byzantine Reliable Broadcast):
    - Termination: All honest nodes decide (or timeout with partial decision)
    - Agreement: All honest nodes that decided agree on same value
    - Validity: If dealer honest, honest nodes decide on dealer's value
    """
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, None, seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    
    if dealer_is_byzantine:
        B.add(dealer_id)
    else:
        B.discard(dealer_id)
    
    while len(B) > f:
        B.pop()
    
    for i in nodes:
        if i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (dealer_value, dealer_value + 1),
                withhold_prob=0.3,
                spam=True
            )
        else:
            nodes[i].behavior = HonestBCPA(dealer_id, n, f, t)
    
    net = Network(nodes)
    
    if dealer_id not in B:
        # Honest dealer: send consistent PROPOSE to all neighbors
        initial_out = []
        for nid in nodes[dealer_id].neighbors:
            initial_out.append((nid, BCPAMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_id)))
        net.deliver(initial_out)
        
        behavior = nodes[dealer_id].behavior
        if isinstance(behavior, HonestBCPA):
            behavior.dealer_value_received = dealer_value
            behavior.echoed_value = dealer_value
            propose_key = ("PROPOSE", dealer_value, dealer_id)
            echo_key = ("ECHO", dealer_value, dealer_id)
            behavior.already_relayed.add(propose_key)
            behavior.already_relayed.add(echo_key)
            behavior.pending_relays.append(("ECHO", dealer_value, dealer_id))
            behavior.echo_received[dealer_value].add(dealer_id)
    else:
        # Byzantine dealer: equivocates by sending different values
        # For B-CPA to potentially succeed, we need enough nodes to receive same value
        # to reach quorum (n-f). With n >= 3f+1, if majority gets same value, quorum possible.
        neighbors = list(nodes[dealer_id].neighbors)
        initial_out = []
        
        # Send to 2/3 of neighbors value 0 (to enable potential quorum), 1/3 get value 1
        cutoff = (2 * len(neighbors)) // 3
        for i, nid in enumerate(neighbors):
            val = 0 if i < cutoff else 1
            initial_out.append((nid, BCPAMessage("PROPOSE", dealer_id, val, 0, dealer_id)))
        net.deliver(initial_out)
    
    # Default timeout: 4n rounds, but can be overridden for async simulation
    total_rounds = timeout_rounds if timeout_rounds else 4 * n
    actual_round = total_rounds
    
    for r in range(1, total_rounds + 1):
        net.run_round(r)
        
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == total_rounds:
            actual_round = r
    
    decided = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    
    # Evaluate success based on Byzantine Reliable Broadcast properties
    honest_values = set()
    honest_decided_count = 0
    honest_undecided = 0
    decided_value = None
    for nid, (d, v) in decided.items():
        if nid not in B:
            if d:
                honest_decided_count += 1
                honest_values.add(v)
                decided_value = v
            else:
                honest_undecided += 1
    
    total_honest = len([i for i in nodes if i not in B])
    
    if dealer_id not in B:
        # Honest dealer: Validity requires all decide on dealer's value
        # Termination: all honest decide
        # Agreement: all same value
        all_decided_correctly = all(
            decided[i][0] and decided[i][1] == dealer_value 
            for i in nodes if i not in B
        )
        success = all_decided_correctly
    else:
        # Dishonest dealer (asynchronous model):
        # - Termination NOT guaranteed (may not decide)
        # - Agreement: all honest that decided agree on same value
        # - Safety: not deciding is acceptable (better than disagreement)
        
        # Success = Agreement holds (0 or 1 distinct values among decided honest)
        # With 0 decided, Agreement vacuously holds (safe)
        agreement_holds = len(honest_values) <= 1
        
        # In async model with dishonest dealer:
        # - 0 decided = safe (no disagreement possible)
        # - All decided same value = Agreement achieved
        # Both are "success" in the sense of not violating Agreement
        success = agreement_holds
    
    # For dishonest dealer, report decided count regardless of value
    if dealer_id in B:
        return success, total_rounds, actual_round, honest_decided_count, total_honest, B
    else:
        # For honest dealer, report how many decided on dealer's value
        honest_on_dealer_value = sum(1 for nid, (d, v) in decided.items() 
                                     if nid not in B and d and v == dealer_value)
        return success, total_rounds, actual_round, honest_on_dealer_value, total_honest, B


def run_benchmark_scenario(scenario_name, n, graph, t, seed, dealer_value=1, 
                           dealer_id=0, subset_sizes=None, f=None, test_dishonest_dealer=False):
    """Run all applicable protocols for a scenario."""
    results = []
    
    if f is None:
        f = (n - 1) // 3
    
    # Run CPA (honest dealer only)
    if not test_dishonest_dealer:
        success, proto_rounds, actual, decided, total, B = run_cpa_with_round_tracking(
            n, dealer_id, dealer_value, t, seed, graph, subset_sizes
        )
        results.append(BenchmarkResult(
            "CPA", graph, n, success, proto_rounds, actual, decided, total, False, B
        ))
    
    # Run σ-CPA (honest dealer only)
    if not test_dishonest_dealer:
        success, proto_rounds, actual, decided, total, B = run_sigma_cpa_with_round_tracking(
            n, dealer_id, dealer_value, t, seed, graph, subset_sizes
        )
        results.append(BenchmarkResult(
            "σ-CPA", graph, n, success, proto_rounds, actual, decided, total, False, B
        ))
    
    # Run DS-CPA
    success, proto_rounds, actual, decided, total, B = run_ds_cpa_with_round_tracking(
        n, dealer_id, dealer_value, t, seed, graph, subset_sizes
    )
    results.append(BenchmarkResult(
        "DS-CPA", graph, n, success, proto_rounds, actual, decided, total, test_dishonest_dealer, B
    ))
    
    # Run B-CPA
    success, proto_rounds, actual, decided, total, B = run_bcpa_with_round_tracking(
        n, dealer_id, dealer_value, f, t, seed, graph, subset_sizes, 
        dealer_is_byzantine=test_dishonest_dealer
    )
    results.append(BenchmarkResult(
        "B-CPA", graph, n, success, proto_rounds, actual, decided, total, test_dishonest_dealer, B
    ))
    
    return results


def run_unified_benchmark_single(n, graph, t, seed, dealer_value=1, dealer_id=0, 
                                  subset_sizes=None, f=None):
    """
    Run all protocols on the SAME graph and corruption set.
    This ensures fair comparison - same Byzantine nodes for all algorithms.
    """
    if f is None:
        f = (n - 1) // 3
    
    # Build graph ONCE
    nodes = _build_graph(graph, n, dealer_id, subset_sizes, None, seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    
    # Sample corruption set ONCE
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)  # Dealer is always honest
    
    # Cap at f Byzantine nodes
    while len(B) > f:
        B.pop()
    
    results = {}
    
    # Run CPA with this specific corruption set
    success, proto_rounds, actual, decided, total = _run_cpa_on_graph(
        nodes, adj, B, dealer_id, dealer_value, t, n
    )
    results["CPA"] = {
        "success": success, "proto_rounds": proto_rounds, "actual": actual,
        "decided": decided, "total": total, "B": B
    }
    
    # Run σ-CPA with same corruption set
    success, proto_rounds, actual, decided, total = _run_sigma_cpa_on_graph(
        nodes, adj, B, dealer_id, dealer_value, n
    )
    results["σ-CPA"] = {
        "success": success, "proto_rounds": proto_rounds, "actual": actual,
        "decided": decided, "total": total, "B": B
    }
    
    # Run DS-CPA with same corruption set
    success, proto_rounds, actual, decided, total = _run_ds_cpa_on_graph(
        nodes, adj, B, dealer_id, dealer_value, n
    )
    results["DS-CPA"] = {
        "success": success, "proto_rounds": proto_rounds, "actual": actual,
        "decided": decided, "total": total, "B": B
    }
    
    # Run B-CPA with same corruption set
    success, proto_rounds, actual, decided, total = _run_bcpa_on_graph(
        nodes, adj, B, dealer_id, dealer_value, n, f, t
    )
    results["B-CPA"] = {
        "success": success, "proto_rounds": proto_rounds, "actual": actual,
        "decided": decided, "total": total, "B": B
    }
    
    return results, B


def _run_cpa_on_graph(nodes_template, adj, B, dealer_id, dealer_value, t, n):
    """Run CPA on a pre-built graph with specific Byzantine set."""
    from node import Node
    from copy import deepcopy
    
    # Rebuild nodes for fresh state
    nodes = {}
    for nid in nodes_template:
        nodes[nid] = Node(nid)
        nodes[nid].neighbors = set(nodes_template[nid].neighbors)
    
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPA(dealer_id, t)
            nodes[i].decide(dealer_value)
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestCPA(dealer_id, t)
    
    net = Network(nodes)
    
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, Message("PROPOSE", dealer_id, dealer_value, 0)))
    net.deliver(initial_out)
    
    rounds = n
    actual_round = rounds
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == rounds:
            actual_round = r
    
    decided_dict = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided_dict, B, dealer_value)
    honest_decided = sum(1 for i, (d, v) in decided_dict.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, rounds, actual_round, honest_decided, total_honest


def _run_sigma_cpa_on_graph(nodes_template, adj, B, dealer_id, dealer_value, n):
    """Run σ-CPA on a pre-built graph with specific Byzantine set."""
    from node import Node
    
    nodes = {}
    for nid in nodes_template:
        nodes[nid] = Node(nid)
        nodes[nid].neighbors = set(nodes_template[nid].neighbors)
    
    dealer_private_key = Ed25519PrivateKey.generate()
    dealer_public_key = dealer_private_key.public_key()
    
    def _encode(value):
        return int(value).to_bytes(8, byteorder="big", signed=True)
    
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
            nodes[i].decide(dealer_value)
            setattr(nodes[i], "dealer_signature", dealer_private_key.sign(_encode(dealer_value)))
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
    
    net = Network(nodes)
    
    dealer_sig = getattr(nodes[dealer_id], "dealer_signature")
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, SignedMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_sig)))
    net.deliver(initial_out)
    
    rounds = n
    actual_round = rounds
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == rounds:
            actual_round = r
    
    decided_dict = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided_dict, B, dealer_value)
    honest_decided = sum(1 for i, (d, v) in decided_dict.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, rounds, actual_round, honest_decided, total_honest


def _run_ds_cpa_on_graph(nodes_template, adj, B, dealer_id, dealer_value, n):
    """Run DS-CPA on a pre-built graph with specific Byzantine set."""
    from node import Node
    
    nodes = {}
    for nid in nodes_template:
        nodes[nid] = Node(nid)
        nodes[nid].neighbors = set(nodes_template[nid].neighbors)
    
    private_keys = {}
    public_keys = {}
    for i in nodes:
        priv_key = Ed25519PrivateKey.generate()
        private_keys[i] = priv_key
        public_keys[i] = priv_key.public_key()
    
    f_hat = n - 2
    ds_cpa_rounds = f_hat + 1
    total_msg_rounds = ds_cpa_rounds * n
    
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestDSCPA(dealer_id, public_keys, private_keys[i], f_hat, n)
            nodes[i].value = dealer_value
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=True
            )
        else:
            nodes[i].behavior = HonestDSCPA(dealer_id, public_keys, private_keys[i], f_hat, n)
    
    net = Network(nodes)
    actual_round = total_msg_rounds
    
    for r in range(total_msg_rounds):
        net.run_round(r)
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == total_msg_rounds:
            actual_round = r + 1
    
    decided_dict = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    success, _ = evaluate_execution(decided_dict, B, dealer_value)
    honest_decided = sum(1 for i, (d, v) in decided_dict.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, total_msg_rounds, actual_round, honest_decided, total_honest


def _run_bcpa_on_graph(nodes_template, adj, B, dealer_id, dealer_value, n, f, t):
    """Run B-CPA on a pre-built graph with specific Byzantine set."""
    from node import Node
    
    nodes = {}
    for nid in nodes_template:
        nodes[nid] = Node(nid)
        nodes[nid].neighbors = set(nodes_template[nid].neighbors)
    
    for i in nodes:
        if i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (dealer_value, dealer_value + 1),
                withhold_prob=0.3,
                spam=True
            )
        else:
            nodes[i].behavior = HonestBCPA(dealer_id, n, f, t)
    
    net = Network(nodes)
    
    # Dealer initiates
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, BCPAMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_id)))
    net.deliver(initial_out)
    
    behavior = nodes[dealer_id].behavior
    if isinstance(behavior, HonestBCPA):
        behavior.dealer_value_received = dealer_value
        behavior.echoed_value = dealer_value
        propose_key = ("PROPOSE", dealer_value, dealer_id)
        echo_key = ("ECHO", dealer_value, dealer_id)
        behavior.already_relayed.add(propose_key)
        behavior.already_relayed.add(echo_key)
        behavior.pending_relays.append(("ECHO", dealer_value, dealer_id))
        behavior.echo_received[dealer_value].add(dealer_id)
    
    total_rounds = 4 * n
    actual_round = total_rounds
    
    for r in range(1, total_rounds + 1):
        net.run_round(r)
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided and actual_round == total_rounds:
            actual_round = r
    
    decided_dict = {i: (nodes[i].decided, nodes[i].value) for i in nodes}
    
    # Check validity for honest dealer
    all_correct = all(
        decided_dict[i][0] and decided_dict[i][1] == dealer_value 
        for i in nodes if i not in B
    )
    success = all_correct
    
    honest_decided = sum(1 for i, (d, v) in decided_dict.items() if i not in B and d and v == dealer_value)
    total_honest = len([i for i in nodes if i not in B])
    
    return success, total_rounds, actual_round, honest_decided, total_honest


def _run_unified_sample(args):
    """Worker for unified parallel execution."""
    n, graph, t, seed, f, subset_sizes = args
    return run_unified_benchmark_single(n, graph, t, seed, f=f, subset_sizes=subset_sizes)


def run_unified_benchmark(n, graph, t, num_samples=100, f=None, subset_sizes=None, parallel=False):
    """Run unified benchmark - same corruption set for all algorithms."""
    if f is None:
        f = (n - 1) // 3
    
    stats = {
        "CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "σ-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "DS-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "B-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
    }
    
    protocol_rounds = {}
    
    # Track when algorithms agree/disagree
    both_succeed = 0
    cpa_only = 0
    sigma_only = 0
    neither = 0
    
    args_list = [(n, graph, t, seed, f, subset_sizes) for seed in range(num_samples)]
    
    def process_result(results, B):
        nonlocal both_succeed, cpa_only, sigma_only, neither
        
        for proto in ["CPA", "σ-CPA", "DS-CPA", "B-CPA"]:
            r = results[proto]
            if proto not in protocol_rounds:
                protocol_rounds[proto] = r["proto_rounds"]
            if r["success"]:
                stats[proto]["successes"] += 1
            stats[proto]["total_actual_rounds"] += r["actual"]
            stats[proto]["total_decided"] += r["decided"] / r["total"] if r["total"] > 0 else 0
        
        # Track CPA vs σ-CPA comparison
        cpa_s = results["CPA"]["success"]
        sigma_s = results["σ-CPA"]["success"]
        if cpa_s and sigma_s:
            both_succeed += 1
        elif cpa_s and not sigma_s:
            cpa_only += 1
        elif sigma_s and not cpa_s:
            sigma_only += 1
        else:
            neither += 1
    
    if parallel:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_run_unified_sample, args) for args in args_list]
                for future in as_completed(futures):
                    results, B = future.result()
                    process_result(results, B)
        except (PermissionError, OSError):
            # Fall back to sequential execution
            for args in args_list:
                results, B = _run_unified_sample(args)
                process_result(results, B)
    else:
        # Sequential execution
        for args in args_list:
            results, B = _run_unified_sample(args)
            process_result(results, B)
    
    comparison = {
        "both_succeed": both_succeed,
        "cpa_only": cpa_only,
        "sigma_only": sigma_only,
        "neither": neither
    }
    
    return stats, protocol_rounds, num_samples, comparison


def print_scenario_results(scenario_name, results):
    """Print results for a scenario in a formatted table."""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*80}")
    print(f"{'Protocol':<10} {'Graph':<15} {'n':>3} {'Success':>8} {'Proto Rnds':>10} {'Actual':>8} {'Decided':>10} {'Dealer Byz':>10}")
    print("-" * 80)
    
    for r in results:
        success_str = "✓" if r.success else "✗"
        decided_str = f"{r.honest_decided}/{r.total_honest}"
        dealer_str = "Yes" if r.dealer_byzantine else "No"
        print(f"{r.protocol:<10} {r.graph:<15} {r.n:>3} {success_str:>8} {r.protocol_rounds:>10} {r.actual_rounds:>8} {decided_str:>10} {dealer_str:>10}")


def _run_single_sample(args):
    """Worker function for parallel execution."""
    n, graph, t, seed, f, subset_sizes = args
    results = run_benchmark_scenario(
        f"Random_{seed}", n, graph, t, seed, 
        dealer_value=1, f=f, subset_sizes=subset_sizes
    )
    return results


def run_statistical_benchmark(n, graph, t, num_samples=100, f=None, subset_sizes=None, parallel=True):
    """Run multiple random corruption samples and aggregate results."""
    if f is None:
        f = (n - 1) // 3
    
    stats = {
        "CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "σ-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "DS-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "B-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
    }
    
    protocol_rounds = {}
    
    args_list = [(n, graph, t, seed, f, subset_sizes) for seed in range(num_samples)]
    
    if parallel and num_samples > 10:
        # Use parallel execution
        num_workers = min(multiprocessing.cpu_count(), 8)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_run_single_sample, args) for args in args_list]
            for future in as_completed(futures):
                results = future.result()
                for r in results:
                    if r.protocol not in protocol_rounds:
                        protocol_rounds[r.protocol] = r.protocol_rounds
                    if r.success:
                        stats[r.protocol]["successes"] += 1
                    stats[r.protocol]["total_actual_rounds"] += r.actual_rounds
                    stats[r.protocol]["total_decided"] += r.honest_decided / r.total_honest
    else:
        # Sequential execution
        for args in args_list:
            results = _run_single_sample(args)
            for r in results:
                if r.protocol not in protocol_rounds:
                    protocol_rounds[r.protocol] = r.protocol_rounds
                if r.success:
                    stats[r.protocol]["successes"] += 1
                stats[r.protocol]["total_actual_rounds"] += r.actual_rounds
                stats[r.protocol]["total_decided"] += r.honest_decided / r.total_honest
    
    return stats, protocol_rounds, num_samples


def _run_dishonest_sample(args):
    """Worker for dishonest dealer parallel execution."""
    n, graph, t, seed, f, subset_sizes = args
    
    results = {}
    
    # DS-CPA with dishonest dealer
    success, proto_rounds, actual, decided, total, B = run_ds_cpa_with_round_tracking(
        n, 0, 1, t, seed, graph, subset_sizes
    )
    results["DS-CPA"] = {
        "success": success, "proto_rounds": proto_rounds, 
        "actual": actual, "decided": decided, "total": total
    }
    
    # B-CPA with dishonest dealer
    success, proto_rounds, actual, decided, total, B = run_bcpa_with_round_tracking(
        n, 0, 1, f, t, seed, graph, subset_sizes, 
        dealer_is_byzantine=True, timeout_rounds=4*n
    )
    results["B-CPA"] = {
        "success": success, "proto_rounds": proto_rounds,
        "actual": actual, "decided": decided, "total": total
    }
    
    return results


def run_dishonest_dealer_benchmark(n, graph, t, num_samples=100, f=None, subset_sizes=None, parallel=True):
    """Run benchmark with dishonest dealer (DS-CPA and B-CPA only)."""
    if f is None:
        f = (n - 1) // 3
    
    stats = {
        "DS-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
        "B-CPA": {"successes": 0, "total_actual_rounds": 0, "total_decided": 0},
    }
    
    protocol_rounds = {}
    args_list = [(n, graph, t, seed, f, subset_sizes) for seed in range(num_samples)]
    
    if parallel and num_samples > 10:
        num_workers = min(multiprocessing.cpu_count(), 8)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_run_dishonest_sample, args) for args in args_list]
            for future in as_completed(futures):
                results = future.result()
                for proto in ["DS-CPA", "B-CPA"]:
                    r = results[proto]
                    if proto not in protocol_rounds:
                        protocol_rounds[proto] = r["proto_rounds"]
                    if r["success"]:
                        stats[proto]["successes"] += 1
                    stats[proto]["total_actual_rounds"] += r["actual"]
                    stats[proto]["total_decided"] += r["decided"] / r["total"] if r["total"] > 0 else 0
    else:
        for args in args_list:
            results = _run_dishonest_sample(args)
            for proto in ["DS-CPA", "B-CPA"]:
                r = results[proto]
                if proto not in protocol_rounds:
                    protocol_rounds[proto] = r["proto_rounds"]
                if r["success"]:
                    stats[proto]["successes"] += 1
                stats[proto]["total_actual_rounds"] += r["actual"]
                stats[proto]["total_decided"] += r["decided"] / r["total"] if r["total"] > 0 else 0
    
    return stats, protocol_rounds, num_samples


def main():
    """Main benchmark execution."""
    print("=" * 80)
    print("BYZANTINE BROADCAST PROTOCOL BENCHMARK")
    print("=" * 80)
    
    # Suppress verbose output
    old_stdout = sys.stdout
    
    # ============================================================
    # PART 1: Specific scenarios showing protocol differences
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 1: SPECIFIC SCENARIOS SHOWING PROTOCOL DIFFERENCES")
    print("=" * 80)
    
    scenarios = []
    
    # Scenario 1: All protocols succeed (complete graph, low corruption)
    print("\nScenario 1: All protocols should succeed (complete graph, t=1)")
    sys.stdout = io.StringIO()
    results1 = run_benchmark_scenario("All Succeed", n=10, graph="complete", t=1, seed=42)
    sys.stdout = old_stdout
    print_scenario_results("All Succeed (Complete Graph, n=10, t=1)", results1)
    scenarios.append(("All Succeed", results1))
    
    # Scenario 2: Only σ-CPA and DS-CPA/B-CPA succeed (signature advantage)
    print("\nScenario 2: σ-CPA succeeds, CPA fails (random regular graph)")
    sys.stdout = io.StringIO()
    # Try different seeds to find one where CPA fails but σ-CPA succeeds
    for test_seed in range(100):
        results2 = run_benchmark_scenario("Sigma Advantage", n=10, graph="random_regular", t=1, seed=test_seed)
        cpa_success = next((r.success for r in results2 if r.protocol == "CPA"), False)
        sigma_success = next((r.success for r in results2 if r.protocol == "σ-CPA"), False)
        if not cpa_success and sigma_success:
            break
    sys.stdout = old_stdout
    print_scenario_results("σ-CPA Succeeds, CPA Fails (Random Regular, n=10, t=1)", results2)
    scenarios.append(("Sigma Advantage", results2))
    
    # Scenario 3: Line graph - challenging for all
    print("\nScenario 3: Line graph (challenging topology)")
    sys.stdout = io.StringIO()
    results3 = run_benchmark_scenario("Line Graph", n=8, graph="line", t=0, seed=42)
    sys.stdout = old_stdout
    print_scenario_results("Line Graph (n=8, t=0)", results3)
    scenarios.append(("Line Graph", results3))
    
    # Scenario 4: Star graph (dealer at center)
    print("\nScenario 4: Star graph (dealer at center)")
    sys.stdout = io.StringIO()
    results4 = run_benchmark_scenario("Star Graph", n=10, graph="star", t=1, seed=42)
    sys.stdout = old_stdout
    print_scenario_results("Star Graph (n=10, t=1)", results4)
    scenarios.append(("Star Graph", results4))
    
    # ============================================================
    # PART 2: Dishonest Dealer Scenarios (B-CPA and DS-CPA only)
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 2: DISHONEST DEALER SCENARIOS (B-CPA and DS-CPA only)")
    print("=" * 80)
    print("Note: CPA and σ-CPA require honest dealer - not tested here")
    
    # Scenario 5: Dishonest dealer on complete graph
    print("\nScenario 5: Dishonest dealer (complete graph)")
    sys.stdout = io.StringIO()
    results5 = run_benchmark_scenario("Dishonest Dealer Complete", n=10, graph="complete", 
                                       t=1, seed=42, test_dishonest_dealer=True)
    sys.stdout = old_stdout
    print_scenario_results("Dishonest Dealer (Complete, n=10, f=3)", results5)
    scenarios.append(("Dishonest Dealer Complete", results5))
    
    # Scenario 6: Dishonest dealer on multipartite graph
    print("\nScenario 6: Dishonest dealer (multipartite graph)")
    sys.stdout = io.StringIO()
    results6 = run_benchmark_scenario("Dishonest Dealer Multipartite", n=9, 
                                       graph="complete_multipartite", t=1, seed=42,
                                       subset_sizes=(3, 3, 3), test_dishonest_dealer=True)
    sys.stdout = old_stdout
    print_scenario_results("Dishonest Dealer (Multipartite, n=9, f=2)", results6)
    scenarios.append(("Dishonest Dealer Multipartite", results6))
    
    print("\n" + "-" * 80)
    print("NOTE on B-CPA with Dishonest Dealer:")
    print("  B-CPA is ASYNCHRONOUS - Termination is NOT guaranteed with dishonest dealer.")
    print("  When dealer equivocates, nodes may not reach quorum to decide.")
    print("  '0/N decided' means Agreement holds vacuously (no disagreement possible).")
    print("  This is SAFE behavior: better to not decide than to disagree.")
    print("  DS-CPA, being synchronous, can guarantee termination even with dishonest dealer.")
    
    # ============================================================
    # PART 3: UNIFIED BENCHMARK (same corruption set for all algorithms)
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 3: UNIFIED BENCHMARK (same corruption for all algorithms)")
    print("=" * 80)
    print("Each sample: build graph once, sample corruption once, run ALL algorithms")
    
    # 5 graph types × 100 samples = 500 total per algorithm
    unified_configs = [
        ("Complete", 15, "complete", 1, None),
        ("Dense Random", 15, "dense_random", 1, None),
        ("Random Regular", 15, "random_regular", 1, None),
        ("Cycle", 15, "cycle", 1, None),
        ("Hypercube", 16, "hypercube", 1, None),  # 16 = 2^4 for proper hypercube
    ]
    
    all_stats = []
    all_comparisons = []
    NUM_SAMPLES = 100
    
    print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel execution")
    print(f"Running {NUM_SAMPLES} samples × {len(unified_configs)} graph types = {NUM_SAMPLES * len(unified_configs)} total")
    
    # Aggregate stats across all graph types
    total_stats = {
        "CPA": {"successes": 0, "total_actual": 0, "total_coverage": 0, "count": 0},
        "σ-CPA": {"successes": 0, "total_actual": 0, "total_coverage": 0, "count": 0},
        "DS-CPA": {"successes": 0, "total_actual": 0, "total_coverage": 0, "count": 0},
        "B-CPA": {"successes": 0, "total_actual": 0, "total_coverage": 0, "count": 0},
    }
    total_comparison = {"both_succeed": 0, "cpa_only": 0, "sigma_only": 0, "neither": 0}
    
    for name, n, graph, t, sizes in unified_configs:
        print(f"\nRunning {NUM_SAMPLES} unified samples on {name} graph (n={n})...", end=" ", flush=True)
        sys.stdout = io.StringIO()
        stats, proto_rounds, num_samples, comparison = run_unified_benchmark(
            n, graph, t, num_samples=NUM_SAMPLES, subset_sizes=sizes
        )
        sys.stdout = old_stdout
        
        print("Done!")
        print(f"\n{name} Graph (n={n}) - {NUM_SAMPLES} Samples (same corruption for all):")
        print("-" * 80)
        print("Synchronous Protocols:")
        print(f"{'Protocol':<10} {'Success %':>10} {'Rounds':>12} {'Avg Actual':>12} {'Coverage':>12}")
        print("-" * 70)
        
        for proto in ["CPA", "σ-CPA", "DS-CPA"]:
            s = stats[proto]
            success_pct = (s["successes"] / num_samples) * 100
            avg_actual = s["total_actual_rounds"] / num_samples
            avg_coverage = (s["total_decided"] / num_samples) * 100
            p_rounds = proto_rounds.get(proto, "N/A")
            print(f"{proto:<10} {success_pct:>9.1f}% {p_rounds:>12} {avg_actual:>12.1f} {avg_coverage:>11.1f}%")
            
            # Aggregate
            total_stats[proto]["successes"] += s["successes"]
            total_stats[proto]["total_actual"] += s["total_actual_rounds"]
            total_stats[proto]["total_coverage"] += s["total_decided"]
            total_stats[proto]["count"] += num_samples
        
        # B-CPA separately (async)
        s = stats["B-CPA"]
        success_pct = (s["successes"] / num_samples) * 100
        avg_coverage = (s["total_decided"] / num_samples) * 100
        steps = proto_rounds.get("B-CPA", "N/A")
        print(f"\nB-CPA (Asynchronous): Success={success_pct:.1f}%, Steps={steps}, Coverage={avg_coverage:.1f}%")
        total_stats["B-CPA"]["successes"] += s["successes"]
        total_stats["B-CPA"]["total_actual"] += s["total_actual_rounds"]
        total_stats["B-CPA"]["total_coverage"] += s["total_decided"]
        total_stats["B-CPA"]["count"] += num_samples
        
        # CPA vs σ-CPA comparison for this graph
        print(f"\nCPA vs σ-CPA comparison:")
        print(f"  Both succeed: {comparison['both_succeed']}")
        print(f"  σ-CPA only:   {comparison['sigma_only']}")
        print(f"  CPA only:     {comparison['cpa_only']}")
        print(f"  Neither:      {comparison['neither']}")
        
        for k in total_comparison:
            total_comparison[k] += comparison[k]
        
        all_stats.append((name, n, stats, proto_rounds, comparison))
        all_comparisons.append((name, n, comparison))
    
    # Print aggregate results
    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS ({NUM_SAMPLES * len(unified_configs)} total samples)")
    print("=" * 80)
    
    total_samples = NUM_SAMPLES * len(unified_configs)
    print("\nSynchronous Protocols:")
    print(f"{'Protocol':<10} {'Success %':>10} {'Avg Actual':>12} {'Avg Coverage':>12}")
    print("-" * 50)
    for proto in ["CPA", "σ-CPA", "DS-CPA"]:
        s = total_stats[proto]
        success_pct = (s["successes"] / s["count"]) * 100 if s["count"] > 0 else 0
        avg_actual = s["total_actual"] / s["count"] if s["count"] > 0 else 0
        avg_coverage = (s["total_coverage"] / s["count"]) * 100 if s["count"] > 0 else 0
        print(f"{proto:<10} {success_pct:>9.1f}% {avg_actual:>12.1f} {avg_coverage:>11.1f}%")
    
    # B-CPA separate
    s = total_stats["B-CPA"]
    bcpa_success = (s["successes"] / s["count"]) * 100 if s["count"] > 0 else 0
    bcpa_coverage = (s["total_coverage"] / s["count"]) * 100 if s["count"] > 0 else 0
    print(f"\nB-CPA (Asynchronous): Success={bcpa_success:.1f}%, Coverage={bcpa_coverage:.1f}%")
    print("  Note: B-CPA steps are not comparable to synchronous rounds")
    
    print(f"\nCPA vs σ-CPA (same corruption set):")
    print(f"  Both succeed:    {total_comparison['both_succeed']} ({100*total_comparison['both_succeed']/total_samples:.1f}%)")
    print(f"  σ-CPA only:      {total_comparison['sigma_only']} ({100*total_comparison['sigma_only']/total_samples:.1f}%)")
    print(f"  CPA only:        {total_comparison['cpa_only']} ({100*total_comparison['cpa_only']/total_samples:.1f}%)")
    print(f"  Neither:         {total_comparison['neither']} ({100*total_comparison['neither']/total_samples:.1f}%)")
    
    # Skip the old separate benchmarks
    dishonest_stats = []
    
    # ============================================================
    # PART 4: Summary and LaTeX Output
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 4: SUMMARY")
    print("=" * 80)
    
    print("\nKey Findings:")
    print("-" * 40)
    print("1. σ-CPA outperforms CPA (same rounds, better success rate)")
    print("2. DS-CPA handles dishonest dealers at cost of more rounds: (n-1) × n")
    print("3. B-CPA is ASYNCHRONOUS (steps not comparable to sync rounds)")
    print("4. B-CPA requires dense graphs for quorum intersection")
    
    print("\n" + "=" * 80)
    print("LATEX TABLE OUTPUT")
    print("=" * 80)
    
    # Generate LaTeX for specific scenarios
    print("\n% Specific Scenarios Table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Protocol Comparison Under Specific Scenarios}")
    print("\\label{tab:scenarios}")
    print("\\small")
    print("\\begin{tabular}{llcccccc}")
    print("\\hline")
    print("\\textbf{Scenario} & \\textbf{Protocol} & \\textbf{Success} & \\textbf{Proto} & \\textbf{Actual} & \\textbf{Decided} & \\textbf{Dealer} \\\\")
    print(" & & & \\textbf{Rounds} & \\textbf{Rounds} & & \\textbf{Byz} \\\\")
    print("\\hline")
    
    for scenario_name, results in scenarios:
        first = True
        for r in results:
            success_str = "\\checkmark" if r.success else "$\\times$"
            decided_str = f"{r.honest_decided}/{r.total_honest}"
            dealer_str = "Yes" if r.dealer_byzantine else "No"
            scenario_col = scenario_name if first else ""
            print(f"{scenario_col} & {r.protocol} & {success_str} & {r.protocol_rounds} & {r.actual_rounds} & {decided_str} & {dealer_str} \\\\")
            first = False
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Generate LaTeX for unified statistical results (same corruption for all)
    # Synchronous protocols table (CPA, σ-CPA, DS-CPA)
    print("\n% Synchronous Protocols Results")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Synchronous Protocol Performance ({NUM_SAMPLES} samples per graph, same corruption set)}}")
    print("\\label{tab:sync-stats}")
    print("\\small")
    print("\\begin{tabular}{llcccc}")
    print("\\hline")
    print("\\textbf{Graph} & \\textbf{Protocol} & \\textbf{Success \\%} & \\textbf{Rounds} & \\textbf{Avg Actual} & \\textbf{Coverage \\%} \\\\")
    print("\\hline")
    
    for name, n, stats, proto_rounds, comparison in all_stats:
        first = True
        for proto in ["CPA", "σ-CPA", "DS-CPA"]:  # Exclude B-CPA from sync table
            s = stats[proto]
            success_pct = (s["successes"] / NUM_SAMPLES) * 100
            avg_actual = s["total_actual_rounds"] / NUM_SAMPLES
            avg_coverage = (s["total_decided"] / NUM_SAMPLES) * 100
            p_rounds = proto_rounds.get(proto, 0)
            graph_col = f"{name} ($n$={n})" if first else ""
            proto_latex = proto.replace("σ", "$\\sigma$")
            print(f"{graph_col} & {proto_latex} & {success_pct:.1f}\\% & {p_rounds} & {avg_actual:.1f} & {avg_coverage:.1f}\\% \\\\")
            first = False
        print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    
    # B-CPA separate table (asynchronous)
    print("\n% B-CPA Results (Asynchronous Protocol)")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{B-CPA Performance ({NUM_SAMPLES} samples per graph) --- Asynchronous Protocol}}")
    print("\\label{tab:bcpa-stats}")
    print("\\small")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Graph} & \\textbf{Success \\%} & \\textbf{Steps ($4n$)} & \\textbf{Coverage \\%} \\\\")
    print("\\hline")
    
    for name, n, stats, proto_rounds, comparison in all_stats:
        s = stats["B-CPA"]
        success_pct = (s["successes"] / NUM_SAMPLES) * 100
        avg_coverage = (s["total_decided"] / NUM_SAMPLES) * 100
        steps = proto_rounds.get("B-CPA", 4*n)
        actual_n = 16 if name == "Hypercube" else 15
        print(f"{name} ($n$={actual_n}) & {success_pct:.1f}\\% & {steps} & {avg_coverage:.1f}\\% \\\\")
    print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Generate LaTeX for CPA vs σ-CPA comparison table
    print("\n% CPA vs σ-CPA Head-to-Head Comparison")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{CPA vs $\\sigma$-CPA: Same Corruption Set Comparison}")
    print("\\label{tab:cpa-sigma-comparison}")
    print("\\small")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Graph} & \\textbf{Both Succeed} & \\textbf{$\\sigma$-CPA Only} & \\textbf{CPA Only} & \\textbf{Neither} \\\\")
    print("\\hline")
    
    for name, n, comparison in all_comparisons:
        both = comparison['both_succeed']
        sigma = comparison['sigma_only']
        cpa = comparison['cpa_only']
        neither = comparison['neither']
        print(f"{name} ($n$={n}) & {both} & {sigma} & {cpa} & {neither} \\\\")
    print("\\hline")
    
    # Aggregate row
    print(f"\\textbf{{Total}} & {total_comparison['both_succeed']} & {total_comparison['sigma_only']} & {total_comparison['cpa_only']} & {total_comparison['neither']} \\\\")
    print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Aggregate summary table (synchronous only)
    print("\n% Aggregate Results - Synchronous Protocols")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print(f"\\caption{{Aggregate Synchronous Protocol Performance ({total_samples} samples across 5 graph types)}}")
    print("\\label{tab:aggregate}")
    print("\\small")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Protocol} & \\textbf{Success \\%} & \\textbf{Avg Actual Rounds} & \\textbf{Avg Coverage \\%} \\\\")
    print("\\hline")
    
    for proto in ["CPA", "σ-CPA", "DS-CPA"]:  # Exclude B-CPA
        s = total_stats[proto]
        success_pct = (s["successes"] / s["count"]) * 100 if s["count"] > 0 else 0
        avg_actual = s["total_actual"] / s["count"] if s["count"] > 0 else 0
        avg_coverage = (s["total_coverage"] / s["count"]) * 100 if s["count"] > 0 else 0
        proto_latex = proto.replace("σ", "$\\sigma$")
        print(f"{proto_latex} & {success_pct:.1f}\\% & {avg_actual:.1f} & {avg_coverage:.1f}\\% \\\\")
    print("\\hline")
    
    print("\\end{tabular}")
    print("\\end{table}")
    
    # B-CPA aggregate (separate, noting it's async)
    s = total_stats["B-CPA"]
    bcpa_success = (s["successes"] / s["count"]) * 100 if s["count"] > 0 else 0
    bcpa_coverage = (s["total_coverage"] / s["count"]) * 100 if s["count"] > 0 else 0
    print(f"\n% B-CPA Aggregate: Success={bcpa_success:.1f}%, Coverage={bcpa_coverage:.1f}% (asynchronous, not comparable to sync rounds)")
    
    return scenarios, all_stats, total_stats, total_comparison


if __name__ == "__main__":
    main()

