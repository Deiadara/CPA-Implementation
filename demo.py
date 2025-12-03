#!/usr/bin/env python3
"""
Demo script for CPA protocol family with clean output for screenshots.

Run with: python demo.py [protocol]
Options: cpa, sigma, ds, bcpa, all

Author: Nikos
"""

import sys
import random
from node import Node, connect_nodes
from network import Network, Message
from CPA import (
    HonestCPA,
    HonestCPAWithDealerSignature,
    HonestDSCPA,
    HonestBCPA,
    SignedMessage,
    DSCPAMessage,
    BCPAMessage,
    _encode_value_bytes,
    _build_graph,
)
from adversary import ByzantineEquivocator
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from utils import sample_t_local_faulty_set


def build_dense_random_graph(n, seed=None):
    """Build a dense random graph using Erdos-Renyi G(n, 0.6)."""
    return _build_graph("dense_random", n, dealer_id=0, subset_sizes=None, 
                        custom_graph_path=None, seed=seed)


def build_random_regular_graph(n, degree=4, seed=None):
    """Build a random regular graph where each node has the same degree."""
    return _build_graph("random_regular", n, dealer_id=0, subset_sizes=None,
                        custom_graph_path=None, seed=seed)


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print("")
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_subheader(title: str):
    print("")
    print("--- %s ---" % title)


def print_graph_info(nodes, adj, B, dealer_id, t, graph_type="random"):
    """Print graph and Byzantine set information."""
    n = len(nodes)
    
    # Compute average degree
    total_edges = sum(len(neighbors) for neighbors in adj.values())
    avg_degree = total_edges / n if n > 0 else 0
    
    print("")
    print("Network Configuration:")
    print("  Nodes: %d" % n)
    print("  Topology: %s (avg degree: %.1f)" % (graph_type, avg_degree))
    print("  Dealer: Node %d" % dealer_id)
    print("  t-local bound: %d (each node has at most %d Byzantine neighbors)" % (t, t))
    if B:
        print("  Byzantine set B: %s (sampled randomly)" % B)
    else:
        print("  Byzantine set B: (empty)")
    print("  Honest nodes: %s" % sorted(set(nodes.keys()) - B))


def demo_cpa(n=15, t=3, dealer_id=0, dealer_value=42, seed=77):
    """Demo of basic CPA protocol."""
    print_header("CPA Protocol Demo")
    
    nodes = build_dense_random_graph(n, seed=seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    
    print_graph_info(nodes, adj, B, dealer_id, t, graph_type="Dense Random (Erdos-Renyi p=0.6)")
    
    # Assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPA(dealer_id, t)
            nodes[i].decide(dealer_value)
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=False
            )
        else:
            nodes[i].behavior = HonestCPA(dealer_id, t)
    
    net = Network(nodes)
    
    # Initial broadcast
    print_subheader("Round 0: Dealer Broadcasts")
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, Message("PROPOSE", dealer_id, dealer_value, 0)))
    print("  Dealer %d sends value=%d to all %d neighbors" % (dealer_id, dealer_value, len(nodes[dealer_id].neighbors)))
    net.deliver(initial_out)
    
    # Run protocol
    rounds = n
    decided_at = {}
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        
        # Track new decisions
        new_decisions = []
        for i in sorted(nodes.keys()):
            if i not in B and nodes[i].decided and i not in decided_at:
                decided_at[i] = r
                new_decisions.append(i)
        
        if new_decisions:
            print_subheader("Round %d" % r)
            for i in new_decisions:
                print("  Node %d decides on value=%s" % (i, nodes[i].value))
        
        # Check if all honest decided
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided:
            print("")
            print("  [All honest nodes have decided by round %d]" % r)
            break
    
    # Final results
    print_subheader("Final Results")
    success = True
    for i in sorted(nodes.keys()):
        if i in B:
            print("  Node %d: Byzantine (corrupted)" % i)
        else:
            status = "decided" if nodes[i].decided else "undecided"
            value = nodes[i].value if nodes[i].decided else "-"
            if nodes[i].decided and nodes[i].value == dealer_value:
                correct = "[OK]"
            else:
                correct = "[FAIL]"
                success = False
            print("  Node %d: %s, value=%s %s" % (i, status, value, correct))
    
    print("")
    if success:
        print("  RESULT: SUCCESS - All honest nodes decided on dealer's value")
    else:
        print("  RESULT: FAILURE - Some honest nodes failed to decide correctly")


def demo_sigma_cpa(n=15, t=3, dealer_id=0, dealer_value=42, seed=55):
    """Demo of sigma-CPA (CPA with dealer signatures)."""
    print_header("sigma-CPA Protocol Demo (Dealer Signatures)")
    
    nodes = build_dense_random_graph(n, seed=seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(dealer_id)
    
    print_graph_info(nodes, adj, B, dealer_id, t, graph_type="Dense Random (Erdos-Renyi p=0.6)")
    
    # Generate dealer keypair
    dealer_private_key = Ed25519PrivateKey.generate()
    dealer_public_key = dealer_private_key.public_key()
    print("")
    print("  Dealer's public key distributed to all nodes (Ed25519)")
    
    # Assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
            nodes[i].decide(dealer_value)
            setattr(nodes[i], "dealer_signature", 
                    dealer_private_key.sign(_encode_value_bytes(dealer_value)))
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=False
            )
        else:
            nodes[i].behavior = HonestCPAWithDealerSignature(dealer_public_key)
    
    net = Network(nodes)
    
    # Initial broadcast with signature
    print_subheader("Round 0: Dealer Broadcasts with Signature")
    dealer_sig = getattr(nodes[dealer_id], "dealer_signature")
    initial_out = []
    for nid in nodes[dealer_id].neighbors:
        initial_out.append((nid, SignedMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_sig)))
    print("  Dealer %d sends value=%d with signature to all neighbors" % (dealer_id, dealer_value))
    net.deliver(initial_out)
    
    # Run protocol
    rounds = n
    decided_at = {}
    
    for r in range(1, rounds + 1):
        net.run_round(r)
        
        new_decisions = []
        for i in sorted(nodes.keys()):
            if i not in B and nodes[i].decided and i not in decided_at:
                decided_at[i] = r
                new_decisions.append(i)
        
        if new_decisions:
            print_subheader("Round %d" % r)
            for i in new_decisions:
                print("  Node %d verifies signature and decides on value=%s" % (i, nodes[i].value))
        
        all_decided = all(nodes[i].decided for i in nodes if i not in B)
        if all_decided:
            print("")
            print("  [All honest nodes have decided by round %d]" % r)
            break
    
    # Final results
    print_subheader("Final Results")
    success = True
    for i in sorted(nodes.keys()):
        if i in B:
            print("  Node %d: Byzantine (signature forgery impossible)" % i)
        else:
            status = "decided" if nodes[i].decided else "undecided"
            value = nodes[i].value if nodes[i].decided else "-"
            if nodes[i].decided and nodes[i].value == dealer_value:
                correct = "[OK]"
            else:
                correct = "[FAIL]"
                success = False
            print("  Node %d: %s, value=%s %s" % (i, status, value, correct))
    
    print("")
    if success:
        print("  RESULT: SUCCESS")
    else:
        print("  RESULT: FAILURE")
    print("  Note: Byzantine nodes cannot forge dealer's signature")


def demo_ds_cpa(n=10, t=2, sender_id=0, sender_value=42, seed=0):
    """Demo of DS-CPA (Dolev-Strong with CPA)."""
    print_header("DS-CPA Protocol Demo (Dolev-Strong)")
    
    nodes = build_dense_random_graph(n, seed=seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    B.discard(sender_id)
    
    print_graph_info(nodes, adj, B, sender_id, t, graph_type="Dense Random (Erdos-Renyi p=0.6)")
    
    # Generate keypairs for all nodes
    private_keys = {}
    public_keys = {}
    for i in nodes:
        priv_key = Ed25519PrivateKey.generate()
        private_keys[i] = priv_key
        public_keys[i] = priv_key.public_key()
    print("")
    print("  All nodes have Ed25519 keypairs (for signature chains)")
    
    f_hat = n - 2
    ds_rounds = f_hat + 1
    total_msg_rounds = ds_rounds * n
    
    print("")
    print("  f_hat = n - 2 = %d" % f_hat)
    print("  DS-CPA rounds: %d (rounds 0 to %d)" % (ds_rounds, f_hat))
    print("  Each DS-round: %d message-passing rounds" % n)
    print("  Total rounds: %d" % total_msg_rounds)
    
    # Assign behaviors
    for i in nodes:
        if i == sender_id:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, n)
            nodes[i].value = sender_value
        elif i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (0, 1),
                withhold_prob=0.2,
                spam=False
            )
        else:
            nodes[i].behavior = HonestDSCPA(sender_id, public_keys, private_keys[i], f_hat, n)
    
    net = Network(nodes)
    
    # Run protocol
    print_subheader("Protocol Execution")
    last_ds_round = -1
    
    for r in range(total_msg_rounds):
        net.run_round(r)
        ds_round = r // n
        
        # Print at DS-round boundaries
        if ds_round != last_ds_round:
            last_ds_round = ds_round
            print("")
            print("  DS-Round %d:" % ds_round)
            
            for i in sorted(nodes.keys()):
                if i not in B:
                    behavior = nodes[i].behavior
                    if isinstance(behavior, HonestDSCPA) and behavior.extracted_set:
                        print("    Node %d extracted set: %s" % (i, behavior.extracted_set))
    
    # Final results
    print_subheader("Final Results (end of DS-Round %d)" % f_hat)
    
    honest_values = set()
    for i in sorted(nodes.keys()):
        if i in B:
            print("  Node %d: Byzantine" % i)
        else:
            if nodes[i].decided:
                print("  Node %d: decided on value=%s" % (i, nodes[i].value))
                honest_values.add(nodes[i].value)
            else:
                print("  Node %d: undecided (defaults to 0)" % i)
    
    agreement = len(honest_values) <= 1
    print("")
    if agreement:
        print("  Agreement: HOLDS")
    else:
        print("  Agreement: VIOLATED")
    print("  Note: DS-CPA guarantees termination even with dishonest dealer")


def demo_bcpa(n=15, f=4, t=2, dealer_id=0, dealer_value=42, seed=22, dishonest_dealer=False):
    """Demo of B-CPA (Bracha's reliable broadcast with CPA)."""
    title = "B-CPA Protocol Demo (Bracha's Broadcast)"
    if dishonest_dealer:
        title += " - DISHONEST DEALER"
    print_header(title)
    
    nodes = build_dense_random_graph(n, seed=seed)
    adj = {i: set(nodes[i].neighbors) for i in nodes}
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    
    if dishonest_dealer:
        B.add(dealer_id)
    else:
        B.discard(dealer_id)
    
    # Cap at f
    while len(B) > f:
        B.pop()
    
    print_graph_info(nodes, adj, B, dealer_id, t, graph_type="Dense Random (Erdos-Renyi p=0.6)")
    print("  Global fault bound f: %d (requires n >= 3f+1 = %d)" % (f, 3*f+1))
    print("  Quorum threshold n-f: %d" % (n - f))
    
    if dishonest_dealer:
        print("")
        print("  WARNING: DEALER IS BYZANTINE - will equivocate")
    
    # Assign behaviors
    for i in nodes:
        if i in B:
            nodes[i].behavior = ByzantineEquivocator(
                value_picker=lambda rnd: (dealer_value, dealer_value + 1),
                withhold_prob=0.3,
                spam=False
            )
        else:
            nodes[i].behavior = HonestBCPA(dealer_id, n, f, t)
    
    net = Network(nodes)
    
    # Dealer initiates
    print_subheader("Phase 0: PROPOSE")
    if dealer_id not in B:
        initial_out = []
        for nid in nodes[dealer_id].neighbors:
            initial_out.append((nid, BCPAMessage("PROPOSE", dealer_id, dealer_value, 0, dealer_id)))
        print("  Dealer %d sends PROPOSE(value=%d) to all neighbors" % (dealer_id, dealer_value))
        net.deliver(initial_out)
        
        # Dealer echoes
        behavior = nodes[dealer_id].behavior
        if isinstance(behavior, HonestBCPA):
            behavior.dealer_value_received = dealer_value
            behavior.echoed_value = dealer_value
            behavior.already_relayed.add(("PROPOSE", dealer_value, dealer_id))
            behavior.already_relayed.add(("ECHO", dealer_value, dealer_id))
            behavior.pending_relays.append(("ECHO", dealer_value, dealer_id))
            behavior.echo_received[dealer_value].add(dealer_id)
    else:
        # Byzantine dealer equivocates
        neighbors = list(nodes[dealer_id].neighbors)
        initial_out = []
        half = len(neighbors) // 2
        for i, nid in enumerate(neighbors):
            val = dealer_value if i < half else dealer_value + 1
            initial_out.append((nid, BCPAMessage("PROPOSE", dealer_id, val, 0, dealer_id)))
        print("  Byzantine dealer sends value=%d to half, value=%d to other half" % (dealer_value, dealer_value + 1))
        net.deliver(initial_out)
    
    # Run protocol
    total_rounds = 4 * n
    phase_names = ["PROPOSE", "ECHO", "VOTE", "DELIVER"]
    
    for r in range(1, total_rounds + 1):
        net.run_round(r)
        
        phase = (r - 1) // n
        round_in_phase = (r - 1) % n
        
        # Print at phase boundaries
        if round_in_phase == 0 and phase <= 3:
            print_subheader("Phase %d: %s" % (phase, phase_names[phase]))
            
            # Show echo/vote counts
            for i in sorted(nodes.keys()):
                if i not in B:
                    behavior = nodes[i].behavior
                    if isinstance(behavior, HonestBCPA):
                        echo_info = {v: len(s) for v, s in behavior.echo_received.items() if s}
                        vote_info = {v: len(s) for v, s in behavior.vote_received.items() if s}
                        if echo_info or vote_info:
                            parts = []
                            if echo_info:
                                parts.append("echoes=%s" % echo_info)
                            if vote_info:
                                parts.append("votes=%s" % vote_info)
                            print("    Node %d: %s" % (i, ", ".join(parts)))
        
        # Show decisions
        if r == total_rounds:
            decided_this_phase = [i for i in nodes if i not in B and nodes[i].decided]
            if decided_this_phase:
                for i in decided_this_phase:
                    print("    Node %d DELIVERS value=%s" % (i, nodes[i].value))
    
    # Final results
    print_subheader("Final Results")
    
    honest_values = set()
    decided_count = 0
    total_honest = len([i for i in nodes if i not in B])
    
    for i in sorted(nodes.keys()):
        if i in B:
            print("  Node %d: Byzantine" % i)
        else:
            if nodes[i].decided:
                decided_count += 1
                honest_values.add(nodes[i].value)
                print("  Node %d: DELIVERED value=%s" % (i, nodes[i].value))
            else:
                print("  Node %d: did not deliver" % i)
    
    # Evaluate
    print("")
    print("  Decided: %d/%d honest nodes" % (decided_count, total_honest))
    agreement = len(honest_values) <= 1
    
    if agreement:
        if decided_count == 0:
            print("  Agreement: HOLDS (vacuously - no decisions)")
        else:
            print("  Agreement: HOLDS (all decided on %s)" % list(honest_values)[0])
    else:
        print("  Agreement: VIOLATED")
    
    if dishonest_dealer:
        print("")
        print("  Note: With dishonest dealer, termination is NOT guaranteed")
        print("  Note: 0 decisions = SAFE behavior (Agreement holds)")
        if agreement:
            print("")
            print("  RESULT: SUCCESS (Agreement maintained)")
    else:
        validity = honest_values <= {dealer_value}
        if validity:
            print("  Validity: HOLDS")
        else:
            print("  Validity: VIOLATED")
        success = agreement and validity and decided_count == total_honest
        print("")
        if success:
            print("  RESULT: SUCCESS")
        else:
            print("  RESULT: FAILURE")


def main():
    """Main demo entry point."""
    if len(sys.argv) < 2:
        protocol = "all"
    else:
        protocol = sys.argv[1].lower()
    
    print("")
    print("=" * 70)
    print("BYZANTINE BROADCAST PROTOCOL DEMOS".center(70))
    print("=" * 70)
    print("")
    print("These demos show protocol executions with multiple Byzantine nodes.")
    print("Parameters: n=15, t=3 (each node has up to 3 Byzantine neighbors).")
    print("Dense random graphs with randomly sampled t-local Byzantine sets.")
    
    if protocol in ["cpa", "all"]:
        demo_cpa()
    
    if protocol in ["sigma", "sigma-cpa", "all"]:
        demo_sigma_cpa()
    
    if protocol in ["ds", "ds-cpa", "dolev", "all"]:
        demo_ds_cpa()
    
    if protocol in ["bcpa", "b-cpa", "bracha", "all"]:
        demo_bcpa(dishonest_dealer=False)
    
    if protocol in ["bcpa-dishonest", "bcpa_byz", "all"]:
        demo_bcpa(dishonest_dealer=True)
    
    if protocol not in ["cpa", "sigma", "sigma-cpa", "ds", "ds-cpa", "dolev", 
                        "bcpa", "b-cpa", "bracha", "bcpa-dishonest", "bcpa_byz", "all"]:
        print("")
        print("Unknown protocol: %s" % protocol)
        print("Options: cpa, sigma, ds, bcpa, bcpa-dishonest, all")
    
    print("")
    print("=" * 70)
    print("END OF DEMOS".center(70))
    print("=" * 70)
    print("")


if __name__ == "__main__":
    main()

