from adversary import ByzantineEquivocator
from graphs import build_line_graph, build_complete_graph, build_complete_multipartite_graph
from network import Message, Network
from utils import sample_t_local_faulty_set

ROUNDS = 100

class Behavior:
    def on_round(self, node, rnd):
        """Return list of (neighbor_id, Message) to send this round."""
        return []

    def on_receive(self, node, msg: Message):
        """Update node state on message reception."""

class CPA:
    def __init__(self, nodes, dealer, value, t):
        self.nodes = nodes
        self.dealer = dealer
        self.value = value
        self.t = t

    def run(self):
        print("CPA started")

        self.dealer.set_value(self.value)
        self.dealer.set_decided(True)

        for node in self.dealer.get_neighbours():
            self.dealer.propose_value(self.value, node)
            node.set_value(self.value)
            node.set_decided(True)
            for neighbour in node.get_neighbours():
                node.propose_value(self.value, neighbour)


        for round in range(ROUNDS):
            for node in self.nodes:
                if node.get_decided():
                    continue
                for value in node.get_potential_values():
                    if node.get_potential_values()[value] > self.t:
                        print(f"Node {node.get_id()} decided on value {value} in round {round}")
                        node.set_value(value)
                        node.set_decided(True)
                        for neighbour in node.get_neighbours():
                            node.propose_value(value, neighbour)

        print("CPA finished")

    def get_nodes(self):
        return self.nodes

    def get_dealer(self):
        return self.dealer

    def get_value(self):
        return self.value

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

def run_cpa_with_adversary(n=10, dealer_id=0, dealer_value=1, t=0, seed=None):
    #nodes = build_line_graph(n, dealer_id)
    #nodes = build_complete_graph(n, dealer_id)
    nodes = build_complete_multipartite_graph(n, dealer_id, (3,3,3))
    adj = {i: set(nodes[i].neighbors) for i in nodes}

    # sample a t-local faulty set
    B = sample_t_local_faulty_set(adj, t=t, seed=seed)
    print(f"Graph topology: {adj}")
    print(f"Byzantine set (t-local): {B}")

    # assign behaviors
    for i in nodes:
        if i == dealer_id:
            nodes[i].behavior = HonestCPA(dealer_id, t)
            nodes[i].decide(dealer_value)   # dealer is correct & starts decided
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

    for r in range(1, 10):  # Reduced rounds for debugging
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
