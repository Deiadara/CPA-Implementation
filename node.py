from collections import defaultdict
from typing import Optional

class Node:
    def __init__(self, node_id, behavior: Optional["Behavior"] = None):
        self.id = node_id
        self.behavior = behavior
        self.neighbors = set()
        self.decided = False
        self.value = None
        self.already_broadcast = False

        # For honest behavior bookkeeping
        self.received_from = defaultdict(set)  # value -> {sender_ids}
        self.seen_from_dealer = set()

        # inbox for current round
        self.inbox = []

    def add_neighbor(self, nid):
        self.neighbors.add(nid)

    def get_neighbours(self):
        return list(self.neighbors)

    def receive(self, msg: "Message"):
        self.inbox.append(msg)

    def decide(self, v):
        if not self.decided:
            self.decided = True
            self.value = v

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_decided(self, decided):
        self.decided = decided

    def get_decided(self):
        return self.decided

    def process_inbox(self):
        for m in self.inbox:
            if self.behavior:
                self.behavior.on_receive(self, m)
        self.inbox.clear()

    def act(self, rnd):
        if self.behavior:
            return self.behavior.on_round(self, rnd)
        return []

def connect_nodes(n1, n2):
    n1.add_neighbor(n2.id)
    n2.add_neighbor(n1.id)



