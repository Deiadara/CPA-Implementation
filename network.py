from dataclasses import dataclass


class Network:
    def __init__(self, nodes: dict[int, "Node"]):
        self.nodes = nodes

    def deliver(self, outgoing):
        for to_id, msg in outgoing:
            if to_id in self.nodes:
                self.nodes[to_id].receive(msg)

    def run_round(self, rnd):
        for node in self.nodes.values():
            node.process_inbox()

        outgoing = []
        for node in self.nodes.values():
            outgoing += node.act(rnd)

        self.deliver(outgoing)

@dataclass
class Message:
    mtype: str
    sender: int
    value: int
    rnd: int
