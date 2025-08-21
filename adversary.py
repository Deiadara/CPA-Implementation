import random

from network import Message

class ByzantineCrash:
    def on_round(self, node, rnd):
        return []

    def on_receive(self, node, msg):
        pass

class ByzantineEquivocator:
    def __init__(self, value_picker=lambda rnd: (rnd % 2, 1 - (rnd % 2)), withhold_prob=0.0, spam=False):
        self.value_picker = value_picker
        self.withhold_prob = withhold_prob
        self.spam = spam

    def on_receive(self, node, msg):
        pass

    def on_round(self, node, rnd):
        if random.random() < self.withhold_prob:
            return []

        vA, vB = self.value_picker(rnd)
        neigh = list(node.neighbors)
        random.shuffle(neigh)
        half = len(neigh) // 2
        groupA, groupB = neigh[:half], neigh[half:]

        msgs = []
        for nid in groupA:
            msgs.append((nid, Message("PROPOSE", node.id, vA, rnd)))
        for nid in groupB:
            msgs.append((nid, Message("PROPOSE", node.id, vB, rnd)))

        if self.spam:
            spam_targets = random.sample(neigh, k=len(neigh)//2 or 1)
            for nid in spam_targets:
                msgs.append((nid, Message("PROPOSE", node.id, vA, rnd)))
                msgs.append((nid, Message("PROPOSE", node.id, vB, rnd)))

        return msgs
