from node import Node
ROUNDS = 100

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