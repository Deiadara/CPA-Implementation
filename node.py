class Node:
    def __init__(self):
        self.id = id(self)
        self.value = -1
        self.neighbours = {}
        self.potential_values = {}
        self.decided = False

    def add_neighbour(self, node):
        self.neighbours[node.id] = node

    def get_neighbours(self):
        return list(self.neighbours.values())

    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def get_id(self):
        return self.id
    
    def add_potential_value(self, value):
        if value not in self.potential_values:
            self.potential_values[value] = 1
        else:
            self.potential_values[value] += 1
    
    def propose_value(self, value, node):
        node.add_potential_value(value)

    def get_potential_values(self):
        return self.potential_values

    def get_decided(self):
        return self.decided

    def set_decided(self, decided):
        self.decided = decided

def connect_nodes(n1, n2):
    n1.add_neighbour(n2)
    n2.add_neighbour(n1)

            


