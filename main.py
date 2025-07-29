from node import Node, connect_nodes
from CPA import CPA

node1 = Node()
node2 = Node()
node6 = Node()
node7 = Node()
node8 = Node()
node9 = Node()
node10 = Node()
node3 = Node()
node4 = Node()
node5 = Node()


connect_nodes(node1, node2)
connect_nodes(node2, node3)
connect_nodes(node3, node4)
connect_nodes(node4, node5)
connect_nodes(node5, node6)
connect_nodes(node6, node7)
connect_nodes(node7, node8)
connect_nodes(node8, node9)
connect_nodes(node9, node10)

nodes = [node1, node8, node9, node6, node7, node10, node3, node4, node5, node2]

print("Running CPA...")
cpa = CPA(nodes, node1, 1, 0)
cpa.run()
