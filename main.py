from CPA import run_cpa_with_adversary

# node1 = Node(1)
# node2 = Node(2)
# node6 = Node(6)
# node7 = Node(7)
# node8 = Node(8)
# node9 = Node(9)
# node10 = Node(10)
# node3 = Node(3)
# node4 = Node(4)
# node5 = Node(5)


# connect_nodes(node1, node2)
# connect_nodes(node2, node3)
# connect_nodes(node3, node4)
# connect_nodes(node4, node5)
# connect_nodes(node5, node6)
# connect_nodes(node6, node7)
# connect_nodes(node7, node8)
# connect_nodes(node8, node9)
# connect_nodes(node9, node10)

# nodes = [node1, node8, node9, node6, node7, node10, node3, node4, node5, node2]

# print("Running CPA...")
# cpa = CPA(nodes, node1, 1, 0)
# cpa.run()


if __name__ == "__main__":
    decided, B = run_cpa_with_adversary(n=10, dealer_id=0, dealer_value=1, t=1)
    print("Byzantine set (t-local):", B)
    for i, (d, v) in sorted(decided.items()):
        print(f"Node {i}: decided={d}, value={v}")
