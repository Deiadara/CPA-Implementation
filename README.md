# CPA-Implementation
An implementation of the Certified Propagation Algorithm for Reliable Broadcast

Two different executions of CPA are currently supported, with different graph types:

--Plain execution--

No signatures / PKIs 

Algorithm :

1. The dealer D sends its initial value X(D) to all of its neighbors, decides on X(D) and terminates.
2. If a node is a neighbor of the dealer, then upon receiving X(D) from the dealer, decides on X(D), sends it to all of its neighbors and terminates.
3. If a node is not a neighbor of the dealer, then upon receiving t + 1 copies of a value x from t + 1 distinct neighbors, it decides on x, sends it to all of its neighbors and terminates.

--Execution with Signatures--

Algorithm :

1. The dealer D signs an initial value X, sends Y = sign(X(D)) to all of its neighbors, decides on X(D) and terminates.

2. A node upon receiving a value Y from a neighbor checks whether it is a valid signature from the Dealer's public key, and if so decides on Y, sends it to all of its neighbors and terminates.
