# CPA-Implementation
An implementation of the Certified Propagation Algorithm for Reliable Broadcast

## Graph Types

The implementation supports various graph types:
- `line`: Line graph
- `complete`: Complete graph
- `complete_multipartite`: Complete multipartite graph (specify subset sizes with `--subset-sizes`)
- `complete_bipartite`: Complete bipartite graph (specify partition sizes with `--subset-sizes`)
- `star`: Star graph
- `hypercube`: Hypercube graph
- `custom`: Custom graph from JSON file (specify path with `--custom-graph`)

### Custom Graphs from JSON

You can provide arbitrary graphs using JSON files with the `--graph custom --custom-graph <path>` options.

Three JSON formats are supported:

**Format 1: Simple node-link (recommended)**
```json
{
  "nodes": [0, 1, 2, 3, 4],
  "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]
}
```

**Format 2: Detailed node-link (NetworkX/D3.js style)**
```json
{
  "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
  "links": [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
}
```

**Format 3: Adjacency list**
```json
{
  "adjacency": {
    "0": [1, 2],
    "1": [0, 2],
    "2": [0, 1, 3],
    "3": [2]
  }
}
```

Example usage:
```bash
--exec plain --graph custom --custom-graph example_graph_simple.json --dealer-id 0 --dealer-value 1 --t 2
```

## Execution Modes

The different executions of CPA currently supported:

### Plain Execution

No signatures / PKIs 
Adversary can corrupt for each node u up to t (constant) nodes in its neighborhood

Algorithm:

1. The dealer D sends its initial value X(D) to all of its neighbors, decides on X(D) and terminates.
2. If a node is a neighbor of the dealer, then upon receiving X(D) from the dealer, decides on X(D), sends it to all of its neighbors and terminates.
3. If a node is not a neighbor of the dealer, then upon receiving t + 1 copies of a value x from t + 1 distinct neighbors, it decides on x, sends it to all of its neighbors and terminates.


### Plain Execution with Per-Node Corruption Function t(u)

No signatures / PKIs
Adversary can corrupt for each node u up to t(u) nodes in its neighborhood

Algorithm:

1. The dealer D sends its initial value X(D) to all of its neighbors, decides on X(D) and terminates.
2. If a node is a neighbor of the dealer, then upon receiving X(D) from the dealer, decides on X(D), sends it to all of its neighbors and terminates.
3. If a node u is not a neighbor of the dealer, then upon receiving t(u) + 1 copies of a value x from t(u) + 1 distinct neighbors, it decides on x, sends it to all of its neighbors and terminates.

### Execution with Signatures

Algorithm:

1. The dealer D signs an initial value X, sends Y = sign(X(D)) to all of its neighbors, decides on X(D) and terminates.
2. A node upon receiving a value Y from a neighbor checks whether it is a valid signature from the Dealer's public key, and if so decides on Y, sends it to all of its neighbors and terminates.

### DS-CPA (Dolev-Strong with CPA)

DS-CPA combines the Dolev-Strong broadcast protocol with CPA. Instead of using direct broadcast, each message relay uses a CPA execution with signatures (σ-CPA).

**Key Properties:**
- Runs for f̂ + 1 rounds where f̂ = n - 2 (accounts for unknown global fault bound)
- Each message carries a signature chain (all nodes that have signed it)
- Nodes relay messages only if they have valid signature chains
- More robust than plain CPA under high Byzantine presence

**Algorithm:**

1. **Round 0:** Sender creates signature for value b, sends ⟨b, sig(b)⟩ to all neighbors via σ-CPA
2. **Rounds 1 to f̂ + 1:** For each message b̃ with r distinct signatures (including sender):
   - If b̃ ∉ V (extracted set): add b̃ to V, sign it, relay with r+1 signatures via σ-CPA
3. **Decision:** At end of round f̂ + 1:
   - If |V| = 1: output the value in V
   - Otherwise: output 0 (default value)

**Usage:**
```bash
--exec ds_cpa --graph complete --n 6 --dealer-id 0 --dealer-value 1 --t 1 --seed 42
```

