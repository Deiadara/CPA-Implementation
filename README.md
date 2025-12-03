# CPA-Implementation

An implementation of the Certified Propagation Algorithm (CPA) family for Byzantine Reliable Broadcast.

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Run CPA demo for all protocols
python demo.py all

# Run specific protocol demo
python demo.py cpa
python demo.py sigma
python demo.py ds
python demo.py bcpa
python demo.py bcpa-dishonest
```

## Protocols

The implementation includes four Byzantine broadcast protocols:

| Protocol | Model | Dealer Honesty | Rounds | Key Feature |
|----------|-------|----------------|--------|-------------|
| CPA | Synchronous | Honest only | n | Threshold-based acceptance |
| σ-CPA | Synchronous | Honest only | n | Dealer signatures |
| DS-CPA | Synchronous | Tolerates dishonest | (f+1)×n | Signature chains |
| B-CPA | Asynchronous | Tolerates dishonest | N/A | Quorum intersection |

## Graph Types

The implementation supports various graph topologies:

| Graph | Command | Description |
|-------|---------|-------------|
| Complete | `--graph complete` | Every node connected to every other |
| Line | `--graph line` | Sequential chain of nodes |
| Cycle | `--graph cycle` | Ring topology |
| Star | `--graph star` | Central hub connected to all |
| Hypercube | `--graph hypercube` | d-dimensional hypercube (2^d nodes) |
| Dense Random | `--graph dense_random` | Erdős-Rényi G(n, 0.6) |
| Sparse Random | `--graph sparse_random` | Erdős-Rényi G(n, 0.2) |
| Random Regular | `--graph random_regular` | Each node has degree 3 |
| Grid | `--graph grid` | 2D lattice structure |
| Multipartite | `--graph complete_multipartite` | Complete k-partite graph |
| Bipartite | `--graph complete_bipartite` | Complete bipartite graph |
| Custom | `--graph custom` | Load from JSON file |

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

## Execution Modes

### CPA (Plain)

Basic CPA without signatures. Adversary can corrupt up to t nodes in each node's neighborhood.

```bash
python main.py --exec plain --graph complete --n 10 --dealer-id 0 --dealer-value 1 --t 2
```

**Algorithm:**
1. Dealer D sends value X(D) to all neighbors, decides on X(D)
2. Neighbors of dealer decide on X(D) upon receiving it directly
3. Other nodes decide on value x upon receiving t+1 copies from distinct neighbors

### σ-CPA (Signed)

CPA with dealer signatures. Byzantine nodes cannot forge the dealer's signature.

```bash
python main.py --exec signed --graph complete --n 10 --dealer-id 0 --dealer-value 1 --t 2
```

**Algorithm:**
1. Dealer signs value X(D), sends signature to all neighbors
2. Nodes verify signature and decide upon receiving valid signature
3. Nodes relay the signed value to their neighbors

### CPA with Per-Node Threshold t(u)

Each node u has its own corruption threshold t(u).

```bash
python main.py --exec per_node_t --graph complete --n 10 --t-func 2
```

**Available t(u) functions:**
- `--t-func 1`: t(u) = 1 (constant)
- `--t-func 2`: t(u) = u (linear in node index)
- `--t-func 3`: t(u) = u² (quadratic)
- `--t-func 4`: t(u) = u mod 2 (alternating)
- `--t-func 5`: t(u) = u mod 5
- `--t-func 6`: t(u) = random(0, n)

### DS-CPA (Dolev-Strong with CPA)

Combines Dolev-Strong broadcast with CPA. Tolerates dishonest dealer.

```bash
python main.py --exec ds_cpa --graph complete --n 6 --dealer-id 0 --dealer-value 1 --t 1
```

**Key Properties:**
- Runs for f̂ + 1 rounds where f̂ = n - 2
- Each message carries a signature chain
- Guarantees termination even with dishonest dealer

**Algorithm:**
1. **Round 0:** Sender signs value, broadcasts via σ-CPA
2. **Rounds 1 to f̂+1:** Relay messages with accumulated signatures
3. **Decision:** Output single value if |V| = 1, else output 0

### B-CPA (Bracha's CPA)

Asynchronous reliable broadcast using Bracha's protocol with CPA for message propagation.

```bash
# Honest dealer
python main.py --exec bcpa --graph complete --n 10 --dealer-id 0 --dealer-value 1 --f 3 --t 1

# Dishonest dealer
python main.py --exec bcpa --graph complete --n 10 --dealer-id 0 --dealer-value 1 --f 3 --t 1 --dealer-byzantine
```

**Key Properties:**
- Requires n ≥ 3f + 1 (quorum intersection)
- Asynchronous: no round structure
- Agreement guaranteed even with dishonest dealer
- Termination NOT guaranteed with dishonest dealer (safe behavior)

**Algorithm:**
1. **PROPOSE:** Dealer broadcasts value via CPA
2. **ECHO:** Upon receiving dealer's value, broadcast ECHO
3. **VOTE:** Upon receiving n-f ECHOs, broadcast VOTE
4. **Amplify:** Upon receiving f+1 VOTEs, broadcast VOTE
5. **DELIVER:** Upon receiving n-f VOTEs, deliver value

**Parameters:**
- `--f`: Maximum global Byzantine nodes (default: (n-1)//3)
- `--t`: t-local corruption bound for CPA propagation
- `--dealer-byzantine`: Make dealer Byzantine (for testing)

## Demo Script

The `demo.py` script provides clean output suitable for screenshots and documentation.

```bash
# Run all protocol demos
python demo.py all

# Run individual demos
python demo.py cpa           # CPA protocol
python demo.py sigma         # σ-CPA protocol
python demo.py ds            # DS-CPA protocol
python demo.py bcpa          # B-CPA with honest dealer
python demo.py bcpa-dishonest  # B-CPA with dishonest dealer
```

Each demo runs on a dense random graph with a randomly sampled t-local Byzantine set.

## Benchmarking

The `benchmark.py` script runs comprehensive tests across multiple graph types and corruption scenarios.

```bash
python benchmark.py
```

This produces:
- Specific scenario comparisons
- Statistical results over 100+ samples per graph type
- Head-to-head CPA vs σ-CPA comparison
- LaTeX table output for papers

## Command Line Reference

```
usage: main.py [options]

options:
  --exec {plain,signed,per_node_t,signed_per_node_t,ds_cpa,bcpa}
                        Execution type (default: plain)
  --graph {line,complete,complete_multipartite,complete_bipartite,star,
           hypercube,cycle,dense_random,sparse_random,random_regular,
           grid,custom}
                        Graph type (default: complete_multipartite)
  --n N                 Number of nodes (default: 10)
  --dealer-id ID        Dealer node id (default: 0)
  --dealer-value VALUE  Dealer value (default: 1)
  --t T                 t-local corruption bound (default: 3)
  --f F                 Global Byzantine bound for B-CPA (default: (n-1)//3)
  --t-func {1,2,3,4,5,6}
                        Per-node t(u) function (default: 1)
  --seed SEED           Random seed for fault sampling
  --subset-sizes SIZES  Subset sizes for multipartite/bipartite (e.g., "3,3,3")
  --custom-graph PATH   Path to JSON file for custom graph
  --dealer-byzantine    Make dealer Byzantine (B-CPA only)
```

## Project Structure

```
CPA-Implementation/
├── main.py           # Command-line interface
├── CPA.py            # Protocol implementations
├── network.py        # Network simulation
├── node.py           # Node and behavior classes
├── adversary.py      # Byzantine adversary behaviors
├── graphs.py         # Graph construction utilities
├── utils.py          # Helper functions (t-local sampling)
├── demo.py           # Demo script for screenshots
├── benchmark.py      # Comprehensive benchmarking
├── test_protocols.py # Unit tests
└── custom_graphs/    # Example JSON graph files
```

## References

- CPA protocol: "Certified Propagation Algorithm" (CPAjournal.pdf)
- Dolev-Strong: Original Byzantine broadcast protocol
- Bracha's Broadcast: Asynchronous reliable broadcast
