## Overview

The implementation supports loading arbitrary graphs from JSON files. This allows you to test the algorithm on any network topology you can define.

## Usage

To use a custom graph, specify `--graph custom` and provide the path to your JSON file with `--custom-graph`:

```bash
--graph custom --custom-graph my_graph.json --dealer-id 0 --dealer-value 1 --t 2
```

Complete example in the REPL:
```bash
cpa> --exec plain --graph custom --custom-graph example_graph_simple.json --dealer-id 0 --dealer-value 1 --t 2
```

## Supported JSON Formats

The implementation supports three common JSON formats for graph representation:

### Format 1: Simple Node-Link Format

This is the simplest and most straightforward format. It explicitly lists nodes and edges.

```json
{
  "nodes": [0, 1, 2, 3, 4, 5],
  "edges": [
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 4],
    [3, 5],
    [4, 5]
  ]
}
```

**Key features:**
- `nodes`: Array of node IDs (integers)
- `edges`: Array of edge pairs `[source, target]`
- Edges are undirected (the graph is automatically treated as undirected)

### Format 2: Detailed Node-Link Format

This format uses objects for nodes and edges, similar to NetworkX's `node_link_data()` format and D3.js graph format.

```json
{
  "nodes": [
    {"id": 0},
    {"id": 1},
    {"id": 2},
    {"id": 3},
    {"id": 4}
  ],
  "links": [
    {"source": 0, "target": 1},
    {"source": 0, "target": 2},
    {"source": 1, "target": 3},
    {"source": 2, "target": 3},
    {"source": 3, "target": 4}
  ]
}
```

**Key features:**
- `nodes`: Array of objects with `id` field
- `edges`: Array of objects with `source` and `target` fields
- Can also use `from` and `to` instead of `source` and `target`

### Format 3: Adjacency List Format

This format directly represents the graph as an adjacency list.

```json
{
  "adjacency": {
    "0": [1, 2, 3],
    "1": [0, 2],
    "2": [0, 1, 3],
    "3": [0, 2, 4],
    "4": [3]
  }
}
```

**Key features:**
- `adjacency`: Object mapping node IDs (as strings) to arrays of neighbor IDs
- Each node lists its direct neighbors
- Edges should be bidirectional for undirected graphs (if A lists B, then B should list A)

## Important Notes

1. **Node IDs**: All node IDs will be converted to sequential integers starting from 0. Make sure your `--dealer-id` corresponds to a valid node in your graph after conversion.

2. **Undirected Graphs**: The implementation treats all graphs as undirected. If you specify an edge from A to B, it automatically implies an edge from B to A.

3. **Validation**: The JSON file must be valid JSON and match one of the three formats above. Invalid formats will result in an error message.

4. **File Paths**: You can use relative paths (from the current directory) or absolute paths for the JSON file.

## Example Graphs

Three example graph files are provided in the repository:

- `example_graph_simple.json` - Simple node-link format (6 nodes, 8 edges)
- `example_graph_detailed.json` - Detailed node-link format (5 nodes, 5 edges)
- `example_graph_adjacency.json` - Adjacency list format (5 nodes)

## Testing Your Custom Graph

A test script is provided to verify your JSON file is valid and correctly formatted:

```bash
python3 test_custom_graph.py
```

This will load all example graphs and display their structure.

To test a specific graph file:

```python
from graphs import build_custom_graph_from_json

nodes = build_custom_graph_from_json('my_graph.json')
print(f"Loaded {len(nodes)} nodes")
for node_id, node in nodes.items():
    print(f"Node {node_id}: neighbors = {node.neighbors}")
```

## Integration with Different Execution Modes

Custom graphs work with all execution modes:

### Plain CPA
```bash
--exec plain --graph custom --custom-graph my_graph.json --dealer-id 0 --dealer-value 1 --t 3
```

### CPA with Dealer Signatures
```bash
--exec signed --graph custom --custom-graph my_graph.json --dealer-id 0 --dealer-value 1 --t 3
```

### CPA with Per-Node Thresholds
```bash
--exec per_node_t --graph custom --custom-graph my_graph.json --dealer-id 0 --t-func 2 --seed 42
```

### CPA with Both Signatures and Per-Node Thresholds
```bash
--exec signed_per_node_t --graph custom --custom-graph my_graph.json --dealer-id 0 --t-func 3 --seed 42
```

## Creating Custom Graphs

You can create custom graphs:

1. **Manually**: Write a JSON file following one of the formats above
2. **From NetworkX**: If you're using Python and NetworkX:
   ```python
   import networkx as nx
   import json
   
   # Create your graph
   G = nx.karate_club_graph()
   
   # Export as node-link format
   data = {
       "nodes": list(G.nodes()),
       "edges": list(G.edges())
   }
   
   with open('my_graph.json', 'w') as f:
       json.dump(data, f, indent=2)
   ```

3. **From other tools**: Many graph visualization and analysis tools can export to JSON formats compatible with this implementation.

## Error Handling

If there's an error loading your custom graph, you'll see an error message like:

```
ValueError: --custom-graph path must be provided when using --graph custom
```
or
```
ValueError: Invalid JSON format. Expected either 'adjacency' or 'nodes' key.
```

Check that:
- The JSON file exists and is readable
- The JSON is valid (use a JSON validator)
- The format matches one of the three supported formats
- All node IDs are valid integers or can be converted to integers

