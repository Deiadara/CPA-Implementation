import argparse
from CPA import (
    run_cpa_with_adversary,
    run_cpa_with_dealer_signature,
    run_cpa_with_per_node_threshold,
    run_cpa_with_dealer_signature_and_per_node_threshold,
    run_ds_cpa,
    run_bcpa,
    predict_cpa_outcome_for_constant_t,
    predict_ds_cpa_outcome,
    predict_bcpa_outcome,
    evaluate_execution,
)


def parse_args_once():
    parser = argparse.ArgumentParser(description="Run CPA or CPA-with-signatures.")
    parser.add_argument("--exec", choices=["plain", "signed", "per_node_t", "signed_per_node_t", "ds_cpa", "bcpa"], default="plain", help="Execution type")
    parser.add_argument("--graph", choices=["line", "complete", "complete_multipartite", "complete_bipartite", "star", "hypercube", "custom"], default="complete_multipartite", help="Graph type")
    parser.add_argument("--n", type=int, default=10, help="Number of nodes")
    parser.add_argument("--dealer-id", type=int, default=0, help="Dealer node id")
    parser.add_argument("--dealer-value", type=int, default=1, help="Dealer value")
    parser.add_argument("--t", type=int, default=3, help="t for t-local faults (sampling); ignored when --exec per_node_t")
    parser.add_argument("--f", type=int, default=None, help="Maximum global Byzantine nodes for B-CPA (default: n//3 - 1)")
    parser.add_argument("--t-func", type=int, choices=[1,2,3,4,5,6], default=1, help="Per-node t(u) function when --exec per_node_t: 1) t(u)=1; 2) t(u)=u; 3) t(u)=u^2; 4) t(u)=u%%2; 5) t(u)=u%%5; 6) t(u)=rand(0,n)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for fault sampling")
    parser.add_argument("--subset-sizes", type=str, default="3,3,3", help="Subset sizes for complete_multipartite or complete_bipartite (e.g. 3,3,3 or 4,6)")
    parser.add_argument("--custom-graph", type=str, default=None, help="Path to JSON file for custom graph (required when --graph custom)")
    parser.add_argument("--dealer-byzantine", action="store_true", help="Make dealer Byzantine (for B-CPA)")
    return parser


def run_once(args):
    subset_sizes = None
    if args.graph in {"complete_multipartite", "complete_bipartite"}:
        try:
            subset_sizes = tuple(int(x.strip()) for x in args.subset_sizes.split(",") if x.strip())
        except Exception:
            subset_sizes = (3, 3, 3) if args.graph == "complete_multipartite" else (args.n // 2, args.n - (args.n // 2))

    common_kwargs = dict(
        n=args.n,
        dealer_id=args.dealer_id,
        dealer_value=args.dealer_value,
        t=args.t,
        seed=args.seed,
        graph=args.graph,
        subset_sizes=subset_sizes,
        custom_graph_path=args.custom_graph,
    )

    if args.exec == "plain":
        decided, B = run_cpa_with_adversary(**common_kwargs)
    elif args.exec == "signed":
        decided, B = run_cpa_with_dealer_signature(**common_kwargs)
    elif args.exec == "per_node_t":
        decided, B = run_cpa_with_per_node_threshold(
            n=args.n,
            dealer_id=args.dealer_id,
            dealer_value=args.dealer_value,
            t_func_id=args.t_func,
            seed=args.seed,
            graph=args.graph,
            subset_sizes=subset_sizes,
            custom_graph_path=args.custom_graph,
        )
    elif args.exec == "ds_cpa":
        decided, B = run_ds_cpa(
            n=args.n,
            sender_id=args.dealer_id,
            sender_value=args.dealer_value,
            t=args.t,
            seed=args.seed,
            graph=args.graph,
            subset_sizes=subset_sizes,
            custom_graph_path=args.custom_graph,
        )
    elif args.exec == "bcpa":
        # B-CPA: f defaults to (n-1)//3 to satisfy n >= 3f+1
        f_val = args.f if args.f is not None else (args.n - 1) // 3
        decided, B = run_bcpa(
            n=args.n,
            dealer_id=args.dealer_id,
            dealer_value=args.dealer_value,
            f=f_val,
            t=args.t,
            seed=args.seed,
            graph=args.graph,
            subset_sizes=subset_sizes,
            custom_graph_path=args.custom_graph,
            dealer_is_byzantine=args.dealer_byzantine,
        )
    else:
        decided, B = run_cpa_with_dealer_signature_and_per_node_threshold(
            n=args.n,
            dealer_id=args.dealer_id,
            dealer_value=args.dealer_value,
            t_func_id=args.t_func,
            seed=args.seed,
            graph=args.graph,
            subset_sizes=subset_sizes,
            custom_graph_path=args.custom_graph,
        )

    print("Byzantine set (t-local):", B)
    for i, (d, v) in sorted(decided.items()):
        print(f"Node {i}: decided={d}, value={v}")

    # Evaluate actual execution success (honest nodes decided on dealer value)
    success, bad_nodes = evaluate_execution(decided, B, args.dealer_value)
    if success:
        print("Execution result: SUCCESS (all honest nodes decided on dealer's value)")
    else:
        print(f"Execution result: FAILURE (honest nodes not agreeing): {sorted(bad_nodes)}")

    # Predict outcome for plain CPA based on K only in constant-t case
    if args.exec == "plain":
        from CPA import _build_graph  # reuse to ensure same adjacency
        nodes_tmp = _build_graph(args.graph, args.n, args.dealer_id, subset_sizes, args.custom_graph)
        adj_tmp = {i: set(nodes_tmp[i].neighbors) for i in nodes_tmp}
        K_val, verdict = predict_cpa_outcome_for_constant_t(adj_tmp, args.dealer_id, args.t)
        print(f"K(G,D)={K_val}; predicted plain CPA outcome at t={args.t}: {verdict}")
    
    # Predict outcome for DS-CPA based on t-local cuts
    if args.exec == "ds_cpa":
        from CPA import _build_graph  # reuse to ensure same adjacency
        nodes_tmp = _build_graph(args.graph, args.n, args.dealer_id, subset_sizes, args.custom_graph)
        adj_tmp = {i: set(nodes_tmp[i].neighbors) for i in nodes_tmp}
        has_cut, verdict = predict_ds_cpa_outcome(adj_tmp, args.dealer_id, args.t)
        if has_cut:
            print(f"t-local cut exists (excluding sender {args.dealer_id}); predicted DS-CPA outcome at t={args.t}: {verdict}")
        else:
            print(f"No t-local cut (sender {args.dealer_id} always honest); predicted DS-CPA outcome at t={args.t}: {verdict}")
        
        # Compare prediction with actual result
        actual_result = "succeeds" if success else "fails"
        if verdict == actual_result:
            print(f"✓ Prediction matches actual result: {actual_result}")
        else:
            print(f"✗ Prediction mismatch: predicted {verdict}, actual {actual_result}")
    
    # Predict outcome for B-CPA
    if args.exec == "bcpa":
        from CPA import _build_graph
        f_val = args.f if args.f is not None else (args.n - 1) // 3
        nodes_tmp = _build_graph(args.graph, args.n, args.dealer_id, subset_sizes, args.custom_graph)
        adj_tmp = {i: set(nodes_tmp[i].neighbors) for i in nodes_tmp}
        valid, verdict = predict_bcpa_outcome(adj_tmp, args.n, f_val)
        print(f"B-CPA prediction (n={args.n}, f={f_val}): {verdict}")
        
        actual_result = "succeeds" if success else "fails"
        if ("succeeds" in verdict) == success:
            print(f"✓ Prediction matches actual result: {actual_result}")
        else:
            print(f"✗ Prediction mismatch: predicted {verdict}, actual {actual_result}")


if __name__ == "__main__":
    parser = parse_args_once()
    import shlex

    def print_help():
        print("\nGeneral usage:")
        print(parser.format_help())
        print("Graph-specific notes:")
        print("- line: uses --n, --dealer-id, --dealer-value, --t, --seed")
        print("- complete: uses --n, --dealer-id, --dealer-value, --t, --seed")
        print("- complete_multipartite: additionally uses --subset-sizes (e.g. 4,3,3)")
        print("- complete_bipartite: uses --subset-sizes as a,b (e.g. 4,6)")
        print("- star: uses --n (creates n-1 leaves)")
        print("- hypercube: uses --n; builds 2^d nodes with 2^d <= n (rounds down)")
        print("- custom: uses --custom-graph to specify path to JSON file\n")
        print("Execution modes:")
        print("- plain: classic CPA, decides at t+1")
        print("- signed: σ-CPA with dealer signature, no threshold; accept only dealer-signed value")
        print("- per_node_t: CPA with per-node threshold t(u); decides at t(u)+1")
        print("- signed_per_node_t: both dealer signatures and per-node t(u) thresholds")
        print("- ds_cpa: Dolev-Strong combined with CPA (runs for (n-1)*n rounds)")
        print("- bcpa: Bracha's CPA for dishonest dealer (requires n >= 3f+1)\n")
        print("t(u) functions (for per_node_t):")
        print("  1) t(u) = 1")
        print("  2) t(u) = u  (u = 1-based node index)")
        print("  3) t(u) = u^2")
        print("  4) t(u) = u % 2")
        print("  5) t(u) = u % 5")
        print("  6) t(u) = rand(0, n)\n")
        print("Examples:")
        print("  --exec plain  --graph complete --n 10 --dealer-id 0 --dealer-value 1 --t 3")
        print("  --exec signed --graph complete_multipartite --subset-sizes 4,3,3 --n 10 --dealer-id 0 --dealer-value 1 --t 3")
        print("  --exec per_node_t --t-func 4 --graph line --n 8 --seed 42")
        print("  --exec signed_per_node_t --t-func 5 --graph complete --n 10 --seed 7")
        print("  --exec signed --graph line --n 8 --seed 42")
        print("  --exec plain --graph custom --custom-graph my_graph.json --dealer-id 0 --dealer-value 1 --t 2")
        print("  --exec ds_cpa --graph complete --n 6 --dealer-id 0 --dealer-value 1 --t 1 --seed 42")
        print("  --exec bcpa --graph complete --n 10 --f 3 --dealer-value 1 --seed 42")
        print("  --exec bcpa --graph complete --n 10 --f 2 --dealer-byzantine --seed 42\n")
        print("Other commands:")
        print("  help  | ?   Show this help")
        print("  exit  | quit  Exit the REPL\n")

    while True:
        try:
            line = input("cpa> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in {"help", "?", "-h", "--help"}:
            print_help()
            continue
        if line in {"quit", "exit"}:
            break
        try:
            if line in {"-h", "--help"}:
                print_help()
                continue
            args = parser.parse_args(shlex.split(line))
            run_once(args)
        except SystemExit:
            continue
