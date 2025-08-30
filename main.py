import argparse
from CPA import (
    run_cpa_with_adversary,
    run_cpa_with_dealer_signature,
    run_cpa_with_per_node_threshold,
    run_cpa_with_dealer_signature_and_per_node_threshold,
    predict_cpa_outcome_for_constant_t,
    evaluate_execution,
)


def parse_args_once():
    parser = argparse.ArgumentParser(description="Run CPA or CPA-with-signatures.")
    parser.add_argument("--exec", choices=["plain", "signed", "per_node_t", "signed_per_node_t"], default="plain", help="Execution type")
    parser.add_argument("--graph", choices=["line", "complete", "complete_multipartite", "complete_bipartite", "star", "hypercube"], default="complete_multipartite", help="Graph type")
    parser.add_argument("--n", type=int, default=10, help="Number of nodes")
    parser.add_argument("--dealer-id", type=int, default=0, help="Dealer node id")
    parser.add_argument("--dealer-value", type=int, default=1, help="Dealer value")
    parser.add_argument("--t", type=int, default=3, help="t for t-local faults (sampling); ignored when --exec per_node_t")
    parser.add_argument("--t-func", type=int, choices=[1,2,3,4,5,6], default=1, help="Per-node t(u) function when --exec per_node_t: 1) t(u)=1; 2) t(u)=u; 3) t(u)=u^2; 4) t(u)=u%%2; 5) t(u)=u%%5; 6) t(u)=rand(0,n)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for fault sampling")
    parser.add_argument("--subset-sizes", type=str, default="3,3,3", help="Subset sizes for complete_multipartite or complete_bipartite (e.g. 3,3,3 or 4,6)")
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
        nodes_tmp = _build_graph(args.graph, args.n, args.dealer_id, subset_sizes)
        adj_tmp = {i: set(nodes_tmp[i].neighbors) for i in nodes_tmp}
        K_val, verdict = predict_cpa_outcome_for_constant_t(adj_tmp, args.dealer_id, args.t)
        print(f"K(G,D)={K_val}; predicted plain CPA outcome at t={args.t}: {verdict}")


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
        print("- hypercube: uses --n; builds 2^d nodes with 2^d <= n (rounds down)\n")
        print("Execution modes:")
        print("- plain: classic CPA, decides at t+1")
        print("- signed: CPA with dealer signature, no threshold; accept only dealer-signed value")
        print("- per_node_t: CPA with per-node threshold t(u); decides at t(u)+1")
        print("- signed_per_node_t: both dealer signatures and per-node t(u) thresholds\n")
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
        print("  --exec signed --graph line --n 8 --seed 42\n")
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
