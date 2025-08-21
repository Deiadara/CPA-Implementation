import argparse
from CPA import run_cpa_with_adversary, run_cpa_with_dealer_signature


def parse_args_once():
    parser = argparse.ArgumentParser(description="Run CPA or CPA-with-signatures.")
    parser.add_argument("--exec", choices=["plain", "signed"], default="plain", help="Execution type")
    parser.add_argument("--graph", choices=["line", "complete", "complete_multipartite"], default="complete_multipartite", help="Graph type")
    parser.add_argument("--n", type=int, default=10, help="Number of nodes")
    parser.add_argument("--dealer-id", type=int, default=0, help="Dealer node id")
    parser.add_argument("--dealer-value", type=int, default=1, help="Dealer value")
    parser.add_argument("--t", type=int, default=3, help="t for t-local faults (sampling)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for fault sampling")
    parser.add_argument("--subset-sizes", type=str, default="3,3,3", help="Subset sizes for complete_multipartite, e.g. 3,3,3")
    return parser


def run_once(args):
    subset_sizes = None
    if args.graph == "complete_multipartite":
        try:
            subset_sizes = tuple(int(x.strip()) for x in args.subset_sizes.split(",") if x.strip())
        except Exception:
            subset_sizes = (3, 3, 3)

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
    else:
        decided, B = run_cpa_with_dealer_signature(**common_kwargs)

    print("Byzantine set (t-local):", B)
    for i, (d, v) in sorted(decided.items()):
        print(f"Node {i}: decided={d}, value={v}")


if __name__ == "__main__":
    parser = parse_args_once()
    import shlex

    def print_help():
        print("\nGeneral usage:")
        print(parser.format_help())
        print("Graph-specific notes:")
        print("- line: uses --n, --dealer-id, --dealer-value, --t, --seed")
        print("- complete: uses --n, --dealer-id, --dealer-value, --t, --seed")
        print("- complete_multipartite: additionally uses --subset-sizes (e.g. 4,3,3)\n")
        print("Examples:")
        print("  --exec plain  --graph complete --n 10 --dealer-id 0 --dealer-value 1 --t 3")
        print("  --exec signed --graph complete_multipartite --subset-sizes 4,3,3 --n 10 --dealer-id 0 --dealer-value 1 --t 3")
        print("  --exec signed --graph line --n 8  --seed 42\n")
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
        if line in {"help", "?"}:
            print_help()
            continue
        if line in {"quit", "exit"}:
            break
        try:
            args = parser.parse_args(shlex.split(line))
            run_once(args)
        except SystemExit:
            continue
