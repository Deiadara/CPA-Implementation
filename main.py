from CPA import run_cpa_with_adversary


if __name__ == "__main__":
    decided, B = run_cpa_with_adversary(n=10, dealer_id=0, dealer_value=1, t=3)
    print("Byzantine set (t-local):", B)
    for i, (d, v) in sorted(decided.items()):
        print(f"Node {i}: decided={d}, value={v}")

    
