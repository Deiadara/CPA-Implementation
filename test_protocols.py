"""
Tests for CPA, σ-CPA, DS-CPA, and B-CPA protocols.

These tests verify:
1. Correct round counts (CPA/σ-CPA: n rounds, DS-CPA: (f̂+1)*n rounds)
2. Same resilience for σ-CPA and DS-CPA (both succeed or both fail for same corruption set)
3. Protocol correctness under various graph topologies
4. CPA and σ-CPA resilience based on K(G,D) from papers
5. B-CPA (Bracha's CPA) correctness with honest and dishonest dealers
6. DS-CPA and B-CPA behavior with dishonest dealers
"""

import unittest
import io
import sys
from CPA import (
    run_cpa_with_adversary,
    run_cpa_with_dealer_signature,
    run_ds_cpa,
    run_bcpa,
    evaluate_execution,
    predict_cpa_outcome_for_constant_t,
    predict_cpa_outcome_for_constant_t_and_signatures,
    predict_ds_cpa_outcome,
    predict_bcpa_outcome,
    compute_K,
    _build_graph,
    _has_t_local_cut_excluding_dealer,
)
from utils import sample_t_local_faulty_set


class TestRoundCounts(unittest.TestCase):
    """Test that protocols run for the correct number of rounds."""
    
    def test_cpa_runs_n_rounds(self):
        """CPA should run for n rounds."""
        n = 5
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_cpa_with_adversary(n=n, graph="line", dealer_id=0, t=0, seed=42)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        # Count the number of rounds printed
        round_count = output.count("--- Round ")
        self.assertEqual(round_count, n, f"Expected {n} rounds, got {round_count}")
    
    def test_sigma_cpa_runs_n_rounds(self):
        """σ-CPA should run for n rounds."""
        n = 5
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_cpa_with_dealer_signature(n=n, graph="line", dealer_id=0, t=0, seed=42)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        # Count the number of rounds printed
        round_count = output.count("--- Round ")
        self.assertEqual(round_count, n, f"Expected {n} rounds, got {round_count}")
    
    def test_ds_cpa_runs_fhat_plus_1_times_n_rounds(self):
        """DS-CPA should run for (f̂+1) * n message-passing rounds."""
        n = 4
        f_hat = n - 2  # f̂ = n - 2 = 2
        expected_total_rounds = (f_hat + 1) * n  # (2+1) * 4 = 12
        
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_ds_cpa(n=n, graph="complete", sender_id=0, t=0, seed=42)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        # Count the number of message rounds printed
        round_count = output.count("--- Message Round ")
        self.assertEqual(round_count, expected_total_rounds,
            f"Expected {expected_total_rounds} message rounds, got {round_count}")


class TestProtocolResilience(unittest.TestCase):
    """Test that σ-CPA and DS-CPA have the same resilience."""
    
    def _run_both_protocols(self, n, graph, dealer_id, t, seed, subset_sizes=None):
        """Run both σ-CPA and DS-CPA with same parameters and return results."""
        # Suppress stdout during protocol runs
        captured = io.StringIO()
        sys.stdout = captured
        try:
            # Run σ-CPA
            sigma_decided, sigma_B = run_cpa_with_dealer_signature(
                n=n, graph=graph, dealer_id=dealer_id, dealer_value=1, 
                t=t, seed=seed, subset_sizes=subset_sizes
            )
            sigma_success, _ = evaluate_execution(sigma_decided, sigma_B, 1)
            
            # Run DS-CPA with same seed (same Byzantine set)
            ds_decided, ds_B = run_ds_cpa(
                n=n, graph=graph, sender_id=dealer_id, sender_value=1,
                t=t, seed=seed, subset_sizes=subset_sizes
            )
            ds_success, _ = evaluate_execution(ds_decided, ds_B, 1)
        finally:
            sys.stdout = sys.__stdout__
        
        # Build graph to get adjacency
        nodes = _build_graph(graph, n, dealer_id, subset_sizes, None, seed)
        adj = {i: set(nodes[i].neighbors) for i in nodes}
        
        return sigma_success, ds_success, sigma_B, ds_B, adj
    
    def test_same_resilience_complete_graph(self):
        """σ-CPA and DS-CPA should both succeed on complete graph (no t-local cut)."""
        sigma_success, ds_success, sigma_B, ds_B, adj = self._run_both_protocols(
            n=5, graph="complete", dealer_id=0, t=1, seed=42
        )
        
        # Both should have same Byzantine set
        self.assertEqual(sigma_B, ds_B, "Byzantine sets should be identical")
        
        # Check if t-local cut exists
        has_cut = _has_t_local_cut_excluding_dealer(adj, 0, 1)
        
        # Both should have same outcome
        self.assertEqual(sigma_success, ds_success,
            f"σ-CPA {'succeeded' if sigma_success else 'failed'} but DS-CPA {'succeeded' if ds_success else 'failed'}")
        
        # If no t-local cut, both should succeed
        if not has_cut:
            self.assertTrue(sigma_success and ds_success,
                "Both should succeed when no t-local cut exists")
    
    def test_same_resilience_line_graph_small_t(self):
        """σ-CPA and DS-CPA should have same resilience on line graph."""
        sigma_success, ds_success, sigma_B, ds_B, adj = self._run_both_protocols(
            n=5, graph="line", dealer_id=0, t=0, seed=42
        )
        
        # Both should have same Byzantine set
        self.assertEqual(sigma_B, ds_B, "Byzantine sets should be identical")
        
        # Both should have same outcome
        self.assertEqual(sigma_success, ds_success,
            f"σ-CPA {'succeeded' if sigma_success else 'failed'} but DS-CPA {'succeeded' if ds_success else 'failed'}")
    
    def test_same_resilience_star_graph(self):
        """σ-CPA and DS-CPA should have same resilience on star graph."""
        # Dealer at center
        sigma_success, ds_success, sigma_B, ds_B, adj = self._run_both_protocols(
            n=6, graph="star", dealer_id=0, t=1, seed=42
        )
        
        # Both should have same Byzantine set
        self.assertEqual(sigma_B, ds_B, "Byzantine sets should be identical")
        
        # Both should have same outcome
        self.assertEqual(sigma_success, ds_success,
            f"σ-CPA {'succeeded' if sigma_success else 'failed'} but DS-CPA {'succeeded' if ds_success else 'failed'}")
    
    def test_same_resilience_multipartite(self):
        """σ-CPA and DS-CPA should have same resilience on multipartite graph."""
        sigma_success, ds_success, sigma_B, ds_B, adj = self._run_both_protocols(
            n=9, graph="complete_multipartite", dealer_id=0, t=1, seed=42,
            subset_sizes=(3, 3, 3)
        )
        
        # Both should have same Byzantine set
        self.assertEqual(sigma_B, ds_B, "Byzantine sets should be identical")
        
        # Both should have same outcome
        self.assertEqual(sigma_success, ds_success,
            f"σ-CPA {'succeeded' if sigma_success else 'failed'} but DS-CPA {'succeeded' if ds_success else 'failed'}")
    
    def test_multiple_seeds_same_resilience(self):
        """σ-CPA and DS-CPA should have same resilience across multiple seeds."""
        for seed in range(5):  # Reduced to 5 for faster testing
            sigma_success, ds_success, _, _, _ = self._run_both_protocols(
                n=6, graph="complete", dealer_id=0, t=1, seed=seed
            )
            
            self.assertEqual(sigma_success, ds_success,
                f"Seed {seed}: σ-CPA {'succeeded' if sigma_success else 'failed'} " \
                f"but DS-CPA {'succeeded' if ds_success else 'failed'}")


class TestProtocolCorrectness(unittest.TestCase):
    """Test basic protocol correctness."""
    
    def test_sigma_cpa_succeeds_no_byzantine(self):
        """σ-CPA should succeed when there are no Byzantine nodes."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            decided, B = run_cpa_with_dealer_signature(
                n=5, graph="complete", dealer_id=0, dealer_value=42, t=0, seed=None
            )
        finally:
            sys.stdout = sys.__stdout__
        
        # All honest nodes should decide on dealer's value
        for nid, (d, v) in decided.items():
            if nid not in B:
                self.assertTrue(d, f"Node {nid} should have decided")
                self.assertEqual(v, 42, f"Node {nid} should have decided on 42, got {v}")
    
    def test_ds_cpa_succeeds_no_byzantine(self):
        """DS-CPA should succeed when there are no Byzantine nodes."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            decided, B = run_ds_cpa(
                n=5, graph="complete", sender_id=0, sender_value=42, t=0, seed=None
            )
        finally:
            sys.stdout = sys.__stdout__
        
        # All honest nodes should decide on sender's value
        for nid, (d, v) in decided.items():
            if nid not in B:
                self.assertTrue(d, f"Node {nid} should have decided")
                self.assertEqual(v, 42, f"Node {nid} should have decided on 42, got {v}")
    
    def test_sigma_cpa_propagates_through_line(self):
        """σ-CPA should propagate through a line graph."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            decided, B = run_cpa_with_dealer_signature(
                n=5, graph="line", dealer_id=0, dealer_value=7, t=0, seed=42
            )
        finally:
            sys.stdout = sys.__stdout__
        
        success, bad_nodes = evaluate_execution(decided, B, 7)
        self.assertTrue(success, f"All honest nodes should decide on dealer's value, bad nodes: {bad_nodes}")
    
    def test_ds_cpa_propagates_through_line(self):
        """DS-CPA should propagate through a line graph."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            decided, B = run_ds_cpa(
                n=5, graph="line", sender_id=0, sender_value=7, t=0, seed=42
            )
        finally:
            sys.stdout = sys.__stdout__
        
        success, bad_nodes = evaluate_execution(decided, B, 7)
        self.assertTrue(success, f"All honest nodes should decide on sender's value, bad nodes: {bad_nodes}")


class TestPredictionMatchesExecution(unittest.TestCase):
    """Test that predictions match actual execution results."""
    
    def test_sigma_cpa_prediction_matches_execution(self):
        """σ-CPA execution should match prediction based on t-local cuts."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            for seed in range(3):  # Reduced for faster testing
                for t in [0, 1]:
                    nodes = _build_graph("complete", 6, 0, None, None, seed)
                    adj = {i: set(nodes[i].neighbors) for i in nodes}
                    
                    # Get prediction
                    has_cut = _has_t_local_cut_excluding_dealer(adj, 0, t)
                    predicted_success = not has_cut
                    
                    # Run protocol
                    decided, B = run_cpa_with_dealer_signature(
                        n=6, graph="complete", dealer_id=0, dealer_value=1, t=t, seed=seed
                    )
                    actual_success, _ = evaluate_execution(decided, B, 1)
                    
                    # If no t-local cut, should succeed
                    if predicted_success:
                        self.assertTrue(actual_success,
                            f"Predicted success (no t-local cut) but failed (seed={seed}, t={t})")
        finally:
            sys.stdout = sys.__stdout__
    
    def test_ds_cpa_prediction_matches_execution(self):
        """DS-CPA execution should match prediction based on t-local cuts."""
        # Suppress stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            for seed in range(3):  # Reduced for faster testing
                for t in [0, 1]:
                    nodes = _build_graph("complete", 6, 0, None, None, seed)
                    adj = {i: set(nodes[i].neighbors) for i in nodes}
                    
                    # Get prediction
                    has_cut, verdict = predict_ds_cpa_outcome(adj, 0, t)
                    predicted_success = verdict == "succeeds"
                    
                    # Run protocol
                    decided, B = run_ds_cpa(
                        n=6, graph="complete", sender_id=0, sender_value=1, t=t, seed=seed
                    )
                    actual_success, _ = evaluate_execution(decided, B, 1)
                    
                    # If no t-local cut, should succeed
                    if predicted_success:
                        self.assertTrue(actual_success,
                            f"Predicted success but failed (seed={seed}, t={t})")
        finally:
            sys.stdout = sys.__stdout__


class TestCPAResilienceKGD(unittest.TestCase):
    """
    Test CPA and σ-CPA resilience based on K(G,D) from papers.
    
    From Theorem (Sufficient Condition):
    - CPA succeeds if 2t < K(G,D) (for plain CPA)
    - σ-CPA succeeds if t < K(G,D) (for signed CPA)
    
    From Theorem (Necessary Condition):
    - σ-CPA fails if t >= K(G,D)
    """
    
    def _suppress_output(self):
        """Context manager to suppress stdout."""
        self._captured = io.StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._captured
    
    def _restore_output(self):
        sys.stdout = self._old_stdout
    
    def test_sigma_cpa_succeeds_when_t_less_than_K(self):
        """σ-CPA should succeed when t < K(G,D)."""
        self._suppress_output()
        try:
            # Complete graph has high K(G,D)
            nodes = _build_graph("complete", 6, 0, None, None)
            adj = {i: set(nodes[i].neighbors) for i in nodes}
            K = compute_K(adj, 0)
            
            # For t < K, σ-CPA should succeed
            for t in range(0, min(K, 3)):
                decided, B = run_cpa_with_dealer_signature(
                    n=6, graph="complete", dealer_id=0, dealer_value=1, t=t, seed=42
                )
                success, _ = evaluate_execution(decided, B, 1)
                self.assertTrue(success, 
                    f"σ-CPA should succeed when t={t} < K={K}")
        finally:
            self._restore_output()
    
    def test_sigma_cpa_K_determines_max_resilience(self):
        """K(G,D) determines maximum resilience for σ-CPA."""
        self._suppress_output()
        try:
            # Line graph has K(G,D) = 1 (each node has only 1 or 2 neighbors)
            nodes = _build_graph("line", 5, 0, None, None)
            adj = {i: set(nodes[i].neighbors) for i in nodes}
            K = compute_K(adj, 0)
            
            # K should be small for line graph
            self.assertLessEqual(K, 2, f"Line graph K={K} should be small")
            
            # With t=0, should succeed
            decided, B = run_cpa_with_dealer_signature(
                n=5, graph="line", dealer_id=0, dealer_value=1, t=0, seed=42
            )
            success, _ = evaluate_execution(decided, B, 1)
            self.assertTrue(success, "σ-CPA should succeed with t=0 on line graph")
        finally:
            self._restore_output()
    
    def test_complete_graph_high_K(self):
        """Complete graph should have high K(G,D)."""
        nodes = _build_graph("complete", 10, 0, None, None)
        adj = {i: set(nodes[i].neighbors) for i in nodes}
        K = compute_K(adj, 0)
        
        # Complete graph: every node is neighbor to dealer, high connectivity
        self.assertGreater(K, 1, f"Complete graph should have K > 1, got K={K}")
    
    def test_star_graph_low_K(self):
        """Star graph with dealer at center should have specific K(G,D)."""
        nodes = _build_graph("star", 6, 0, None, None)
        adj = {i: set(nodes[i].neighbors) for i in nodes}
        K = compute_K(adj, 0)
        
        # Star with dealer at center: all nodes are direct neighbors
        # but leaves only connect through center
        self.assertGreaterEqual(K, 1, f"Star graph K={K} should be at least 1")


class TestBCPAProtocol(unittest.TestCase):
    """Test B-CPA (Bracha's CPA) protocol."""
    
    def _suppress_output(self):
        self._captured = io.StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._captured
    
    def _restore_output(self):
        sys.stdout = self._old_stdout
    
    def test_bcpa_succeeds_honest_dealer(self):
        """B-CPA should succeed with honest dealer when n >= 3f+1."""
        self._suppress_output()
        try:
            # n=10, f=3 satisfies n >= 3f+1 (10 >= 10)
            decided, B = run_bcpa(
                n=10, dealer_id=0, dealer_value=42, f=3, t=1,
                graph="complete", seed=42, dealer_is_byzantine=False
            )
            
            # All honest nodes should decide on dealer's value
            for nid, (d, v) in decided.items():
                if nid not in B:
                    self.assertTrue(d, f"Node {nid} should have decided")
                    self.assertEqual(v, 42, f"Node {nid} should decide on 42, got {v}")
        finally:
            self._restore_output()
    
    def test_bcpa_requires_3f_plus_1(self):
        """B-CPA prediction should require n >= 3f+1."""
        nodes = _build_graph("complete", 10, 0, None, None)
        adj = {i: set(nodes[i].neighbors) for i in nodes}
        
        # n=10, f=3: 10 >= 10, should succeed
        valid, verdict = predict_bcpa_outcome(adj, 10, 3)
        self.assertTrue(valid)
        self.assertIn("succeeds", verdict)
        
        # n=10, f=4: 10 < 13, should fail
        valid, verdict = predict_bcpa_outcome(adj, 10, 4)
        self.assertFalse(valid)
        self.assertIn("fails", verdict)
    
    def test_bcpa_agreement_with_byzantine_nodes(self):
        """B-CPA should maintain agreement even with Byzantine nodes."""
        self._suppress_output()
        try:
            # n=10, f=2: 10 >= 7, should work
            decided, B = run_bcpa(
                n=10, dealer_id=0, dealer_value=7, f=2, t=1,
                graph="complete", seed=42, dealer_is_byzantine=False
            )
            
            # Collect all values decided by honest nodes
            honest_values = set()
            for nid, (d, v) in decided.items():
                if nid not in B and d:
                    honest_values.add(v)
            
            # All honest nodes that decided should agree on same value
            self.assertLessEqual(len(honest_values), 1,
                f"Honest nodes should agree, but got values: {honest_values}")
        finally:
            self._restore_output()
    
    def test_bcpa_run_time(self):
        """B-CPA should run for 4n rounds."""
        n = 4
        expected_rounds = 4 * n
        
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_bcpa(n=n, dealer_id=0, dealer_value=1, f=1, t=0,
                    graph="complete", seed=42)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured.getvalue()
        round_count = output.count("--- Round ")
        self.assertEqual(round_count, expected_rounds,
            f"Expected {expected_rounds} rounds, got {round_count}")


class TestDishonestDealer(unittest.TestCase):
    """Test protocols with dishonest dealer scenarios."""
    
    def _suppress_output(self):
        self._captured = io.StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._captured
    
    def _restore_output(self):
        sys.stdout = self._old_stdout
    
    def test_bcpa_handles_dishonest_dealer(self):
        """B-CPA should handle dishonest dealer (agreement still holds)."""
        self._suppress_output()
        try:
            # With dishonest dealer, honest nodes should still agree (on some value)
            # They may not deliver if dealer sends conflicting values
            decided, B = run_bcpa(
                n=10, dealer_id=0, dealer_value=1, f=2, t=1,
                graph="complete", seed=42, dealer_is_byzantine=True
            )
            
            # Collect values from honest nodes that decided
            honest_values = set()
            honest_decided_count = 0
            for nid, (d, v) in decided.items():
                if nid not in B:
                    if d:
                        honest_decided_count += 1
                        honest_values.add(v)
            
            # Agreement: if multiple honest nodes decide, they should agree
            if honest_decided_count > 1:
                self.assertLessEqual(len(honest_values), 1,
                    f"Multiple honest nodes decided on different values: {honest_values}")
        finally:
            self._restore_output()
    
    def test_bcpa_totality_with_honest_dealer(self):
        """B-CPA Totality: if dealer is honest, all honest nodes deliver."""
        self._suppress_output()
        try:
            decided, B = run_bcpa(
                n=10, dealer_id=0, dealer_value=99, f=2, t=1,
                graph="complete", seed=42, dealer_is_byzantine=False
            )
            
            # All honest nodes should decide
            for nid, (d, v) in decided.items():
                if nid not in B:
                    self.assertTrue(d, f"Honest node {nid} should have delivered")
                    self.assertEqual(v, 99, f"Honest node {nid} should deliver dealer's value")
        finally:
            self._restore_output()
    
    def test_ds_cpa_with_honest_dealer(self):
        """DS-CPA should succeed when dealer is honest."""
        self._suppress_output()
        try:
            decided, B = run_ds_cpa(
                n=6, sender_id=0, sender_value=55, t=1,
                graph="complete", seed=42
            )
            
            success, bad_nodes = evaluate_execution(decided, B, 55)
            self.assertTrue(success,
                f"DS-CPA should succeed with honest dealer, bad nodes: {bad_nodes}")
        finally:
            self._restore_output()
    
    def test_sigma_cpa_honest_dealer_required(self):
        """σ-CPA assumes honest dealer - test that it works with honest dealer."""
        self._suppress_output()
        try:
            # σ-CPA with honest dealer should succeed
            decided, B = run_cpa_with_dealer_signature(
                n=6, dealer_id=0, dealer_value=77, t=1,
                graph="complete", seed=42
            )
            
            success, bad_nodes = evaluate_execution(decided, B, 77)
            self.assertTrue(success,
                f"σ-CPA should succeed with honest dealer, bad nodes: {bad_nodes}")
        finally:
            self._restore_output()


class TestProtocolComparison(unittest.TestCase):
    """Compare different protocols under same conditions."""
    
    def _suppress_output(self):
        self._captured = io.StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._captured
    
    def _restore_output(self):
        sys.stdout = self._old_stdout
    
    def test_all_protocols_succeed_no_corruption(self):
        """All protocols should succeed with no Byzantine nodes."""
        self._suppress_output()
        try:
            value = 123
            
            # Plain CPA
            decided_cpa, B_cpa = run_cpa_with_adversary(
                n=5, dealer_id=0, dealer_value=value, t=0, graph="complete"
            )
            success_cpa, _ = evaluate_execution(decided_cpa, B_cpa, value)
            
            # σ-CPA
            decided_sigma, B_sigma = run_cpa_with_dealer_signature(
                n=5, dealer_id=0, dealer_value=value, t=0, graph="complete"
            )
            success_sigma, _ = evaluate_execution(decided_sigma, B_sigma, value)
            
            # DS-CPA
            decided_ds, B_ds = run_ds_cpa(
                n=5, sender_id=0, sender_value=value, t=0, graph="complete"
            )
            success_ds, _ = evaluate_execution(decided_ds, B_ds, value)
            
            # B-CPA
            decided_bcpa, B_bcpa = run_bcpa(
                n=5, dealer_id=0, dealer_value=value, f=1, t=0, graph="complete"
            )
            success_bcpa, _ = evaluate_execution(decided_bcpa, B_bcpa, value)
            
            self.assertTrue(success_cpa, "Plain CPA should succeed")
            self.assertTrue(success_sigma, "σ-CPA should succeed")
            self.assertTrue(success_ds, "DS-CPA should succeed")
            self.assertTrue(success_bcpa, "B-CPA should succeed")
        finally:
            self._restore_output()
    
    def test_sigma_cpa_and_ds_cpa_same_resilience_multiple_topologies(self):
        """σ-CPA and DS-CPA should have same resilience across topologies."""
        self._suppress_output()
        try:
            topologies = [
                ("complete", None),
                ("line", None),
                ("star", None),
                ("cycle", None),
            ]
            
            for graph, sizes in topologies:
                decided_sigma, B_sigma = run_cpa_with_dealer_signature(
                    n=6, dealer_id=0, dealer_value=1, t=1, 
                    graph=graph, seed=42, subset_sizes=sizes
                )
                success_sigma, _ = evaluate_execution(decided_sigma, B_sigma, 1)
                
                decided_ds, B_ds = run_ds_cpa(
                    n=6, sender_id=0, sender_value=1, t=1,
                    graph=graph, seed=42, subset_sizes=sizes
                )
                success_ds, _ = evaluate_execution(decided_ds, B_ds, 1)
                
                self.assertEqual(success_sigma, success_ds,
                    f"Resilience mismatch on {graph}: σ-CPA={success_sigma}, DS-CPA={success_ds}")
        finally:
            self._restore_output()


if __name__ == "__main__":
    unittest.main(verbosity=2)

