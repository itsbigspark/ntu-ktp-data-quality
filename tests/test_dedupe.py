"""Tests for core/dedup_clustering.py — cluster_duplicates and apply_deduplication."""

import unittest

import pandas as pd

from core.dedup_clustering import apply_deduplication, cluster_duplicates


def _pairs(rows):
    """Build a pairs DataFrame from a list of (id_A, id_B) tuples."""
    return pd.DataFrame(rows, columns=["id_A", "id_B"])


class TestClusterDuplicates(unittest.TestCase):

    def test_single_pair_forms_cluster(self):
        pairs = _pairs([(1, 2)])
        clusters = cluster_duplicates(pairs)
        # Both ids must be in the same cluster
        flat = {v for ids in clusters.values() for v in ids}
        self.assertIn(1, flat)
        self.assertIn(2, flat)

    def test_transitive_chain(self):
        # 1-2, 2-3 → single cluster {1,2,3}
        pairs = _pairs([(1, 2), (2, 3)])
        clusters = cluster_duplicates(pairs)
        flat = {v for ids in clusters.values() for v in ids}
        self.assertEqual(flat, {1, 2, 3})
        self.assertEqual(len(clusters), 1)

    def test_two_separate_clusters(self):
        pairs = _pairs([(1, 2), (3, 4)])
        clusters = cluster_duplicates(pairs)
        self.assertEqual(len(clusters), 2)
        all_ids = {v for ids in clusters.values() for v in ids}
        self.assertEqual(all_ids, {1, 2, 3, 4})

    def test_empty_pairs_returns_empty(self):
        pairs = _pairs([])
        clusters = cluster_duplicates(pairs)
        self.assertEqual(clusters, {})

    def test_custom_column_names(self):
        pairs = pd.DataFrame({"a": [10], "b": [20]})
        clusters = cluster_duplicates(pairs, id_col_a="a", id_col_b="b")
        flat = {v for ids in clusters.values() for v in ids}
        self.assertEqual(flat, {10, 20})

    def test_self_reference_handled(self):
        # Shouldn't crash or create spurious clusters
        pairs = _pairs([(1, 1)])
        clusters = cluster_duplicates(pairs)
        self.assertIsInstance(clusters, dict)

    def test_cluster_ids_are_integers(self):
        pairs = _pairs([(1, 2)])
        clusters = cluster_duplicates(pairs)
        for cid in clusters:
            self.assertIsInstance(cid, int)


class TestApplyDeduplication(unittest.TestCase):

    def _base_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Alyce", "Bob", "Bobby"],
            "score": [10, 20, 30, 40],
        })

    def test_keep_master_removes_duplicates(self):
        df = self._base_df()
        clusters = {0: [0, 1]}       # rows 0 and 1 are duplicates
        master_selections = {0: 0}   # keep row 0
        result = apply_deduplication(df, clusters, master_selections)
        # Row 1 (Alyce) should be gone; rows 2 and 3 remain
        names = result["name"].tolist()
        self.assertNotIn("Alyce", names)
        self.assertIn("Alice", names)

    def test_no_master_keeps_all(self):
        df = self._base_df()
        clusters = {0: [0, 1]}
        master_selections = {}  # no master → keep all
        result = apply_deduplication(df, clusters, master_selections)
        self.assertEqual(len(result), len(df))

    def test_non_duplicate_rows_preserved(self):
        df = self._base_df()
        clusters = {0: [0, 1]}
        master_selections = {0: 0}
        result = apply_deduplication(df, clusters, master_selections)
        names = result["name"].tolist()
        self.assertIn("Bob", names)
        self.assertIn("Bobby", names)

    def test_returns_dataframe(self):
        df = self._base_df()
        result = apply_deduplication(df, {}, {})
        self.assertIsInstance(result, pd.DataFrame)

    def test_empty_clusters_returns_full_df(self):
        df = self._base_df()
        result = apply_deduplication(df, {}, {})
        self.assertEqual(len(result), len(df))

    def test_multiple_clusters(self):
        df = self._base_df()
        clusters = {0: [0, 1], 1: [2, 3]}
        master_selections = {0: 0, 1: 2}
        result = apply_deduplication(df, clusters, master_selections)
        # Only 2 records should remain
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
