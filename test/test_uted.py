import unittest

import edist.uted as uted


# the purpose of this test is to reproduce results mentioned in Guignard et al. (2020) with the tools used
# for the results in the publication
class TestTreex(unittest.TestCase):
    # simple test case to test the implementation of the Zhang edit distance
    def test_edist_zhang_edit_distance_tree1_tree2(self):
        tree1_nodes = ["a", "b", "c"]
        tree1_adj = [[1, 2], [], []]
        tree1_attributes = {"a": 20, "b": 10, "c": 30}
        tree2_nodes = ["a", "b", "c"]
        tree2_adj = [[1, 2], [], []]
        tree2_attributes = {"a": 30, "b": 10, "c": 20}

        def local_cost(x, y):
            if x is None and y is None:
                return 0
            elif x is None:
                return tree2_attributes[y]
            elif y is None:
                return tree1_attributes[x]
            return abs(tree1_attributes[x] - tree2_attributes[y])

        edist_result = uted.uted(
            tree1_nodes, tree1_adj, tree2_nodes, tree2_adj, local_cost
        )
        self.assertEqual(20, edist_result)

    # Guignard et al. (2020) Fig. S23
    # https://www.science.org/doi/suppl/10.1126/science.aar5663/suppl_file/aar5663_guignard_sm.pdf"
    def test_edist_zhang_edit_distance_tree_guignard_t1_tree_guignard_t2(self):
        t1_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        t1_adj = [[1, 4], [2, 3], [], [], [5, 6], [], [7, 8], [], []]
        t1_attributes = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 9,
            5: 10,
            6: 10,
            7: 10,
            8: 10,
        }
        t2_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        t2_adj = [[1, 6], [2, 5], [3, 4], [], [], [], [7, 8], [], []]
        t2_attributes = {
            0: 1,
            1: 1,
            2: 2,
            3: 1,
            4: 1,
            5: 1,
            6: 10,
            7: 10,
            8: 10,
        }

        def local_cost(t1, t2):
            if t1 is None and t2 is None:
                return 0
            elif t1 is None:
                return t2_attributes[t2]
            elif t2 is None:
                return t1_attributes[t1]
            return abs(t1_attributes[t1] - t2_attributes[t2])

        def local_cost_normalized(t1, t2):
            if t1 is None and t2 is None:
                return 0
            elif t1 is None or t2 is None or t2 is None:
                return 1
            return abs(t1_attributes[t1] - t2_attributes[t2]) / (
                t1_attributes[t1] + t2_attributes[t2]
            )

        self.assertEqual(
            22, uted.uted(t1_nodes, t1_adj, t2_nodes, t2_adj, local_cost)
        )
        # NB: the publication does not illustrate the optimal tree edit distance on purpose,
        # because the primary goal of the figure s23 is to explain all possible edit operations in on figure
        self.assertEqual(
            4 / 90,
            uted.uted(
                t1_nodes, t1_adj, t2_nodes, t2_adj, local_cost_normalized
            )
            / (sum(t1_attributes.values()) + sum(t2_attributes.values())),
        )

    # a8.0007* of Pm01
    # a8.0008* of Pm01
    # <a href="https://figshare.com/projects/Phallusiamammillata_embryonic_development/64301">Phallusia mammillata
    # embryonic development data</a>
    def test_edist_uted_Pm01a80007_a80008(self):
        t_a80007_nodes = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
        t_a80007_adj = [
            [1, 8],
            [2, 5],
            [3, 4],
            [],
            [],
            [6, 7],
            [],
            [],
            [9, 12],
            [10, 11],
            [],
            [],
            [13, 16],
            [14, 15],
            [],
            [],
            [],
        ]
        t_a80007_attributes = {
            0: 36,
            1: 56,
            2: 72,
            3: 6,
            4: 6,
            5: 66,
            6: 12,
            7: 12,
            8: 36,
            9: 49,
            10: 49,
            11: 49,
            12: 46,
            13: 50,
            14: 2,
            15: 2,
            16: 52,
        }
        t_a80008_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        t_a80008_adj = [
            [1, 8],
            [2, 5],
            [3, 4],
            [],
            [],
            [6, 7],
            [],
            [],
            [9, 12],
            [10, 11],
            [],
            [],
            [13, 14],
            [],
            [],
        ]
        t_a80008_attributes = {
            0: 38,
            1: 39,
            2: 45,
            3: 48,
            4: 48,
            5: 50,
            6: 43,
            7: 43,
            8: 46,
            9: 66,
            10: 20,
            11: 20,
            12: 66,
            13: 20,
            14: 20,
        }

        def local_cost(t1, t2):
            if t1 is None and t2 is None:
                return 0
            elif t1 is None:
                return t_a80008_attributes[t2]
            elif t2 is None:
                return t_a80007_attributes[t1]
            return abs(t_a80007_attributes[t1] - t_a80008_attributes[t2])

        def local_cost_normalized(t1, t2):
            if t1 is None and t2 is None:
                return 0
            elif t1 is None or t2 is None:
                return 1
            return abs(t_a80007_attributes[t1] - t_a80008_attributes[t2]) / (
                t_a80007_attributes[t1] + t_a80008_attributes[t2]
            )

        self.assertEqual(
            89,
            uted.uted(
                t_a80007_nodes,
                t_a80007_adj,
                t_a80008_nodes,
                t_a80008_adj,
                local_cost,
            ),
        )
        # ~0.0033d, // NB: the publication says this should be 0.04d (cf. Fig 3B)
        self.assertEqual(
            3.9974005474699665 / 1213,
            uted.uted(
                t_a80007_nodes,
                t_a80007_adj,
                t_a80008_nodes,
                t_a80008_adj,
                local_cost_normalized,
            )
            / (
                sum(t_a80007_attributes.values())
                + sum(t_a80008_attributes.values())
            ),
        )
