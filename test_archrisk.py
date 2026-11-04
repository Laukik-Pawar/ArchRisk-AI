
import unittest
import networkx as nx

from archrisk import build_graph, simulation_scores, validate


class TestArchRisk(unittest.TestCase):

    def setUp(self):
        self.architecture = {
            "system": "Test E-Commerce",
            "components": [
                {
                    "id": "frontend",
                    "name": "Frontend",
                    "type": "gateway",
                    "description": "User interface",
                    "criticality": 0.5,
                },
                {
                    "id": "order_service",
                    "name": "Order Service",
                    "type": "service",
                    "description": "Processes orders",
                    "criticality": 0.8,
                },
                {
                    "id": "database",
                    "name": "Database",
                    "type": "database",
                    "description": "Stores orders",
                    "criticality": 0.9,
                },
            ],
            "dependencies": [
                {"source": "frontend", "target": "order_service"},
                {"source": "order_service", "target": "database"},
            ],
        }

    def test_architecture_validation(self):
        result = validate(self.architecture)
        self.assertEqual(len(result["components"]), 3)

    def test_graph_construction(self):
        graph = build_graph(self.architecture)

        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)

        self.assertTrue(
            graph.has_edge("frontend", "order_service")
        )

    def test_failure_propagation(self):
        graph = build_graph(self.architecture)
        scores = simulation_scores(graph)

        self.assertAlmostEqual(scores["database"], 1.0)
        self.assertAlmostEqual(scores["order_service"], 0.5)
        self.assertAlmostEqual(scores["frontend"], 0.0)

    def test_duplicate_components_rejected(self):
        invalid = {
            **self.architecture,
            "components": (
                self.architecture["components"]
                + [self.architecture["components"][0].copy()]
            ),
        }

        with self.assertRaises(ValueError):
            validate(invalid)

    def test_unknown_dependency_rejected(self):
        invalid = {
            **self.architecture,
            "dependencies": [
                {"source": "frontend", "target": "missing_service"}
            ],
        }

        with self.assertRaises(ValueError):
            validate(invalid)


if __name__ == "__main__":
    unittest.main()
