import unittest
from backend import Review, _parse_reviewer_json, route_after_review


class TestMultiAgentLogic(unittest.TestCase):
    def test_parse_reviewer_json_clean(self):
        sample = '{"decision": "PASS", "score": 0.95, "feedback": "", "passed_rules": ["Analogy", "Example"], "failed_rules": []}'
        res = _parse_reviewer_json(sample)
        self.assertEqual(res.decision, "PASS")
        self.assertGreaterEqual(res.score, 0.9)
        self.assertEqual(res.feedback, "")

    def test_parse_reviewer_json_markdown_wrapped(self):
        sample = '```json\n{"decision": "REVISE", "score": 0.6, "feedback": "Add an everyday analogy", "passed_rules": [], "failed_rules": ["analogy"]}\n```'
        res = _parse_reviewer_json(sample)
        self.assertEqual(res.decision, "REVISE")
        self.assertEqual(res.feedback, "Add an everyday analogy")

    def test_routing_logic(self):
        # Case 1: PASS -> done
        state_pass = {"decision": "PASS", "revision_count": 0}
        self.assertEqual(route_after_review(state_pass), "done")

        # Case 2: REVISE with revisions remaining -> revise
        state_revise = {"decision": "REVISE", "revision_count": 1}
        self.assertEqual(route_after_review(state_revise), "revise")

        # Case 3: REVISE with max revisions exceeded -> done
        state_max = {"decision": "REVISE", "revision_count": 3}
        self.assertEqual(route_after_review(state_max), "done")


if __name__ == "__main__":
    unittest.main()
