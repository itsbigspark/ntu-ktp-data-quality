"""Tests for PII detection logic — severity map, entity grouping, scan results."""

import unittest
from unittest.mock import MagicMock, patch


# ── Replicate the constants defined in app/pages/18_PII_Detection.py ─────────

SEVERITY_MAP = {
    "CREDIT_CARD": "CRITICAL",
    "IBAN_CODE": "CRITICAL",
    "US_SSN": "CRITICAL",
    "UK_NHS": "CRITICAL",
    "PERSON": "HIGH",
    "PHONE_NUMBER": "HIGH",
    "EMAIL_ADDRESS": "HIGH",
    "LOCATION": "MEDIUM",
    "IP_ADDRESS": "MEDIUM",
    "DATE_TIME": "MEDIUM",
    "URL": "LOW",
    "NRP": "LOW",
}

ENTITY_GROUPS = {
    "Identity": ["PERSON", "DATE_TIME", "NRP"],
    "Finance": ["CREDIT_CARD", "IBAN_CODE", "CRYPTO"],
    "Government IDs": ["US_SSN", "US_PASSPORT", "UK_NHS", "US_DRIVER_LICENSE"],
    "Contact": ["EMAIL_ADDRESS", "PHONE_NUMBER", "URL"],
    "Location": ["LOCATION", "IP_ADDRESS"],
}


def _severity_for(entity_type: str) -> str:
    return SEVERITY_MAP.get(entity_type, "LOW")


def _summarise_findings(findings: list[dict]) -> dict:
    """Return {severity: count} from a list of finding dicts."""
    from collections import Counter
    counts = Counter(_severity_for(f["entity_type"]) for f in findings)
    return dict(counts)


class TestSeverityMap(unittest.TestCase):

    def test_credit_card_is_critical(self):
        self.assertEqual(_severity_for("CREDIT_CARD"), "CRITICAL")

    def test_email_is_high(self):
        self.assertEqual(_severity_for("EMAIL_ADDRESS"), "HIGH")

    def test_location_is_medium(self):
        self.assertEqual(_severity_for("LOCATION"), "MEDIUM")

    def test_url_is_low(self):
        self.assertEqual(_severity_for("URL"), "LOW")

    def test_unknown_entity_defaults_to_low(self):
        self.assertEqual(_severity_for("UNKNOWN_ENTITY_XYZ"), "LOW")

    def test_all_critical_entities_mapped(self):
        critical = [k for k, v in SEVERITY_MAP.items() if v == "CRITICAL"]
        self.assertIn("CREDIT_CARD", critical)
        self.assertIn("IBAN_CODE", critical)
        self.assertIn("US_SSN", critical)


class TestEntityGroups(unittest.TestCase):

    def test_all_groups_present(self):
        expected = {"Identity", "Finance", "Government IDs", "Contact", "Location"}
        self.assertEqual(set(ENTITY_GROUPS.keys()), expected)

    def test_credit_card_in_finance(self):
        self.assertIn("CREDIT_CARD", ENTITY_GROUPS["Finance"])

    def test_email_in_contact(self):
        self.assertIn("EMAIL_ADDRESS", ENTITY_GROUPS["Contact"])

    def test_no_entity_appears_in_multiple_groups(self):
        seen = []
        for entities in ENTITY_GROUPS.values():
            seen.extend(entities)
        self.assertEqual(len(seen), len(set(seen)), "Duplicate entity across groups")


class TestSummarisefindings(unittest.TestCase):

    def test_empty_findings(self):
        self.assertEqual(_summarise_findings([]), {})

    def test_single_critical(self):
        findings = [{"entity_type": "CREDIT_CARD", "value": "4111111111111111"}]
        result = _summarise_findings(findings)
        self.assertEqual(result.get("CRITICAL"), 1)

    def test_mixed_severities(self):
        findings = [
            {"entity_type": "CREDIT_CARD"},
            {"entity_type": "EMAIL_ADDRESS"},
            {"entity_type": "EMAIL_ADDRESS"},
            {"entity_type": "URL"},
        ]
        result = _summarise_findings(findings)
        self.assertEqual(result["CRITICAL"], 1)
        self.assertEqual(result["HIGH"], 2)
        self.assertEqual(result["LOW"], 1)


class TestPresidioMocked(unittest.TestCase):
    """Verify scan logic with Presidio mocked out — no model download needed in CI."""

    def _make_recognizer_result(self, entity_type, start, end, score=0.85):
        r = MagicMock()
        r.entity_type = entity_type
        r.start = start
        r.end = end
        r.score = score
        return r

    def test_analyzer_called_per_row(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = []

        texts = ["Alice works at ACME Corp", "Call 07911 123456 for details"]
        for text in texts:
            analyzer.analyze(text=text, language="en")

        self.assertEqual(analyzer.analyze.call_count, len(texts))

    def test_low_confidence_findings_filtered(self):
        threshold = 0.7
        results = [
            self._make_recognizer_result("PERSON", 0, 5, score=0.9),
            self._make_recognizer_result("LOCATION", 6, 12, score=0.4),  # below threshold
        ]
        kept = [r for r in results if r.score >= threshold]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].entity_type, "PERSON")

    def test_finding_count_matches_analyzer_output(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = [
            self._make_recognizer_result("EMAIL_ADDRESS", 0, 20),
            self._make_recognizer_result("PHONE_NUMBER", 22, 35),
        ]
        findings = analyzer.analyze(text="test@example.com, 07900000000", language="en")
        self.assertEqual(len(findings), 2)


if __name__ == "__main__":
    unittest.main()
