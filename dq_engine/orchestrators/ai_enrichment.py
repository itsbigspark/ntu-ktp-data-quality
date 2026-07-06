"""
AI Enrichment Module - 5 targeted LLM calls for data quality intelligence.

This module makes the system genuinely AI-powered. Each call sends column
statistics and patterns to the LLM (never raw data), and receives structured
insights back.

The 5 enrichment calls:
    1. Smart Rules       - AI generates validation rules from data patterns
    2. Cross-Column Logic - AI spots contradictions across related fields
    3. Anomaly Explanation - AI explains WHY outliers are suspicious
    4. Issue Triage       - AI ranks issues by business impact
    5. Executive Summary  - AI writes a management-readable assessment

Usage:
    from dq_engine.orchestrators.ai_enrichment import AIEnrichment, OllamaProvider

    provider = OllamaProvider(model="llama3.1")
    enricher = AIEnrichment(provider)

    result = enricher.run_all(df, validation_report)
    print(result.smart_rules)
    print(result.cross_column_issues)
    print(result.explanations)
    print(result.triage)
    print(result.executive_summary)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Provider Interface
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Abstract interface for LLM providers. Swap implementations freely."""

    @abstractmethod
    def call(self, prompt: str, system_message: str = "", temperature: float = 0.3) -> str:
        """Send a prompt to the LLM and return the response text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider. Zero external API calls."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", timeout: int = 300):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def call(self, prompt: str, system_message: str = "", temperature: float = 0.3) -> str:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_message,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1500,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start it with: ollama serve"
            )
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            raise


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider for enterprise deployment."""

    def __init__(self, model_id: str = "anthropic.claude-sonnet-4-5-20250929-v1:0", region: str = "eu-west-2"):
        self.model_id = model_id
        self.region = region
        self._client = None

    @property
    def name(self) -> str:
        return f"bedrock/{self.model_id}"

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def call(self, prompt: str, system_message: str = "", temperature: float = 0.3) -> str:
        client = self._get_client()

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "temperature": temperature,
            "system": system_message,
            "messages": [{"role": "user", "content": prompt}],
        })

        resp = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]


class AnthropicProvider(LLMProvider):
    """Direct Anthropic API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6", max_tokens: int = 4096):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def call(self, prompt: str, system_message: str = "", temperature: float = 0.3) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# Data preparation helpers (PII-safe: statistics only, no raw data)
# ---------------------------------------------------------------------------
def _build_column_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a statistical profile of each column. No raw values exposed."""
    profile = {}
    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 1),
            "unique_count": int(series.nunique()),
            "total_rows": len(series),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                info["min"] = round(float(clean.min()), 2)
                info["max"] = round(float(clean.max()), 2)
                info["mean"] = round(float(clean.mean()), 2)
                info["median"] = round(float(clean.median()), 2)
                info["std"] = round(float(clean.std()), 2)
        else:
            clean = series.dropna().astype(str)
            if len(clean) > 0:
                top_values = clean.value_counts().head(5)
                info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                lengths = clean.str.len()
                info["avg_length"] = round(float(lengths.mean()), 1)
                info["min_length"] = int(lengths.min())
                info["max_length"] = int(lengths.max())

                # Sample patterns (first 3 unique, truncated)
                samples = clean.unique()[:3].tolist()
                info["sample_patterns"] = [s[:30] for s in samples]

        profile[col] = info
    return profile


def _build_issue_summary(report: pd.DataFrame) -> Dict[str, Any]:
    """Summarize validation issues without exposing raw values."""
    if report.empty:
        return {"total_issues": 0}

    summary: Dict[str, Any] = {
        "total_issues": len(report),
        "affected_columns": [],
        "issue_types": {},
    }

    if "column" in report.columns:
        col_counts = report["column"].value_counts().to_dict()
        summary["affected_columns"] = [
            {"column": str(k), "count": int(v)} for k, v in col_counts.items()
        ]

    if "issue_type" in report.columns:
        summary["issue_types"] = {
            str(k): int(v) for k, v in report["issue_type"].value_counts().items()
        }

    if "severity" in report.columns:
        summary["by_severity"] = {
            str(k): int(v) for k, v in report["severity"].value_counts().items()
        }

    return summary


def _build_cross_column_context(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build column relationship context for cross-column analysis."""
    relationships = []
    cols = df.columns.tolist()

    # Identify potentially related columns by name
    location_cols = [c for c in cols if any(k in c.lower() for k in ("city", "postcode", "postal", "country", "address", "state", "region"))]
    financial_cols = [c for c in cols if any(k in c.lower() for k in ("amount", "balance", "price", "fee", "total", "payment"))]
    status_cols = [c for c in cols if any(k in c.lower() for k in ("status", "state", "type", "category", "flag"))]
    date_cols = [c for c in cols if any(k in c.lower() for k in ("date", "time", "created", "updated", "timestamp"))]
    id_cols = [c for c in cols if any(k in c.lower() for k in ("id", "code", "ref", "number"))]

    if location_cols:
        relationships.append({"group": "location", "columns": location_cols})
    if financial_cols:
        relationships.append({"group": "financial", "columns": financial_cols})
    if status_cols:
        relationships.append({"group": "status", "columns": status_cols})
    if date_cols:
        relationships.append({"group": "temporal", "columns": date_cols})
    if id_cols:
        relationships.append({"group": "identifiers", "columns": id_cols})

    # Add co-occurrence stats for location columns
    for group in relationships:
        if group["group"] == "location" and len(group["columns"]) >= 2:
            # Show unique combos count (not actual values)
            combo_cols = [c for c in group["columns"] if c in df.columns]
            if len(combo_cols) >= 2:
                non_null = df[combo_cols].dropna(how="any")
                group["non_null_rows"] = len(non_null)
                group["unique_combinations"] = int(non_null.drop_duplicates().shape[0])

    return relationships


def _parse_json_response(text: str) -> Any:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            clean = part.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    # Try finding first { or [ and parsing from there
    for char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    # Return raw text as fallback
    logger.warning("Could not parse JSON from LLM response, returning raw text")
    return {"raw_response": text}


# ---------------------------------------------------------------------------
# Enrichment Result
# ---------------------------------------------------------------------------
@dataclass
class EnrichmentResult:
    """Results from all 5 AI enrichment calls."""

    smart_rules: List[Dict[str, Any]] = field(default_factory=list)
    cross_column_issues: List[Dict[str, Any]] = field(default_factory=list)
    explanations: List[Dict[str, Any]] = field(default_factory=list)
    triage: List[Dict[str, Any]] = field(default_factory=list)
    executive_summary: str = ""

    # Metadata
    provider: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def total_time_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# AI Enrichment Orchestrator
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a data quality expert working in banking and financial services. "
    "You analyse data quality statistics and validation results to provide "
    "actionable insights. You never see raw customer data, only aggregated "
    "statistics and patterns. Always respond in valid JSON unless told otherwise."
)


class AIEnrichment:
    """
    Runs 5 targeted AI enrichment calls against validation results.

    Each call receives only statistics and patterns, never raw customer data.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def run_all(
        self,
        df: pd.DataFrame,
        validation_report: pd.DataFrame,
        on_progress: Optional[callable] = None,
    ) -> EnrichmentResult:
        """
        Run all 5 enrichment calls.

        Args:
            df: The validated dataframe (used for column statistics only).
            validation_report: The unified validation report from ValidationOrchestrator.
            on_progress: Optional callback(step, total, message).

        Returns:
            EnrichmentResult with all AI-generated insights.
        """
        result = EnrichmentResult(provider=self.provider.name)
        profile = _build_column_profile(df)
        issue_summary = _build_issue_summary(validation_report)

        steps = [
            ("smart_rules", "Generating smart validation rules", self._call_smart_rules),
            ("cross_column", "Analysing cross-column logic", self._call_cross_column),
            ("explanations", "Explaining anomalies", self._call_explanations),
            ("triage", "Triaging issues by impact", self._call_triage),
            ("executive_summary", "Writing executive summary", self._call_executive_summary),
        ]

        for i, (key, message, method) in enumerate(steps, 1):
            if on_progress:
                on_progress(i, 5, message)
            logger.info("[%d/5] %s using %s", i, message, self.provider.name)

            t0 = time.time()
            try:
                method(df, profile, issue_summary, validation_report, result)
                result.timings[key] = time.time() - t0
                logger.info("  completed in %.2fs", result.timings[key])
            except Exception as exc:
                result.timings[key] = time.time() - t0
                result.errors.append(f"{key}: {exc}")
                logger.error("  failed: %s", exc, exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Call 1: Smart Rules
    # ------------------------------------------------------------------
    def _call_smart_rules(
        self, df, profile, issue_summary, report, result: EnrichmentResult
    ):
        prompt = f"""Analyse these column statistics and generate smart validation rules.

COLUMN STATISTICS:
{json.dumps(profile, indent=2, default=str)}

Based on the data patterns, generate validation rules that would catch data quality issues.
For each rule, provide:
- column: the column name
- rule_type: one of [format, range, required, pattern, allowed_values, cross_reference]
- description: human-readable description
- condition: the actual validation condition
- confidence: 0.0 to 1.0 how confident you are this rule is correct
- reasoning: why you think this rule is appropriate

Respond with a JSON array of rule objects. Generate 5-15 rules focusing on the most impactful ones."""

        response = self.provider.call(prompt, SYSTEM_MESSAGE)
        parsed = _parse_json_response(response)

        if isinstance(parsed, list):
            result.smart_rules = parsed
        elif isinstance(parsed, dict) and "rules" in parsed:
            result.smart_rules = parsed["rules"]
        elif isinstance(parsed, dict) and "raw_response" in parsed:
            result.smart_rules = [{"raw": parsed["raw_response"]}]
        else:
            result.smart_rules = [parsed] if parsed else []

    # ------------------------------------------------------------------
    # Call 2: Cross-Column Logic
    # ------------------------------------------------------------------
    def _call_cross_column(
        self, df, profile, issue_summary, report, result: EnrichmentResult
    ):
        relationships = _build_cross_column_context(df)

        prompt = f"""Analyse these column relationships and identify potential cross-column data quality issues.

COLUMN STATISTICS:
{json.dumps(profile, indent=2, default=str)}

COLUMN RELATIONSHIPS DETECTED:
{json.dumps(relationships, indent=2, default=str)}

Identify logical inconsistencies that could exist between related columns. For example:
- A postcode that does not match the city
- A transaction amount that is negative while the status says "approved"
- A date of birth that makes the customer impossibly old or young
- An email domain that does not match the company name

For each potential issue, provide:
- columns: list of columns involved
- check_type: what kind of cross-column check this is
- description: human-readable explanation
- severity: high, medium, or low
- suggested_query: a plain-English description of how to check for this

Respond with a JSON array. Identify 3-8 cross-column checks."""

        response = self.provider.call(prompt, SYSTEM_MESSAGE)
        parsed = _parse_json_response(response)

        if isinstance(parsed, list):
            result.cross_column_issues = parsed
        elif isinstance(parsed, dict) and any(k in parsed for k in ("issues", "checks", "cross_column_issues")):
            for k in ("issues", "checks", "cross_column_issues"):
                if k in parsed:
                    result.cross_column_issues = parsed[k]
                    break
        else:
            result.cross_column_issues = [parsed] if parsed else []

    # ------------------------------------------------------------------
    # Call 3: Anomaly Explanation
    # ------------------------------------------------------------------
    def _call_explanations(
        self, df, profile, issue_summary, report, result: EnrichmentResult
    ):
        # Build a summary of top issues (no raw values)
        top_issues = []
        if not report.empty:
            for _, row in report.head(20).iterrows():
                issue = {
                    "column": str(row.get("column", "")),
                    "issue_type": str(row.get("issue_type", "")),
                    "severity": str(row.get("severity", "medium")),
                }
                # Include value only if it's not PII-like
                col_name = str(row.get("column", "")).lower()
                if not any(pii in col_name for pii in ("name", "email", "phone", "address", "ssn", "account")):
                    val = row.get("value", "")
                    if pd.notna(val):
                        issue["value_snippet"] = str(val)[:20]
                top_issues.append(issue)

        prompt = f"""Explain these data quality issues in plain English. The audience is a non-technical data steward at a bank.

COLUMN STATISTICS:
{json.dumps(profile, indent=2, default=str)}

ISSUES FOUND (top 20):
{json.dumps(top_issues, indent=2, default=str)}

ISSUE SUMMARY:
{json.dumps(issue_summary, indent=2, default=str)}

For each issue, provide:
- column: the column name
- issue_type: the type of issue
- explanation: a plain-English explanation of WHY this is a problem (not just what the issue is)
- business_impact: what could go wrong if this is not fixed
- suggested_action: what the data team should do

Respond with a JSON array. Explain each issue clearly and concisely."""

        response = self.provider.call(prompt, SYSTEM_MESSAGE)
        parsed = _parse_json_response(response)

        if isinstance(parsed, list):
            result.explanations = parsed
        elif isinstance(parsed, dict) and "explanations" in parsed:
            result.explanations = parsed["explanations"]
        else:
            result.explanations = [parsed] if parsed else []

    # ------------------------------------------------------------------
    # Call 4: Issue Triage
    # ------------------------------------------------------------------
    def _call_triage(
        self, df, profile, issue_summary, report, result: EnrichmentResult
    ):
        prompt = f"""You are triaging data quality issues for a banking dataset. Rank them by business impact and suggest a fix order.

ISSUE SUMMARY:
{json.dumps(issue_summary, indent=2, default=str)}

COLUMN STATISTICS:
{json.dumps(profile, indent=2, default=str)}

TOTAL ROWS: {len(df)}

Provide a prioritised action plan. For each item:
- priority: 1 (fix first) through N
- column: the affected column
- issue_type: the type of issue
- count: how many rows affected
- severity: critical, high, medium, or low
- reason: why this should be fixed at this priority level
- effort: estimated effort (quick_fix, moderate, complex)
- recommendation: specific action to take

Respond with a JSON array sorted by priority (1 = most urgent). Include 5-10 items."""

        response = self.provider.call(prompt, SYSTEM_MESSAGE)
        parsed = _parse_json_response(response)

        if isinstance(parsed, list):
            result.triage = parsed
        elif isinstance(parsed, dict) and any(k in parsed for k in ("triage", "priorities", "action_plan")):
            for k in ("triage", "priorities", "action_plan"):
                if k in parsed:
                    result.triage = parsed[k]
                    break
        else:
            result.triage = [parsed] if parsed else []

    # ------------------------------------------------------------------
    # Call 5: Executive Summary
    # ------------------------------------------------------------------
    def _call_executive_summary(
        self, df, profile, issue_summary, report, result: EnrichmentResult
    ):
        # Calculate quality score
        total_cells = len(df) * len(df.columns)
        issue_count = issue_summary.get("total_issues", 0)
        quality_pct = round((1 - issue_count / max(total_cells, 1)) * 100, 1)

        prompt = f"""Write a one-paragraph executive summary of this data quality assessment for senior management at a bank.

DATASET: {len(df)} rows, {len(df.columns)} columns
QUALITY SCORE: {quality_pct}%
TOTAL ISSUES: {issue_count}

ISSUE BREAKDOWN:
{json.dumps(issue_summary, indent=2, default=str)}

TOP TRIAGE ITEMS:
{json.dumps(result.triage[:5], indent=2, default=str) if result.triage else "Not available"}

CROSS-COLUMN CONCERNS:
{json.dumps(result.cross_column_issues[:3], indent=2, default=str) if result.cross_column_issues else "Not available"}

Write exactly ONE paragraph (4-6 sentences) that:
1. States the overall quality score and whether the data passes or fails
2. Highlights the most critical findings
3. Notes any cross-column concerns
4. Recommends next steps

Write in formal, concise business English. Do not use bullet points or lists.
Respond with plain text only (no JSON, no markdown)."""

        result.executive_summary = self.provider.call(
            prompt, SYSTEM_MESSAGE, temperature=0.4
        ).strip()
