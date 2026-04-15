"""
core/ai_enrichment.py  --  AI Enrichment Layer for the DQ Pipeline
===================================================================
5 targeted LLM calls per batch that add intelligence on top of the
deterministic validation engine.

Supports 3 providers (configured via config.yaml):
  - anthropic:  Claude via Anthropic API (cloud)
  - bedrock:    Claude via AWS Bedrock (cloud, VPC-contained)
  - ollama:     Any local model via Ollama (fully on-premises)

Each enrichment function:
  - Accepts a config dict (reads provider from config["llm"])
  - Sends ONLY anonymised samples / aggregated stats to the LLM
  - Returns structured output that merges into the pipeline result
  - Fails gracefully — enrichment is optional, pipeline never breaks

Usage:
  from core.ai_enrichment import run_all_enrichments
  enriched = run_all_enrichments(result, df, config)
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger("dq_engine.ai")


# ============================================================================
# LLM Client — unified interface across providers
# ============================================================================

class LLMClient:
    """
    Unified LLM interface. One class, three backends.

    Usage:
        client = LLMClient(config)
        response = client.call("You are a data analyst.", "Explain this anomaly...")
    """

    def __init__(self, config: Dict[str, Any]):
        llm_cfg = config.get("llm", {})
        self.provider = llm_cfg.get("provider", "ollama")
        self.model = llm_cfg.get("model", "")
        self._config = llm_cfg
        self._client = None

        logger.info(f"LLM provider: {self.provider} | model: {self.model}")

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Send a prompt to the configured LLM and return the text response."""
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, max_tokens)
        elif self.provider == "bedrock":
            return self._call_bedrock(system_prompt, user_prompt, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        """Claude via Anthropic API."""
        import anthropic

        api_key = self._config.get("anthropic", {}).get("api_key", "")
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not self._client:
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        model = self.model or "claude-sonnet-4-20250514"
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _call_bedrock(self, system: str, user: str, max_tokens: int) -> str:
        """Claude via AWS Bedrock."""
        import anthropic

        region = self._config.get("bedrock", {}).get("region", "eu-west-2")
        model_id = self._config.get("bedrock", {}).get("model_id", "anthropic.claude-sonnet-4-20250514-v1:0")

        if not self._client:
            self._client = anthropic.AnthropicBedrock(aws_region=region)

        response = self._client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        """Any model via local Ollama."""
        import ollama as ollama_lib

        host = self._config.get("ollama", {}).get("host", "http://localhost:11434")
        model = self._config.get("ollama", {}).get("model", "llama3.1:8b")

        # Set host if non-default
        if host != "http://localhost:11434":
            import os
            os.environ["OLLAMA_HOST"] = host

        response = ollama_lib.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.message.content


# ============================================================================
# PII Masking — protects data before it reaches the LLM
# ============================================================================

def _mask_pii_columns(df: pd.DataFrame, columns_to_mask: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Mask columns that likely contain PII.
    Auto-detects by column name if no explicit list is provided.
    """
    pii_hints = [
        "name", "email", "phone", "address", "ssn", "national_id",
        "passport", "dob", "date_of_birth", "account_number", "card",
        "iban", "sort_code", "routing", "social_security",
    ]

    masked = df.copy()

    if columns_to_mask is None:
        columns_to_mask = []
        for col in df.columns:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            if any(hint in col_lower for hint in pii_hints):
                columns_to_mask.append(col)

    for col in columns_to_mask:
        if col in masked.columns:
            masked[col] = "[MASKED]"

    if columns_to_mask:
        logger.info(f"PII masked: {columns_to_mask}")

    return masked


def _sample_rows(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Take a representative sample for LLM context."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


# ============================================================================
# Enrichment 1: Intelligent Rule Generation
# ============================================================================

def enrich_rules(
    df: pd.DataFrame,
    existing_rules: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask the LLM to review column names + sample values and suggest
    smarter rules than pure regex inference can produce.

    What the LLM sees: column names + 10 sample values per column (PII masked).
    What the LLM does NOT see: full dataset.
    """
    t0 = time.time()
    client = LLMClient(config)

    # Prepare context — masked samples
    mask_pii = config.get("llm", {}).get("context", {}).get("mask_pii", True)
    sample_df = _sample_rows(df, n=10)
    if mask_pii:
        sample_df = _mask_pii_columns(sample_df)

    column_info = []
    for col in df.columns:
        vals = sample_df[col].dropna().astype(str).tolist()[:10]
        nulls = int(df[col].isna().sum())
        uniques = int(df[col].nunique())
        column_info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nulls": nulls,
            "unique_values": uniques,
            "total_rows": len(df),
            "sample_values": vals,
        })

    system = (
        "You are a data quality expert. Given column metadata and sample values, "
        "suggest validation rules. Return valid JSON only — no markdown, no explanation."
    )

    user = f"""Analyse these columns and suggest validation rules.
For each column, suggest: allowed patterns, expected types, value ranges, or allowed values.

Current rules (auto-inferred): {json.dumps(existing_rules, indent=2)[:2000]}

Column metadata:
{json.dumps(column_info, indent=2)}

Return a JSON object where keys are column names and values are rule objects like:
{{"column_name": {{"type": "string", "pattern": "regex", "allowed_values": [...], "min": 0, "max": 100, "nullable": true, "notes": "..."}}}}

Only include columns where you can add value beyond the existing rules."""

    try:
        response = client.call(system, user, max_tokens=2048)
        # Parse JSON from response
        rules = _extract_json(response)
        duration = round((time.time() - t0) * 1000)
        logger.info(f"AI rule enrichment: {len(rules)} columns enhanced ({duration}ms)")
        return {
            "enriched_rules": rules,
            "duration_ms": duration,
            "llm_provider": client.provider,
        }
    except Exception as e:
        logger.warning(f"AI rule enrichment failed: {e}")
        return {"enriched_rules": {}, "duration_ms": round((time.time() - t0) * 1000), "error": str(e)}


# ============================================================================
# Enrichment 2: Cross-Column Relationship Validation
# ============================================================================

def enrich_cross_column(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ask the LLM to spot logical inconsistencies across columns.

    Examples: UK customer with Indian phone, closed account with recent transaction,
    child age with adult account type.

    What the LLM sees: 30 masked rows.
    """
    t0 = time.time()
    client = LLMClient(config)

    mask_pii = config.get("llm", {}).get("context", {}).get("mask_pii", True)
    sample = _sample_rows(df, n=30)
    if mask_pii:
        sample = _mask_pii_columns(sample)

    system = (
        "You are a data quality analyst reviewing banking data. "
        "Look for logical inconsistencies ACROSS columns — values that individually "
        "look fine but together don't make sense. Return valid JSON only."
    )

    user = f"""Review these rows for cross-column logical inconsistencies.

Columns: {list(df.columns)}

Sample data (JSON):
{sample.to_json(orient='records', indent=2)[:3000]}

Return a JSON array of findings:
[{{"row_index": 0, "columns_involved": ["col1", "col2"], "issue": "description", "severity": "high|medium|low"}}]

Only report genuine logical conflicts. Return [] if none found."""

    try:
        response = client.call(system, user, max_tokens=1024)
        findings = _extract_json(response)
        if not isinstance(findings, list):
            findings = []
        duration = round((time.time() - t0) * 1000)
        logger.info(f"AI cross-column check: {len(findings)} findings ({duration}ms)")
        return {
            "cross_column_findings": findings,
            "duration_ms": duration,
            "llm_provider": client.provider,
        }
    except Exception as e:
        logger.warning(f"AI cross-column check failed: {e}")
        return {"cross_column_findings": [], "duration_ms": round((time.time() - t0) * 1000), "error": str(e)}


# ============================================================================
# Enrichment 3: Anomaly Explanation
# ============================================================================

def enrich_anomaly_explanations(
    issues: pd.DataFrame,
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For anomalies flagged by the ML/statistical engine, ask the LLM
    to explain WHY they might be anomalous in plain English.

    What the LLM sees: up to 20 flagged rows (PII masked) with their issue descriptions.
    """
    t0 = time.time()

    # Filter to anomaly-type issues only
    anomaly_issues = issues[issues["issue"].str.contains("anomaly|outlier|unusual", case=False, na=False)]
    if anomaly_issues.empty:
        logger.info("No anomalies to explain")
        return {"anomaly_explanations": [], "duration_ms": 0}

    client = LLMClient(config)

    # Get the flagged rows with context
    flagged_rows = anomaly_issues.head(20)
    mask_pii = config.get("llm", {}).get("context", {}).get("mask_pii", True)

    context_rows = []
    for _, issue_row in flagged_rows.iterrows():
        rid = int(issue_row.get("row_id", 0))
        if rid < len(df):
            row_data = df.iloc[rid].to_dict()
            if mask_pii:
                row_data = _mask_pii_columns(pd.DataFrame([row_data])).iloc[0].to_dict()
            context_rows.append({
                "row_id": rid,
                "column": str(issue_row.get("column", "")),
                "value": str(issue_row.get("value", "")),
                "issue": str(issue_row.get("detail", "")),
                "full_row": {k: str(v) for k, v in row_data.items()},
            })

    if not context_rows:
        return {"anomaly_explanations": [], "duration_ms": 0}

    system = (
        "You are a data quality analyst. Explain why each flagged row is anomalous "
        "in plain business English. Consider the full row context. Return valid JSON only."
    )

    user = f"""Explain these anomalies in plain English:

{json.dumps(context_rows[:10], indent=2)[:3000]}

Return a JSON array:
[{{"row_id": 0, "explanation": "Plain English explanation of why this is unusual", "risk_level": "high|medium|low", "recommendation": "What to do about it"}}]"""

    try:
        response = client.call(system, user, max_tokens=1500)
        explanations = _extract_json(response)
        if not isinstance(explanations, list):
            explanations = []
        duration = round((time.time() - t0) * 1000)
        logger.info(f"AI anomaly explanations: {len(explanations)} explained ({duration}ms)")
        return {
            "anomaly_explanations": explanations,
            "duration_ms": duration,
            "llm_provider": client.provider,
        }
    except Exception as e:
        logger.warning(f"AI anomaly explanation failed: {e}")
        return {"anomaly_explanations": [], "duration_ms": round((time.time() - t0) * 1000), "error": str(e)}


# ============================================================================
# Enrichment 4: Smart Issue Categorisation (by business impact)
# ============================================================================

def enrich_issue_triage(
    issues: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Categorise all issues by business impact rather than technical type.

    What the LLM sees: issue type counts and column distributions (NO actual data).
    """
    t0 = time.time()

    if issues.empty:
        return {"triage": [], "duration_ms": 0}

    client = LLMClient(config)

    # Aggregate — no actual data values sent
    issue_col = "issue" if "issue" in issues.columns else "issue_type"
    summary = {
        "total_issues": len(issues),
        "by_type": issues[issue_col].value_counts().to_dict() if issue_col in issues.columns else {},
        "by_severity": issues["severity"].value_counts().to_dict() if "severity" in issues.columns else {},
        "by_column": issues["column"].value_counts().head(10).to_dict() if "column" in issues.columns else {},
        "columns_affected": int(issues["column"].nunique()) if "column" in issues.columns else 0,
    }

    system = (
        "You are a banking compliance data analyst. Categorise data quality issues "
        "by business impact for a compliance team. Return valid JSON only."
    )

    user = f"""Categorise these data quality issues by business impact:

Issue summary (NO actual data — just counts):
{json.dumps(summary, indent=2)}

Return a JSON array of priority categories:
[{{"category": "Potential Fraud Indicators", "severity": "critical", "issue_types": [...], "estimated_count": 0, "action": "Escalate to compliance team", "reason": "Why this matters"}}]

Categories to consider: fraud risk, regulatory compliance (KYC/AML), data entry errors (auto-fixable), missing mandatory fields, duplicates, format inconsistencies."""

    try:
        response = client.call(system, user, max_tokens=1024)
        triage = _extract_json(response)
        if not isinstance(triage, list):
            triage = []
        duration = round((time.time() - t0) * 1000)
        logger.info(f"AI issue triage: {len(triage)} categories ({duration}ms)")
        return {
            "triage": triage,
            "duration_ms": duration,
            "llm_provider": client.provider,
        }
    except Exception as e:
        logger.warning(f"AI issue triage failed: {e}")
        return {"triage": [], "duration_ms": round((time.time() - t0) * 1000), "error": str(e)}


# ============================================================================
# Enrichment 5: Executive Summary Report
# ============================================================================

def enrich_executive_summary(
    result: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a one-paragraph executive summary of the batch results.

    What the LLM sees: aggregated scores and counts only (NO actual data).
    """
    t0 = time.time()
    client = LLMClient(config)

    # Only aggregated stats — zero actual data
    context = {
        "batch_id": result.get("batch_id", ""),
        "rows_processed": result.get("rows_processed", 0),
        "issues_count": result.get("issues_count", 0),
        "overall_score": result.get("overall_score", 0),
        "pass": result.get("pass", False),
        "pass_threshold": result.get("pass_threshold", 85.0),
        "quality_scores": {k: v for k, v in result.get("quality_scores", {}).items() if k != "missing_by_column"},
        "columns": result.get("columns", []),
    }

    system = (
        "You are a data quality analyst writing a brief executive summary "
        "for a banking compliance manager. Be concise, factual, and actionable. "
        "Write in plain English, not technical jargon."
    )

    user = f"""Write a 3-4 sentence executive summary for this data quality batch:

{json.dumps(context, indent=2)}

Include: overall assessment, key risk areas, and recommended next steps.
Return plain text only — no JSON, no markdown headers."""

    try:
        response = client.call(system, user, max_tokens=300)
        duration = round((time.time() - t0) * 1000)
        logger.info(f"AI executive summary generated ({duration}ms)")
        return {
            "executive_summary": response.strip(),
            "duration_ms": duration,
            "llm_provider": client.provider,
        }
    except Exception as e:
        logger.warning(f"AI executive summary failed: {e}")
        return {"executive_summary": "", "duration_ms": round((time.time() - t0) * 1000), "error": str(e)}


# ============================================================================
# Master function — runs all 5 enrichments
# ============================================================================

def run_all_enrichments(
    result: Dict[str, Any],
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all 5 AI enrichment steps. Each fails independently.
    Returns a dict with all enrichment results + total stats.

    Controlled by config["llm"]["enrichment"]["enabled"] (default: False).
    Individual enrichments can be toggled:
        config["llm"]["enrichment"]["rule_generation"] = True
        config["llm"]["enrichment"]["cross_column"] = True
        config["llm"]["enrichment"]["anomaly_explanation"] = True
        config["llm"]["enrichment"]["issue_triage"] = True
        config["llm"]["enrichment"]["executive_summary"] = True
    """
    enrich_cfg = config.get("llm", {}).get("enrichment", {})
    if not enrich_cfg.get("enabled", False):
        logger.info("AI enrichment disabled (set llm.enrichment.enabled=true to enable)")
        return {"enrichment_enabled": False, "llm_calls": 0}

    t_start = time.time()
    enriched: Dict[str, Any] = {"enrichment_enabled": True}
    llm_calls = 0
    issues = result.get("issues", pd.DataFrame())
    rules = result.get("rules", {})

    # 1. Rule generation
    if enrich_cfg.get("rule_generation", True):
        logger.info("Enrichment 1/5: Intelligent rule generation")
        enriched["rules"] = enrich_rules(df, rules, config)
        llm_calls += 1

    # 2. Cross-column checks
    if enrich_cfg.get("cross_column", True):
        logger.info("Enrichment 2/5: Cross-column relationship validation")
        enriched["cross_column"] = enrich_cross_column(df, config)
        llm_calls += 1

    # 3. Anomaly explanations
    if enrich_cfg.get("anomaly_explanation", True):
        logger.info("Enrichment 3/5: Anomaly explanations")
        enriched["anomaly_explanations"] = enrich_anomaly_explanations(issues, df, config)
        llm_calls += 1

    # 4. Issue triage
    if enrich_cfg.get("issue_triage", True):
        logger.info("Enrichment 4/5: Smart issue categorisation")
        enriched["issue_triage"] = enrich_issue_triage(issues, config)
        llm_calls += 1

    # 5. Executive summary
    if enrich_cfg.get("executive_summary", True):
        logger.info("Enrichment 5/5: Executive summary")
        enriched["executive_summary"] = enrich_executive_summary(result, config)
        llm_calls += 1

    enriched["llm_calls"] = llm_calls
    enriched["total_duration_ms"] = round((time.time() - t_start) * 1000)
    enriched["llm_provider"] = config.get("llm", {}).get("provider", "unknown")

    logger.info(
        f"AI enrichment complete: {llm_calls} LLM calls, "
        f"{enriched['total_duration_ms']}ms total, "
        f"provider: {enriched['llm_provider']}"
    )
    return enriched


# ============================================================================
# Helpers
# ============================================================================

def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Could not parse JSON from LLM response: {text[:200]}")
        return {}
