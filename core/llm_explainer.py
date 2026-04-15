"""
LLM-Powered Data Quality Explainer

This module provides natural language explanations for data quality issues
and contextual understanding for fuzzy matching using Large Language Models.

Supports multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models via Ollama
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 500


class DataQualityExplainer:
    """
    Generates natural language explanations for data quality issues using LLMs.

    Features:
    - Explain validation errors in human-readable format
    - Suggest fixes for common data quality issues
    - Provide context-aware explanations
    - Generate summary reports
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the explainer with LLM configuration.

        Args:
            config: LLM configuration (provider, model, API key, etc.)
        """
        self.config = config
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider"""
        if self.config.provider == LLMProvider.OPENAI:
            try:
                import openai
                return openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.config.provider == LLMProvider.OLLAMA:
            try:
                import ollama
                return ollama
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        if self.config.provider == LLMProvider.OPENAI:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content

        elif self.config.provider == LLMProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.config.provider == LLMProvider.OLLAMA:
            response = self.client.chat(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']

        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def explain_validation_issue(
        self,
        column: str,
        value: Any,
        issue: str,
        detail: str,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a natural language explanation for a validation issue.

        Args:
            column: Column name with the issue
            value: The problematic value
            issue: Issue type (e.g., "missing_value", "format_error")
            detail: Detailed description of the issue
            expected: Expected value or format
            context: Additional context (row data, column statistics, etc.)

        Returns:
            Natural language explanation of the issue
        """
        prompt = f"""You are a data quality expert. Explain the following data quality issue in clear, simple language that a business user can understand.

Column: {column}
Value: {value}
Issue Type: {issue}
Details: {detail}
Expected: {expected if expected else 'N/A'}

Additional Context:
{json.dumps(context, indent=2) if context else 'None'}

Provide:
1. A clear explanation of what's wrong
2. Why this is a problem
3. Suggested fix (be specific)

Keep it concise (2-3 sentences maximum)."""

        return self._call_llm(prompt)

    def explain_duplicate_pair(
        self,
        record1: Dict[str, Any],
        record2: Dict[str, Any],
        similarity_score: float,
        matching_columns: List[str]
    ) -> str:
        """
        Explain why two records are considered duplicates.

        Args:
            record1: First record
            record2: Second record
            similarity_score: Similarity score (0-1)
            matching_columns: Columns that contributed to the match

        Returns:
            Natural language explanation of why records are duplicates
        """
        prompt = f"""You are a data quality expert. Explain why these two records are considered potential duplicates.

Record 1:
{json.dumps(record1, indent=2)}

Record 2:
{json.dumps(record2, indent=2)}

Similarity Score: {similarity_score:.2%}
Matching Columns: {', '.join(matching_columns)}

Provide:
1. Why these records appear to be duplicates
2. What fields are similar/different
3. Whether this is likely a true duplicate or false positive
4. Recommended action (merge, keep separate, needs review)

Keep it concise (3-4 sentences)."""

        return self._call_llm(prompt)

    def generate_data_quality_summary(
        self,
        validation_results: Any,
        total_rows: int,
        total_issues: int
    ) -> str:
        """
        Generate a comprehensive summary of data quality issues.

        Args:
            validation_results: DataFrame or dict with validation results
            total_rows: Total number of rows in the dataset
            total_issues: Total number of issues found

        Returns:
            Natural language summary report
        """
        # Convert validation results to JSON-friendly format
        if hasattr(validation_results, 'to_dict'):
            results_dict = validation_results.head(10).to_dict('records')
        else:
            results_dict = validation_results

        prompt = f"""You are a data quality analyst. Generate an executive summary of the data quality issues found.

Dataset Size: {total_rows} rows
Total Issues Found: {total_issues}
Issue Rate: {(total_issues/total_rows*100):.2f}%

Sample Issues (first 10):
{json.dumps(results_dict, indent=2)}

Provide:
1. Overall data quality assessment (Excellent/Good/Fair/Poor)
2. Top 3 most critical issues
3. Recommended priority actions
4. Estimated business impact

Format as a brief executive summary (5-6 sentences)."""

        return self._call_llm(prompt)

    def suggest_validation_rules(
        self,
        column_name: str,
        sample_values: List[Any],
        column_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest validation rules for a column based on sample data.

        Args:
            column_name: Name of the column
            sample_values: Sample values from the column
            column_stats: Statistics about the column (type, null%, unique%, etc.)

        Returns:
            List of suggested validation rules
        """
        prompt = f"""You are a data quality expert. Suggest validation rules for this column.

Column Name: {column_name}
Sample Values: {sample_values[:20]}
Statistics:
{json.dumps(column_stats, indent=2)}

Suggest 3-5 specific validation rules that would catch data quality issues.
Format each rule as a JSON object with keys: "rule_type", "description", "reasoning"

Return only valid JSON array."""

        response = self._call_llm(prompt)

        # Try to parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            rules = json.loads(response)
            return rules if isinstance(rules, list) else [rules]
        except json.JSONDecodeError:
            # Return raw response if JSON parsing fails
            return [{"rule_type": "manual_review", "description": response}]


class ContextualFuzzyMatcher:
    """
    LLM-powered fuzzy matching that understands semantic context.

    Unlike traditional fuzzy matching (Levenshtein, TF-IDF), this uses LLMs
    to understand domain-specific semantics and context.

    Examples:
    - "NYC" matches "New York City" (abbreviation)
    - "Dr. Smith" matches "Doctor Smith" (title expansion)
    - "100 Main St" matches "100 Main Street" (abbreviation)
    """

    def __init__(self, config: LLMConfig, domain: Optional[str] = None):
        """
        Initialize contextual matcher.

        Args:
            config: LLM configuration
            domain: Optional domain context (e.g., "healthcare", "finance", "addresses")
        """
        self.config = config
        self.domain = domain
        self.client = DataQualityExplainer(config).client
        self._call_llm = DataQualityExplainer(config)._call_llm

    def are_semantically_similar(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.8
    ) -> Tuple[bool, float, str]:
        """
        Determine if two text values are semantically similar.

        Args:
            text1: First text value
            text2: Second text value
            threshold: Similarity threshold (0-1)

        Returns:
            Tuple of (is_similar, confidence_score, explanation)
        """
        domain_context = f"\nDomain: {self.domain}" if self.domain else ""

        prompt = f"""Compare these two values and determine if they represent the same entity or concept.{domain_context}

Value 1: "{text1}"
Value 2: "{text2}"

Consider:
- Abbreviations (NYC = New York City)
- Titles (Dr. = Doctor)
- Synonyms (car = automobile)
- Formatting differences (100 Main St = 100 Main Street)
- Common variations

Respond in JSON format:
{{
    "similar": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = self._call_llm(prompt)

        # Parse JSON response
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            is_similar = result.get('similar', False) and result.get('confidence', 0) >= threshold
            return is_similar, result.get('confidence', 0.0), result.get('reasoning', '')

        except json.JSONDecodeError:
            # Fallback: simple keyword matching
            return False, 0.0, "Failed to parse LLM response"

    def batch_match(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Find the most semantically similar candidates for a query.

        Args:
            query: Query string to match
            candidates: List of candidate strings
            top_k: Number of top matches to return

        Returns:
            List of (candidate, score, explanation) tuples
        """
        domain_context = f"\nDomain: {self.domain}" if self.domain else ""

        prompt = f"""Find the top {top_k} most similar matches for the query from the candidate list.{domain_context}

Query: "{query}"

Candidates:
{json.dumps(candidates[:50], indent=2)}

Rank by semantic similarity, considering abbreviations, synonyms, and context.

Return JSON array of top matches:
[
    {{"candidate": "...", "score": 0.0-1.0, "reasoning": "why it matches"}},
    ...
]

Return only valid JSON array."""

        response = self._call_llm(prompt)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            results = json.loads(response)
            return [(r['candidate'], r['score'], r['reasoning']) for r in results[:top_k]]

        except (json.JSONDecodeError, KeyError):
            return []


# Convenience functions for quick usage
def explain_issue(
    column: str,
    value: Any,
    issue: str,
    detail: str,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Quick function to explain a single data quality issue.

    Example:
        explanation = explain_issue(
            column="email",
            value="john@domain",
            issue="format_error",
            detail="Missing top-level domain",
            provider="openai",
            model="gpt-3.5-turbo"
        )
    """
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable not set")

    config = LLMConfig(
        provider=LLMProvider(provider),
        model_name=model,
        api_key=api_key
    )

    explainer = DataQualityExplainer(config)
    return explainer.explain_validation_issue(column, value, issue, detail)


def contextual_match(
    text1: str,
    text2: str,
    domain: Optional[str] = None,
    provider: str = "openai",
    model: str = "gpt-3.5-turbo"
) -> Tuple[bool, float, str]:
    """
    Quick function for contextual fuzzy matching.

    Example:
        is_match, score, reason = contextual_match(
            "NYC",
            "New York City",
            domain="addresses"
        )
    """
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable not set")

    config = LLMConfig(
        provider=LLMProvider(provider),
        model_name=model,
        api_key=api_key
    )

    matcher = ContextualFuzzyMatcher(config, domain)
    return matcher.are_semantically_similar(text1, text2)
