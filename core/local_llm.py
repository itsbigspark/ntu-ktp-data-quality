"""
Local On-Premise LLM with Continuous Learning

This module provides a privacy-focused, on-premise LLM solution using Ollama
with context memory and continuous learning capabilities.

Features:
- Runs completely offline (no external API calls)
- Learns from your data quality patterns over time
- Maintains context memory for domain-specific understanding
- Fine-tunable for your specific use cases
- Zero data leakage (everything stays on your infrastructure)
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import hashlib


class LocalLLMConfig:
    """Configuration for on-premise LLM"""
    def __init__(
        self,
        model_name: str = "mistral:7b-instruct",  # Recommended: best quality/performance
        context_window: int = 8192,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        num_predict: int = 500,
        repeat_penalty: float = 1.1,
        memory_db_path: str = "data/llm_memory.db"
    ):
        self.model_name = model_name
        self.context_window = context_window
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.memory_db_path = memory_db_path


class ContextMemory:
    """
    Persistent memory system for the LLM to learn from past interactions.

    This allows the LLM to:
    - Remember data quality patterns in your domain
    - Learn from corrections and feedback
    - Build domain-specific knowledge over time
    """

    def __init__(self, db_path: str = "data/llm_memory.db"):
        """Initialize context memory with SQLite database"""
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create database and tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for storing learned patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                column_name TEXT,
                pattern TEXT NOT NULL,
                explanation TEXT,
                frequency INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for storing validation explanations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_type TEXT NOT NULL,
                column_type TEXT,
                explanation TEXT NOT NULL,
                feedback TEXT,
                usefulness_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for storing duplicate matching rules
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matching_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                text1 TEXT NOT NULL,
                text2 TEXT NOT NULL,
                is_match BOOLEAN NOT NULL,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for domain vocabulary (company-specific terms)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE NOT NULL,
                meaning TEXT,
                category TEXT,
                examples TEXT,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def store_pattern(
        self,
        pattern_type: str,
        pattern: str,
        column_name: Optional[str] = None,
        explanation: Optional[str] = None
    ):
        """Store a learned data quality pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern already exists
        cursor.execute(
            "SELECT id, frequency FROM learned_patterns WHERE pattern_type=? AND pattern=?",
            (pattern_type, pattern)
        )
        result = cursor.fetchone()

        if result:
            # Increment frequency
            cursor.execute(
                "UPDATE learned_patterns SET frequency=?, last_seen=? WHERE id=?",
                (result[1] + 1, datetime.now(), result[0])
            )
        else:
            # Insert new pattern
            cursor.execute(
                """INSERT INTO learned_patterns
                   (pattern_type, column_name, pattern, explanation)
                   VALUES (?, ?, ?, ?)""",
                (pattern_type, column_name, pattern, explanation)
            )

        conn.commit()
        conn.close()

    def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict]:
        """Retrieve learned patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if pattern_type:
            cursor.execute(
                """SELECT pattern_type, column_name, pattern, explanation, frequency, confidence
                   FROM learned_patterns WHERE pattern_type=?
                   ORDER BY frequency DESC, confidence DESC""",
                (pattern_type,)
            )
        else:
            cursor.execute(
                """SELECT pattern_type, column_name, pattern, explanation, frequency, confidence
                   FROM learned_patterns
                   ORDER BY frequency DESC, confidence DESC"""
            )

        results = cursor.fetchall()
        conn.close()

        return [
            {
                'pattern_type': r[0],
                'column_name': r[1],
                'pattern': r[2],
                'explanation': r[3],
                'frequency': r[4],
                'confidence': r[5]
            }
            for r in results
        ]

    def store_explanation(
        self,
        issue_type: str,
        explanation: str,
        column_type: Optional[str] = None
    ):
        """Store a validation explanation for future reference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO validation_explanations
               (issue_type, column_type, explanation)
               VALUES (?, ?, ?)""",
            (issue_type, column_type, explanation)
        )

        conn.commit()
        conn.close()

    def get_similar_explanations(self, issue_type: str) -> List[str]:
        """Get previously successful explanations for similar issues"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT explanation FROM validation_explanations
               WHERE issue_type=?
               ORDER BY usefulness_score DESC, created_at DESC
               LIMIT 3""",
            (issue_type,)
        )

        results = cursor.fetchall()
        conn.close()

        return [r[0] for r in results]

    def store_matching_rule(
        self,
        text1: str,
        text2: str,
        is_match: bool,
        reasoning: str,
        domain: Optional[str] = None
    ):
        """Store a matching rule for future reference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO matching_rules
               (domain, text1, text2, is_match, reasoning)
               VALUES (?, ?, ?, ?, ?)""",
            (domain, text1, text2, is_match, reasoning)
        )

        conn.commit()
        conn.close()

    def add_domain_term(self, term: str, meaning: str, category: str = "general"):
        """Add company/domain-specific terminology"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT OR REPLACE INTO domain_vocabulary
               (term, meaning, category)
               VALUES (?, ?, ?)""",
            (term, meaning, category)
        )

        conn.commit()
        conn.close()

    def get_domain_context(self) -> str:
        """Get domain-specific context for the LLM"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT term, meaning, category FROM domain_vocabulary
               ORDER BY frequency DESC LIMIT 50"""
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return ""

        context = "\n\nDomain-Specific Terminology:\n"
        for term, meaning, category in results:
            context += f"- {term} ({category}): {meaning}\n"

        return context

    def get_stats(self) -> Dict:
        """Get statistics about learned knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Count patterns
        cursor.execute("SELECT COUNT(*) FROM learned_patterns")
        stats['learned_patterns'] = cursor.fetchone()[0]

        # Count explanations
        cursor.execute("SELECT COUNT(*) FROM validation_explanations")
        stats['stored_explanations'] = cursor.fetchone()[0]

        # Count matching rules
        cursor.execute("SELECT COUNT(*) FROM matching_rules")
        stats['matching_rules'] = cursor.fetchone()[0]

        # Count domain terms
        cursor.execute("SELECT COUNT(*) FROM domain_vocabulary")
        stats['domain_terms'] = cursor.fetchone()[0]

        conn.close()
        return stats


class LocalLLM:
    """
    On-premise LLM with continuous learning capabilities.

    Uses Ollama for local inference with persistent memory to learn
    from your data quality domain over time.
    """

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        """
        Initialize local LLM.

        Args:
            config: LLM configuration. If None, uses default settings.
        """
        self.config = config or LocalLLMConfig()
        self.memory = ContextMemory(self.config.memory_db_path)
        self.client = self._initialize_ollama()

    def _initialize_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            # Test connection
            ollama.list()
            return ollama
        except ImportError:
            raise ImportError(
                "Ollama not installed. Install with: pip install ollama\n"
                "Also install Ollama server from: https://ollama.ai"
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama. Ensure it's running: ollama serve\n"
                f"Error: {e}"
            )

    def _build_prompt_with_context(self, prompt: str, include_memory: bool = True) -> str:
        """Enhance prompt with learned domain context"""
        if not include_memory:
            return prompt

        # Add domain-specific context
        domain_context = self.memory.get_domain_context()

        if domain_context:
            prompt = domain_context + "\n\n" + prompt

        return prompt

    def generate(
        self,
        prompt: str,
        include_memory: bool = True,
        stream: bool = False
    ) -> str:
        """
        Generate response using local LLM.

        Args:
            prompt: Input prompt
            include_memory: Include learned domain context
            stream: Stream response (useful for long outputs)

        Returns:
            Generated text response
        """
        enhanced_prompt = self._build_prompt_with_context(prompt, include_memory)

        response = self.client.chat(
            model=self.config.model_name,
            messages=[{"role": "user", "content": enhanced_prompt}],
            options={
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty
            },
            stream=stream
        )

        if stream:
            return response  # Return iterator for streaming
        else:
            return response['message']['content']

    def explain_validation_issue(
        self,
        column: str,
        value: Any,
        issue: str,
        detail: str,
        expected: Optional[str] = None,
        learn: bool = True
    ) -> str:
        """
        Explain a validation issue in natural language.

        Args:
            column: Column name
            value: Problematic value
            issue: Issue type
            detail: Detailed description
            expected: Expected value/format
            learn: Store this explanation for future learning

        Returns:
            Natural language explanation
        """
        # Check if we have similar explanations stored
        similar = self.memory.get_similar_explanations(issue)
        similar_context = ""
        if similar:
            similar_context = "\n\nPreviously successful explanations for similar issues:\n"
            for i, exp in enumerate(similar, 1):
                similar_context += f"{i}. {exp}\n"

        prompt = f"""You are a data quality expert for this organization. Explain this data quality issue clearly and concisely.

Column: {column}
Value: {value}
Issue Type: {issue}
Details: {detail}
Expected: {expected if expected else 'N/A'}
{similar_context}

Provide a clear 2-3 sentence explanation:
1. What's wrong
2. Why it matters
3. How to fix it

Keep it simple and actionable."""

        explanation = self.generate(prompt, include_memory=True)

        # Store for future learning
        if learn:
            self.memory.store_explanation(issue, explanation, column)

        return explanation

    def contextual_match(
        self,
        text1: str,
        text2: str,
        domain: Optional[str] = None,
        threshold: float = 0.7,
        learn: bool = True
    ) -> Tuple[bool, float, str]:
        """
        Determine if two texts are semantically similar with domain context.

        Args:
            text1: First text
            text2: Second text
            domain: Domain context (e.g., "companies", "addresses")
            threshold: Similarity threshold
            learn: Store this matching rule for future learning

        Returns:
            (is_match, confidence, reasoning)
        """
        domain_note = f"\nDomain: {domain}" if domain else ""

        prompt = f"""Compare these two values and determine if they represent the same entity.{domain_note}

Value 1: "{text1}"
Value 2: "{text2}"

Consider abbreviations, synonyms, formatting differences, and common variations.

Respond in JSON format:
{{
    "similar": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = self.generate(prompt, include_memory=True)

        # Parse JSON response
        try:
            # Extract JSON from markdown if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            is_match = result.get('similar', False) and result.get('confidence', 0) >= threshold
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')

            # Store for learning
            if learn:
                self.memory.store_matching_rule(
                    text1, text2, is_match, reasoning, domain
                )

            return is_match, confidence, reasoning

        except json.JSONDecodeError:
            # Fallback
            return False, 0.0, "Failed to parse LLM response"

    def suggest_validation_rules(
        self,
        column_name: str,
        sample_values: List[Any],
        column_stats: Dict[str, Any]
    ) -> List[Dict]:
        """
        Suggest validation rules based on data patterns.

        Args:
            column_name: Name of column
            sample_values: Sample values
            column_stats: Statistics about the column

        Returns:
            List of suggested validation rules
        """
        # Check if we've seen similar patterns before
        patterns = self.memory.get_patterns(pattern_type="validation_rule")
        pattern_context = ""
        if patterns:
            pattern_context = "\n\nPreviously learned patterns:\n"
            for p in patterns[:5]:
                pattern_context += f"- {p['pattern']}: {p['explanation']}\n"

        prompt = f"""Analyze this column and suggest validation rules.

Column: {column_name}
Sample Values: {sample_values[:20]}
Statistics: {json.dumps(column_stats, indent=2)}
{pattern_context}

Suggest 3-5 validation rules as JSON array:
[
    {{
        "rule_type": "type",
        "description": "what to validate",
        "reasoning": "why this rule"
    }}
]

Return only valid JSON."""

        response = self.generate(prompt, include_memory=True)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            rules = json.loads(response)

            # Store learned patterns
            for rule in rules:
                self.memory.store_pattern(
                    pattern_type="validation_rule",
                    pattern=rule.get('description', ''),
                    column_name=column_name,
                    explanation=rule.get('reasoning', '')
                )

            return rules if isinstance(rules, list) else [rules]

        except json.JSONDecodeError:
            return [{"rule_type": "manual_review", "description": response}]

    def teach_domain_term(self, term: str, meaning: str, category: str = "general"):
        """
        Teach the LLM about domain-specific terminology.

        This allows the LLM to understand your company's specific jargon,
        abbreviations, and terminology.

        Example:
            llm.teach_domain_term("BACS", "Bankers' Automated Clearing Services", "finance")
            llm.teach_domain_term("NHS", "National Health Service", "healthcare")
        """
        self.memory.add_domain_term(term, meaning, category)

    def get_knowledge_stats(self) -> Dict:
        """Get statistics about what the LLM has learned"""
        return self.memory.get_stats()


# Convenience functions
def create_local_llm(
    model_name: str = "mistral:7b-instruct",
    memory_path: str = "data/llm_memory.db"
) -> LocalLLM:
    """
    Create a local LLM instance with default configuration.

    Args:
        model_name: Ollama model to use
        memory_path: Path to memory database

    Returns:
        Configured LocalLLM instance
    """
    config = LocalLLMConfig(
        model_name=model_name,
        memory_db_path=memory_path
    )
    return LocalLLM(config)
