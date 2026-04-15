"""
BERT-Based Data Quality Explainer with Domain Adaptation

This module uses BERT (Bidirectional Encoder Representations from Transformers)
for data quality explanations and contextual understanding.

Advantages over generative LLMs:
- Smaller model size (~400MB vs 4GB+)
- Faster inference (10-100x faster)
- Better for classification and similarity tasks
- Easy to fine-tune on your domain data
- Completely offline, no external API calls
- Lower compute requirements

Uses:
- Sentence-BERT for semantic similarity
- BERT embeddings for context understanding
- Fine-tuning on your data quality patterns
- Template-based explanations enhanced with BERT context
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sqlite3


class BERTConfig:
    """Configuration for BERT-based explainer"""
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # Fast, efficient sentence-BERT
        similarity_threshold: float = 0.75,
        cache_embeddings: bool = True,
        memory_db_path: str = "data/bert_memory.db"
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.cache_embeddings = cache_embeddings
        self.memory_db_path = memory_db_path


class DomainKnowledgeBase:
    """
    Knowledge base that learns from your data quality domain.

    Stores:
    - Common error patterns and their explanations
    - Domain-specific terminology and abbreviations
    - Matching rules and examples
    - Fine-tuning examples for BERT
    """

    def __init__(self, db_path: str = "data/bert_memory.db"):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create database schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Explanation templates for different issue types
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS explanation_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_type TEXT NOT NULL,
                column_pattern TEXT,
                template TEXT NOT NULL,
                example_value TEXT,
                example_explanation TEXT,
                usefulness_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Domain terminology (abbreviations, jargon, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE NOT NULL,
                full_form TEXT,
                meaning TEXT,
                category TEXT,
                examples TEXT,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Semantic equivalence rules (for matching)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equivalence_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text1 TEXT NOT NULL,
                text2 TEXT NOT NULL,
                are_equivalent BOOLEAN NOT NULL,
                domain TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Error patterns learned from data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_name TEXT,
                error_type TEXT NOT NULL,
                pattern_regex TEXT,
                pattern_description TEXT,
                fix_suggestion TEXT,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def add_explanation_template(
        self,
        issue_type: str,
        template: str,
        column_pattern: Optional[str] = None,
        example_value: Optional[str] = None,
        example_explanation: Optional[str] = None
    ):
        """Add explanation template for an issue type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO explanation_templates
               (issue_type, column_pattern, template, example_value, example_explanation)
               VALUES (?, ?, ?, ?, ?)""",
            (issue_type, column_pattern, template, example_value, example_explanation)
        )

        conn.commit()
        conn.close()

    def get_explanation_template(self, issue_type: str) -> Optional[str]:
        """Get best explanation template for issue type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT template FROM explanation_templates
               WHERE issue_type=?
               ORDER BY usefulness_score DESC, created_at DESC
               LIMIT 1""",
            (issue_type,)
        )

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def add_domain_term(
        self,
        term: str,
        full_form: Optional[str] = None,
        meaning: Optional[str] = None,
        category: str = "general"
    ):
        """Add domain-specific terminology"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT OR REPLACE INTO domain_terms
               (term, full_form, meaning, category)
               VALUES (?, ?, ?, ?)""",
            (term, full_form, meaning, category)
        )

        conn.commit()
        conn.close()

    def get_domain_terms(self, category: Optional[str] = None) -> List[Dict]:
        """Get domain terminology"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category:
            cursor.execute(
                """SELECT term, full_form, meaning, category
                   FROM domain_terms WHERE category=?
                   ORDER BY frequency DESC""",
                (category,)
            )
        else:
            cursor.execute(
                """SELECT term, full_form, meaning, category
                   FROM domain_terms
                   ORDER BY frequency DESC"""
            )

        results = cursor.fetchall()
        conn.close()

        return [
            {
                'term': r[0],
                'full_form': r[1],
                'meaning': r[2],
                'category': r[3]
            }
            for r in results
        ]

    def add_equivalence_rule(
        self,
        text1: str,
        text2: str,
        are_equivalent: bool,
        domain: Optional[str] = None,
        confidence: float = 1.0
    ):
        """Add semantic equivalence rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO equivalence_rules
               (text1, text2, are_equivalent, domain, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (text1, text2, are_equivalent, domain, confidence)
        )

        conn.commit()
        conn.close()

    def check_equivalence(self, text1: str, text2: str) -> Optional[Tuple[bool, float]]:
        """Check if equivalence rule exists for this pair"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check both orderings
        cursor.execute(
            """SELECT are_equivalent, confidence FROM equivalence_rules
               WHERE (text1=? AND text2=?) OR (text1=? AND text2=?)
               LIMIT 1""",
            (text1, text2, text2, text1)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return (bool(result[0]), result[1])
        return None

    def add_error_pattern(
        self,
        error_type: str,
        pattern_description: str,
        column_name: Optional[str] = None,
        fix_suggestion: Optional[str] = None
    ):
        """Add learned error pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO error_patterns
               (column_name, error_type, pattern_description, fix_suggestion)
               VALUES (?, ?, ?, ?)""",
            (column_name, error_type, pattern_description, fix_suggestion)
        )

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM explanation_templates")
        stats['explanation_templates'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM domain_terms")
        stats['domain_terms'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM equivalence_rules")
        stats['equivalence_rules'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM error_patterns")
        stats['error_patterns'] = cursor.fetchone()[0]

        conn.close()
        return stats


class BERTExplainer:
    """
    BERT-based data quality explainer with domain adaptation.

    Uses sentence-BERT for semantic understanding and template-based
    explanations that improve over time as the system learns your domain.
    """

    def __init__(self, config: Optional[BERTConfig] = None):
        """Initialize BERT explainer"""
        self.config = config or BERTConfig()
        self.knowledge_base = DomainKnowledgeBase(self.config.memory_db_path)
        self.model = self._load_model()
        self._initialize_default_templates()

    def _load_model(self):
        """Load sentence-BERT model"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.config.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def _initialize_default_templates(self):
        """Initialize default explanation templates if knowledge base is empty"""
        stats = self.knowledge_base.get_stats()

        if stats['explanation_templates'] == 0:
            # Add default templates
            templates = [
                ("format_error", "The value '{value}' in column '{column}' doesn't match the expected format. {detail}. To fix this, {fix_suggestion}."),
                ("missing_value", "The column '{column}' has a missing value at row {row}. This field appears to be {importance}. Please provide a valid value."),
                ("typo", "The value '{value}' in column '{column}' appears to have a typo. {detail}. Did you mean '{expected}'?"),
                ("whitespace", "The value '{value}' in column '{column}' has extra whitespace. {detail}. Remove leading/trailing spaces."),
                ("case_inconsistency", "The value '{value}' in column '{column}' has inconsistent capitalization. {detail}. Use consistent casing."),
                ("duplicate", "This record appears to be a duplicate. {detail}. Review and merge or remove as appropriate."),
                ("outlier", "The value '{value}' in column '{column}' is unusual compared to other values. {detail}. Verify this is correct."),
                ("invalid_reference", "The value '{value}' in column '{column}' doesn't match any valid reference. {detail}. Use a value from the allowed list."),
            ]

            for issue_type, template in templates:
                self.knowledge_base.add_explanation_template(issue_type, template)

            # Add common domain terms
            common_terms = [
                ("Ltd", "Limited", "Company suffix indicating limited liability", "company"),
                ("PLC", "Public Limited Company", "Publicly traded company designation", "company"),
                ("Inc", "Incorporated", "Corporation designation", "company"),
                ("Corp", "Corporation", "Corporation designation", "company"),
                ("St", "Street", "Address abbreviation", "address"),
                ("Ave", "Avenue", "Address abbreviation", "address"),
                ("Rd", "Road", "Address abbreviation", "address"),
                ("Dr", "Doctor or Drive", "Title or address abbreviation", "general"),
                ("UK", "United Kingdom", "Country code", "geography"),
                ("USA", "United States of America", "Country code", "geography"),
            ]

            for term, full, meaning, category in common_terms:
                self.knowledge_base.add_domain_term(term, full, meaning, category)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to BERT embeddings"""
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using BERT embeddings"""
        embeddings = self.encode([text1, text2])
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def explain_validation_issue(
        self,
        column: str,
        value: Any,
        issue: str,
        detail: str,
        expected: Optional[str] = None,
        row: Optional[int] = None
    ) -> str:
        """
        Generate natural language explanation for validation issue.

        Args:
            column: Column name
            value: Problematic value
            issue: Issue type
            detail: Detailed description
            expected: Expected value/format
            row: Row number

        Returns:
            Natural language explanation
        """
        # Get template for this issue type
        template = self.knowledge_base.get_explanation_template(issue)

        if not template:
            # Fallback to generic template
            template = "Issue in column '{column}': {detail}. Value: '{value}'."

        # Determine fix suggestion based on issue type
        fix_suggestions = {
            "format_error": "reformat the value to match the expected pattern",
            "missing_value": "provide a valid value for this field",
            "typo": f"correct the spelling to '{expected}'" if expected else "correct the spelling",
            "whitespace": "trim leading and trailing spaces",
            "case_inconsistency": "use consistent capitalization throughout",
            "duplicate": "review and either merge or remove the duplicate",
            "outlier": "verify the value is correct or update it",
            "invalid_reference": "use a value from the allowed reference list"
        }

        fix_suggestion = fix_suggestions.get(issue, "review and correct the value")

        # Determine importance
        importance = "important" if issue in ["missing_value", "format_error"] else "optional"

        # Fill template
        explanation = template.format(
            column=column,
            value=value,
            detail=detail,
            expected=expected or "correct format",
            fix_suggestion=fix_suggestion,
            row=row or "unknown",
            importance=importance
        )

        return explanation

    def contextual_match(
        self,
        text1: str,
        text2: str,
        domain: Optional[str] = None,
        threshold: Optional[float] = None,
        learn: bool = True
    ) -> Tuple[bool, float, str]:
        """
        Determine if two texts are semantically equivalent using BERT.

        Args:
            text1: First text
            text2: Second text
            domain: Optional domain context
            threshold: Similarity threshold (uses config default if None)
            learn: Store result for future learning

        Returns:
            (is_match, confidence, reasoning)
        """
        threshold = threshold or self.config.similarity_threshold

        # Check if we have a learned rule for this pair
        cached = self.knowledge_base.check_equivalence(text1, text2)
        if cached:
            are_equivalent, confidence = cached
            reasoning = f"Learned rule: these {'match' if are_equivalent else 'do not match'} (confidence: {confidence:.2%})"
            return are_equivalent, confidence, reasoning

        # Compute BERT similarity
        sim_score = self.similarity(text1, text2)

        # Check for exact abbreviation matches in knowledge base
        terms = self.knowledge_base.get_domain_terms(category=domain)
        for term_info in terms:
            term = term_info['term']
            full = term_info['full_form']
            if full and ((text1.upper() == term.upper() and text2.upper() == full.upper()) or
                        (text2.upper() == term.upper() and text1.upper() == full.upper())):
                reasoning = f"'{term}' is the abbreviation for '{full}'"
                if learn:
                    self.knowledge_base.add_equivalence_rule(text1, text2, True, domain, 1.0)
                return True, 1.0, reasoning

        # Determine match based on similarity
        is_match = sim_score >= threshold

        # Generate reasoning
        if sim_score > 0.9:
            reasoning = f"Very high semantic similarity ({sim_score:.2%}) - likely the same"
        elif sim_score > threshold:
            reasoning = f"High semantic similarity ({sim_score:.2%}) - probably the same"
        elif sim_score > 0.5:
            reasoning = f"Moderate similarity ({sim_score:.2%}) - possibly related but different"
        else:
            reasoning = f"Low similarity ({sim_score:.2%}) - likely different"

        # Store for learning
        if learn:
            self.knowledge_base.add_equivalence_rule(text1, text2, is_match, domain, sim_score)

        return is_match, sim_score, reasoning

    def batch_match(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Find most similar candidates to query using BERT.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top matches to return

        Returns:
            List of (candidate, score, reasoning) tuples
        """
        # Encode all texts
        query_embedding = self.encode([query])[0]
        candidate_embeddings = self.encode(candidates)

        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidates):
            sim = np.dot(query_embedding, candidate_embeddings[i]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embeddings[i])
            )
            similarities.append((candidate, float(sim)))

        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]

        # Generate reasoning
        results = []
        for candidate, score in top_matches:
            if score > 0.9:
                reasoning = "Very high similarity - likely the same entity"
            elif score > 0.7:
                reasoning = "High similarity - probably related"
            elif score > 0.5:
                reasoning = "Moderate similarity - possibly related"
            else:
                reasoning = "Low similarity - likely different"

            results.append((candidate, score, reasoning))

        return results

    def teach_term(
        self,
        term: str,
        full_form: Optional[str] = None,
        meaning: Optional[str] = None,
        category: str = "general"
    ):
        """
        Teach the system about domain-specific terminology.

        Example:
            explainer.teach_term("BACS", "Bankers' Automated Clearing Services", category="finance")
            explainer.teach_term("NHS", "National Health Service", category="healthcare")
        """
        self.knowledge_base.add_domain_term(term, full_form, meaning, category)

    def teach_equivalence(
        self,
        text1: str,
        text2: str,
        are_equivalent: bool = True,
        domain: Optional[str] = None
    ):
        """
        Teach the system that two texts are (or are not) equivalent.

        Example:
            explainer.teach_equivalence("NYC", "New York City", True, "addresses")
            explainer.teach_equivalence("Apple Inc", "Microsoft Corp", False, "companies")
        """
        self.knowledge_base.add_equivalence_rule(text1, text2, are_equivalent, domain, 1.0)

    def get_knowledge_stats(self) -> Dict:
        """Get statistics about learned domain knowledge"""
        return self.knowledge_base.get_stats()


# Convenience function
def create_bert_explainer(memory_path: str = "data/bert_memory.db") -> BERTExplainer:
    """Create BERT explainer with default configuration"""
    config = BERTConfig(memory_db_path=memory_path)
    return BERTExplainer(config)
