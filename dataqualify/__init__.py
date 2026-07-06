"""
DataQualify — explainable, weakly-supervised data quality for tabular data.

Public API (stable surface for library users)::

    import dataqualify as dq
    import pandas as pd

    df = pd.read_csv("customers.csv")

    # One-liner: infer rules from the data, validate, return an issue report.
    issues = dq.validate(df)

    # With a clean reference dataset (recommended — infers rules from clean data):
    issues = dq.validate(df, reference=pd.read_csv("clean_reference.csv"))

    # Full pipeline: scores across six dimensions, corrected copy, audit trail.
    result = dq.run_pipeline(df, dq.load_config())

The heavy ML / NLP / cloud dependencies are optional extras (see pyproject.toml);
the core validation surface needs only pandas, numpy and scikit-learn.
"""
from __future__ import annotations

from importlib import metadata as _metadata
from typing import Any, Dict, Optional

import pandas as pd

try:
    __version__ = _metadata.version("dataqualify")
except Exception:  # not installed as a distribution (e.g. running from source)
    __version__ = "0.1.0"

# Re-export the framework-agnostic engine entry points.
from core.engine import run_pipeline, load_config, load_data  # noqa: E402
from core.validator.discover import infer_rules_from_unclean as infer_rules  # noqa: E402

# Source connectors + headless tracked batch runner (the autonomous spine).
from .sources import parse_source, SourceConnector  # noqa: E402
from .batch import run_batch, SQLiteBatchStore, BatchRecord  # noqa: E402


def validate(
    df: pd.DataFrame,
    rules: Optional[Dict[str, Any]] = None,
    reference: Optional[pd.DataFrame] = None,
    typo_detection: bool = True,
) -> pd.DataFrame:
    """Validate a DataFrame and return a cell-level issue report.

    Args:
        df: the data to validate.
        rules: explicit validation rules. If omitted, rules are inferred from
            ``reference`` when given, otherwise from ``df`` itself.
        reference: a clean reference dataset to infer rules from (recommended:
            keeps inferred bounds/allowed-values free of the errors in ``df``).
        typo_detection: enable the Levenshtein typo layer.

    Returns:
        A pandas DataFrame with one row per issue: ``row_id``, ``column``,
        ``issue``, ``detail``, ``severity``, ``value``, ``expected``, ``rule``,
        and a human-readable ``description``.
    """
    from core.validator.discover import infer_rules_from_unclean
    from core.validator.validate import validate_df, attach_issue_descriptions

    if rules is None:
        src = reference if reference is not None else df
        rules = infer_rules_from_unclean(src)

    report = validate_df(df, rules, enable_typo_detection=typo_detection)
    return attach_issue_descriptions(report)


__all__ = [
    "validate",
    "run_pipeline",
    "infer_rules",
    "load_config",
    "load_data",
    "parse_source",
    "SourceConnector",
    "run_batch",
    "SQLiteBatchStore",
    "BatchRecord",
    "__version__",
]
