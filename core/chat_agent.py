"""
Data Quality Chat Agent
=======================
Conversational interface backed by a local Ollama model with tool calling.

The LLM decides which tool to call based on the user's message.
Tools available:
  - get_summary        Summary of dataset and validation results
  - count_issues       Count issues (all or by type)
  - get_worst_rows     Rows with the most issues
  - explain_row        All issues for a specific row
  - get_column_issues  Issues in a specific column
  - get_quality_score  Data quality dimension scores
  - get_fixable_issues Auto-fixable issue breakdown
  - show_data          Show the first N rows of the dataset
  - open_tool          Open a tool panel in the app

Entry point: answer(user_message, state, history, model)
Returns: {"response": str, "open_tool": str | None}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS = 8  # Max tool-chaining loops per query

def _truncate_result(text: str, max_chars: int = 4000) -> str:
    """Truncate tool result for LLM context (full result kept for user)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text):,} chars total)"

# ─────────────────────────────────────────────────────────
# Tool Schemas (Ollama tool calling API)
# ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "Get a comprehensive summary of the dataset and data quality "
                "analysis results."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_issues",
            "description": (
                "Count data quality issues, optionally filtered by type "
                "(missing, typo, format, anomaly, duplicate, canonical). "
                "Use 'all' for the total count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "description": (
                            "Type of issue: all, missing, typo, format, "
                            "anomaly, duplicate, canonical, error, warning"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_worst_rows",
            "description": "Get the rows with the most data quality issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of worst rows to return (default 10, max 50)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_row",
            "description": (
                "Explain all data quality issues found in a specific row by its row number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row_id": {
                        "type": "integer",
                        "description": "The row number to explain (0-based index)",
                    }
                },
                "required": ["row_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_issues",
            "description": "Get data quality issues for a specific column in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to check",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quality_score",
            "description": (
                "Get the overall data quality scores across all dimensions "
                "(completeness, validity, consistency, uniqueness, accuracy, timeliness)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fixable_issues",
            "description": "Get a breakdown of data quality issues that can be automatically fixed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_data",
            "description": (
                "Show the FIRST N rows of the loaded dataset. "
                "Use this only when the user wants a general preview like 'show first 10 rows' or 'preview data'. "
                "Do NOT use this when the user asks for a specific row like 'show row 5' or 'the 5th row' — use get_row instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to show (default 10, max 100)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_value",
            "description": (
                "Look up a specific field value for a named record in the dataset. "
                "Use when the user asks: 'what is the phone number of Fatima Taylor', "
                "'what is John's email', 'find the address of Smith Ltd', "
                "'what is X's Y'. Searches the dataset and returns the requested field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_value": {
                        "type": "string",
                        "description": "The name or identifier to look up (e.g., 'Fatima Taylor', 'CUST001')",
                    },
                    "return_field": {
                        "type": "string",
                        "description": "The field/column to return (e.g., 'phone', 'email', 'address'). Empty = return all fields.",
                    },
                    "search_column": {
                        "type": "string",
                        "description": "Column to search in. Empty = search all text columns.",
                    },
                },
                "required": ["search_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": (
                "Filter and show rows matching a condition. "
                "Use for: 'show rows with missing values', 'show rows where email is invalid', "
                "'find rows containing X', 'show rows where column = value'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "One of: missing, not_missing, invalid, contains, equals",
                    },
                    "column": {
                        "type": "string",
                        "description": "Column name to filter on (optional — omit for all columns)",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to search for (used with 'contains' or 'equals')",
                    },
                },
                "required": ["condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_row",
            "description": (
                "Show the raw data values for a specific single row, identified by its row number. "
                "Use this when the user asks for a specific row like 'show me the 5th row', 'row 3', 'row number 10'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row_number": {
                        "type": "integer",
                        "description": "The 1-based row number to show (e.g. 5 for 'the 5th row')",
                    }
                },
                "required": ["row_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_duplicate_detection",
            "description": (
                "Actually run duplicate detection on the loaded dataset right now — "
                "no need to open the Dedupe panel. "
                "Finds similar/duplicate rows using TF-IDF similarity. "
                "Use this when the user asks to find, detect, or show duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duplicate_threshold": {
                        "type": "number",
                        "description": "Similarity score above which two rows are considered duplicates (default 0.85, range 0.5-1.0)",
                    },
                    "similar_threshold": {
                        "type": "number",
                        "description": "Similarity score above which two rows are considered similar (default 0.6)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_validation",
            "description": (
                "Actually run the full data validation pipeline on the loaded dataset right now — "
                "no need to open the Validate & Fix panel. "
                "Checks for missing values, typos, format errors, anomalies, and more. "
                "Set use_ml=true for ML anomaly detection (Isolation Forest, LOF, SVM). "
                "Use this when the user asks to validate, check, or analyse data quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "use_ml": {
                        "type": "boolean",
                        "description": (
                            "Enable ML-based anomaly detection (Isolation Forest, LOF, One-Class SVM). "
                            "Default false. Set true when user asks for anomaly detection, ML analysis, "
                            "or deeper/advanced validation."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_duplicates",
            "description": (
                "Get duplicate detection results. "
                "Returns row-level duplicate pairs/clusters if deduplication has been run, "
                "or column-level duplicates. "
                "If duplicate detection has not been run yet, tells the user to run it first."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_tool",
            "description": (
                "Open a specific tool panel in the application for the user "
                "(e.g., to load data, run validation, view profiles)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": (
                            "Name of the tool panel to open. Options: Load, "
                            "Validate & Fix, Data Profiling, Rules, "
                            "Corpus Manager, Pipeline Manager, "
                            "Vector Index Manager, Cleaning & EDA Trim, "
                            "Dedupe, Auto Report, Data Quality Scorecard, "
                            "Entity Resolution, Entity Graph"
                        ),
                    }
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_data_profiling",
            "description": (
                "Run a full data profile on the loaded dataset. "
                "Generates column statistics, distributions, outlier detection, "
                "correlations, and pattern analysis. "
                "Use when user asks to profile, analyse, or explore the data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_rules",
            "description": (
                "Auto-generate data quality rules by inferring them from the loaded dataset. "
                "Detects column types, formats, value ranges, and patterns automatically. "
                "Use when user asks to generate rules, infer rules, or set up validation rules."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fixes",
            "description": (
                "Apply suggested fixes from the validation report to the dataset. "
                "Creates a corrected version of the dataset. "
                "Use when user asks to apply fixes, correct errors, fix issues, or clean the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": (
                            "Which fixes to apply: "
                            "'all' to apply every suggested fix, "
                            "'high_confidence' to apply only fixes with confidence >= 0.9 (default)."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_basic_cleaning",
            "description": (
                "Run basic text cleaning on the dataset: "
                "strip whitespace, normalize punctuation, lowercase text. "
                "Use when user asks to clean, trim, normalize, or tidy the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lowercase": {
                        "type": "boolean",
                        "description": "Convert text to lowercase (default: true)",
                    },
                    "strip_whitespace": {
                        "type": "boolean",
                        "description": "Strip leading/trailing whitespace (default: true)",
                    },
                    "normalize_punctuation": {
                        "type": "boolean",
                        "description": "Normalize punctuation (default: true)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_column_trimming",
            "description": (
                "Remove low-information, high-missing, and highly-correlated columns from the dataset. "
                "Produces a trimmed dataset with only the most useful columns. "
                "Use when user asks to trim columns, remove useless columns, or reduce features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "missing_threshold": {
                        "type": "number",
                        "description": "Drop columns with more than this fraction of missing values (default: 0.8)",
                    },
                    "correlation_threshold": {
                        "type": "number",
                        "description": "Drop one of any two columns correlated above this threshold (default: 0.9)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_duplicates",
            "description": (
                "Merge/remove duplicate records found by duplicate detection. "
                "Keeps one master record per cluster and removes the rest. "
                "Requires duplicate detection to have been run first. "
                "Use when user asks to merge duplicates, remove duplicates, deduplicate, or apply deduplication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Merge strategy: "
                            "'keep_master' keeps the first record in each cluster (default), "
                            "'fill_nulls' fills missing values in master from other records, "
                            "'most_common' uses the most common value per field."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_auto_report",
            "description": (
                "Run the full end-to-end data quality pipeline automatically: "
                "validate → clean → deduplicate → score. "
                "Produces a complete quality report and cleaned dataset in one go. "
                "Use when user asks for a full report, complete analysis, or one-click quality check."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pipeline",
            "description": (
                "Execute a saved data quality pipeline by name. "
                "Lists available pipelines if no name is given. "
                "Use when user asks to run a pipeline, execute a workflow, or apply a saved pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_name": {
                        "type": "string",
                        "description": "Name of the saved pipeline to execute. Leave empty to list available pipelines.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_entity_resolution",
            "description": (
                "Find matching entities between the loaded dataset and the reference dataset. "
                "Requires both a main dataset and a reference dataset to be loaded. "
                "Use when user asks to match records, find entities, resolve entities, or cross-reference datasets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score to consider a match (default: 0.8, range 0.5-1.0)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": (
                "Query profiling results for a specific column or the whole dataset. "
                "Returns null rate, unique values, data type, min/max/mean for numeric columns, "
                "and detected patterns (email, phone, postcode, etc.). "
                "Use ONLY when user asks about null rate, missing rate, unique count, "
                "data type, distributions, or column statistics. "
                "Do NOT use for frequency counts, most common values, or specific value lookups — use query_data for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Column name to profile (e.g. 'email', 'company_name'). "
                            "Leave empty to show a summary of all columns."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_next_steps",
            "description": (
                "Analyse the current pipeline state and recommend the most logical next steps "
                "in the data quality workflow. "
                "Use when user asks what to do next, what's the next step, where to start, "
                "what should I do, what's left, or how to improve the data."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_corpus_standardization",
            "description": (
                "Apply corpus alias replacements to standardize non-canonical values in the dataset. "
                "Replaces variant spellings, abbreviations, and aliases with their canonical forms "
                "using the loaded alias corpora. Produces a standardized version of the dataset. "
                "Use after check_corpus has identified issues, when user asks to standardize values, "
                "apply canonical forms, fix non-standard values, or apply corpus corrections."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_versions",
            "description": (
                "Compare the original (raw) dataset against the current cleaned/processed version to show "
                "how much data quality has improved. Shows before/after quality scores, row counts, issue counts, "
                "and what changed. Use when user asks how much cleaning helped, before/after comparison, "
                "improvement summary, or quality delta."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Answer any ad-hoc question about the dataset by evaluating a pandas expression. "
                "Use for: specific value lookups, frequency counts, filters, averages, max/min, "
                "group-by aggregations, or any question that requires computing from actual data rows. "
                "Examples: df['customer_name'].value_counts().head(5) to find most frequent customers, "
                "df[df['age'] > 50] to filter rows, df['amount'].mean() for averages, "
                "df['city'].nunique() for unique count, df.groupby('city')['revenue'].sum() for group totals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A pandas expression using 'df' as the dataframe variable. "
                            "Use df['col'].value_counts() for frequencies, "
                            "df[df['col'] == 'val'] for filtering, "
                            "df['col'].mean() / .sum() / .max() / .min() for aggregations."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_corpora",
            "description": (
                "List all corpora currently loaded in Redis, including their type, record count, and which "
                "columns they are likely to validate. Use when user asks what corpora are loaded, what "
                "reference data is available, or what canonical datasets exist."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rules",
            "description": (
                "List the active validation rules, optionally for a specific column. "
                "Use when user asks what rules are set up, what validation rules exist, "
                "or what rules apply to a specific column or field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Filter rules to a specific column name. Leave empty to list all rules.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_corpus",
            "description": (
                "Run corpus/canonical value validation against the dataset using the Redis-backed "
                "reference corpora (canonical names, valid postcodes, valid email domains, company names, etc.). "
                "Flags non-canonical values and suggests the correct canonical replacement. "
                "Use when user asks to check canonical values, check corpus, validate against reference data, "
                "standardize values, or find non-standard entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": (
                            "Specific column to check against corpus (e.g. 'email', 'company_name'). "
                            "Leave empty to auto-detect and check all mappable columns."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_cell",
            "description": (
                "Edit one or more cell values in the dataset. Use when the user asks to change, "
                "update, fix, correct, or set a specific value in a specific row and column. "
                "Examples: 'change row 5 email to john@example.com', 'fix the phone number in row 12', "
                "'set customer_name in row 3 to Fatima Ali', 'update rows 1-5 city to London'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": (
                            "1-based row number to edit. For multiple rows, this is the start row."
                        ),
                    },
                    "end_row": {
                        "type": "integer",
                        "description": (
                            "Optional 1-based end row for range edits (e.g. rows 3-7). "
                            "If omitted, only the single row specified by 'row' is edited."
                        ),
                    },
                    "column": {
                        "type": "string",
                        "description": "The column name to edit (e.g. 'email', 'customer_name', 'phone').",
                    },
                    "new_value": {
                        "type": "string",
                        "description": "The new value to set in the cell(s).",
                    },
                },
                "required": ["row", "column", "new_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_data",
            "description": (
                "Prepare the current dataset (raw or cleaned/corrected) for download. "
                "Returns the data as a CSV string that the UI will convert to a download button. "
                "Use when user asks to download, export, or save the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "description": (
                            "Which version to download: "
                            "'raw' for the original uploaded data, "
                            "'corrected' for the version after fixes have been applied, "
                            "'cleaned' for the version after basic cleaning. "
                            "Default: 'corrected' (falls back to 'raw' if no corrected version exists)."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    # ── NEW TOOLS ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_bert_validation",
            "description": (
                "Run BERT-enhanced validation that produces human-friendly explanations "
                "for data quality issues. Uses sentence-transformer embeddings. "
                "Use when user asks for AI explanations, BERT analysis, or wants to "
                "understand WHY an issue was flagged."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "explain_sample": {
                        "type": "integer",
                        "description": "Number of issues to explain (default 10). Higher = slower.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_vector_index",
            "description": (
                "Build a vector index (FAISS/ChromaDB) from a column for semantic similarity search. "
                "Use when user asks to build an index, create embeddings, set up semantic search, "
                "or prepare a column for similarity queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column to build the vector index from.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name for the vector collection (default: column name).",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vector_index",
            "description": (
                "Search a vector index for values semantically similar to a query. "
                "Use when user asks to find similar values, search by meaning, "
                "or do semantic similarity lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text to search for.",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the vector collection to search.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 10).",
                    },
                },
                "required": ["query", "collection_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_pipeline",
            "description": (
                "Save the current workflow steps as a named pipeline for later re-use. "
                "Use when user asks to save a pipeline, create a reusable workflow, "
                "or save the current steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the pipeline.",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of step names to include. Valid steps: "
                            "'validate', 'clean', 'deduplicate', 'corpus_standardize', "
                            "'column_trim', 'entity_resolution', 'score'. "
                            "Default: ['validate', 'clean', 'deduplicate', 'score']."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scorecard",
            "description": (
                "Get a detailed quality scorecard with per-column breakdown and "
                "before/after comparison if cleaning has been done. "
                "Use when user asks for scorecard, detailed quality breakdown, "
                "per-column scores, or quality comparison."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_rule",
            "description": (
                "Add, modify, or delete a validation rule for a specific column. "
                "Use when user asks to add a regex rule, change a rule threshold, "
                "update severity, delete a rule, or edit validation rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'add', 'modify', or 'delete'.",
                        "enum": ["add", "modify", "delete"],
                    },
                    "column": {
                        "type": "string",
                        "description": "Column name the rule applies to.",
                    },
                    "rule_type": {
                        "type": "string",
                        "description": "Type of rule: 'regex', 'range', 'type', 'required', 'categorical', 'uniqueness', 'length'.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (for regex rules) or value constraint.",
                    },
                    "severity": {
                        "type": "string",
                        "description": "Rule severity: 'high', 'medium', 'low'. Default 'medium'.",
                        "enum": ["high", "medium", "low"],
                    },
                    "min_value": {
                        "type": "number",
                        "description": "Minimum value (for range rules).",
                    },
                    "max_value": {
                        "type": "number",
                        "description": "Maximum value (for range rules).",
                    },
                },
                "required": ["action", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_fixes",
            "description": (
                "Show pending fix suggestions for review before applying. "
                "Returns fixes one by one or for a specific column so user can approve/reject each. "
                "Use when user asks to review fixes, see suggested corrections, approve fixes row by row, "
                "or wants to selectively apply corrections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Optional column to filter fixes for.",
                    },
                    "start_row": {
                        "type": "integer",
                        "description": "Starting row to show fixes from (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max fixes to show (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_corpus",
            "description": (
                "Map a dataset column to a specific corpus for validation or standardization. "
                "Use when user asks to map a column to a corpus, assign a corpus to a column, "
                "set up corpus validation for a column, or configure column-corpus mapping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Dataset column to map.",
                    },
                    "corpus_name": {
                        "type": "string",
                        "description": "Name of the loaded corpus to map to.",
                    },
                },
                "required": ["column", "corpus_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_validation",
            "description": (
                "Configure advanced validation settings: ML anomaly detection parameters, "
                "BERT model selection, API validation toggle. "
                "Use when user asks to adjust ML contamination rate, select ML models, "
                "choose BERT model, enable API validation, or configure validation parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ml_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ML models to use: 'isolation_forest', 'lof', 'one_class_svm'. Default all three.",
                    },
                    "contamination": {
                        "type": "number",
                        "description": "Expected fraction of anomalies (0.01-0.5). Default 0.05.",
                    },
                    "bert_model": {
                        "type": "string",
                        "description": "Sentence-transformer model name. Default 'all-MiniLM-L6-v2'.",
                    },
                    "use_api": {
                        "type": "boolean",
                        "description": "Enable external API validation. Default false.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_entity_graph",
            "description": (
                "Show entity resolution results as a text-based graph summary. "
                "Lists matched entity clusters, node connections, and match scores. "
                "Use when user asks about entity graph, entity matches, entity clusters, "
                "or connected entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_clusters": {
                        "type": "integer",
                        "description": "Maximum clusters to show (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Generate a data visualization chart. "
                "DATA QUALITY charts: issues_by_column, issues_by_type, issues_by_severity, "
                "missing_heatmap, completeness_by_column, quality_score_breakdown, "
                "issues_severity_by_column, top_issue_rows. "
                "DATA EXPLORATION charts: histogram, bar_chart, pie_chart, scatter, "
                "box_plot, correlation_matrix, time_series. "
                "Use when user asks for any chart, graph, plot, visualization, or visual breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "description": (
                            "Type of chart. Quality: issues_by_column, issues_by_type, "
                            "issues_by_severity, missing_heatmap, completeness_by_column, "
                            "quality_score_breakdown, issues_severity_by_column, top_issue_rows. "
                            "Exploration: histogram, bar_chart, pie_chart, scatter, "
                            "box_plot, correlation_matrix, time_series."
                        ),
                    },
                    "column": {
                        "type": "string",
                        "description": "Column name (for histogram, bar_chart, pie_chart, box_plot).",
                    },
                    "x": {
                        "type": "string",
                        "description": "X-axis column (for scatter plot).",
                    },
                    "y": {
                        "type": "string",
                        "description": "Y-axis column (for scatter plot).",
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Date column (for time_series).",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Value column to plot over time (for time_series).",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of items to show (default 10-15).",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of histogram bins (default 30).",
                    },
                },
                "required": ["chart_type"],
            },
        },
    },
    # ── Agentic workflow tools ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_batch_history",
            "description": (
                "Get historical batch quality reports from the database. "
                "Returns quality scores, issue counts, and timestamps for past validation runs. "
                "Use when user asks about trends, history, past batches, or wants to compare with previous runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "type": "integer",
                        "description": "Number of most recent batches to return (default 10, max 50).",
                    },
                    "column": {
                        "type": "string",
                        "description": "Optional: filter history for a specific column's issues.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_batches",
            "description": (
                "Compare two batch validation results side by side. "
                "Shows what changed between batches — score changes, new issues, resolved issues. "
                "Use when user asks 'what changed', 'compare with last batch', or 'what got worse'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_a": {
                        "type": "string",
                        "description": "First batch ID (e.g., 'batch_20260414_120000'). Use 'latest' for the most recent.",
                    },
                    "batch_b": {
                        "type": "string",
                        "description": "Second batch ID to compare against. Use 'previous' for the one before batch_a.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_results_to_s3",
            "description": (
                "Save the current validation results and corrected dataset to S3. "
                "Use when the user asks to save, export to S3, persist results, or after applying fixes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_corrected": {
                        "type": "boolean",
                        "description": "Whether to also save the corrected dataset (default true).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate",
            "description": (
                "Run a full autonomous investigation on the loaded dataset. "
                "This is the main agentic workflow — the AI will: "
                "1) Run validation, 2) Identify worst columns, 3) Profile them, "
                "4) Check batch history for trends, 5) Suggest and apply fixes, "
                "6) Report findings with a before/after comparison. "
                "Use when user says 'investigate', 'analyse everything', 'full check', "
                "'what's wrong with my data', or 'run the full pipeline'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "auto_fix": {
                        "type": "boolean",
                        "description": "Whether to automatically apply high-confidence fixes (default false — will ask for confirmation).",
                    },
                },
                "required": [],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# Auto-chain Helpers
# ─────────────────────────────────────────────────────────

def _ensure_validation(state: Dict[str, Any]) -> str:
    """
    If validation hasn't been run yet, run it silently and return a note.
    Returns a note string to prepend to the response, or "" if already done.
    """
    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty:
        return ""  # Already done
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""  # No data — let the handler return its own error
    try:
        from core.validator.validate import validate_and_anomaly_report
        rules = state.get("rules_merged") or {}
        report = validate_and_anomaly_report(df, rules=rules, use_ml=False)
        state["unified_issues_report"] = report
        return f"*Validation ran automatically ({len(report):,} issues found).*\n\n"
    except Exception as e:
        return f"*Could not auto-run validation: {e}*\n\n"


def _ensure_dedup(state: Dict[str, Any]) -> str:
    """
    If duplicate detection hasn't been run yet, run it silently and return a note.
    Returns a note string to prepend to the response, or "" if already done.
    """
    pairs = state.get("dedup_pairs")
    if isinstance(pairs, pd.DataFrame) and not pairs.empty:
        return ""  # Already done
    df = None
    for key in ("df_trimmed", "df_precleaned", "df_corrected", "df_raw"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            df = state[key]
            break
    if df is None or len(df) < 2:
        return ""
    if len(df) > 5000:
        return "*Dataset too large for auto dedup via chat (>5000 rows). Open the Dedupe panel.*\n\n"
    try:
        from core.vectorizers import build_vectorizers
        from core.matcher import compute_pair_scores, label_pairs
        from core.dedup_clustering import cluster_duplicates, create_cluster_summary
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            return ""
        vec_cfg = {"ngram_range": (1, 2), "max_features": 10000,
                   "word_range": (1, 2), "svd_components": 0}
        vec_pack = build_vectorizers(df, str_cols, vec_cfg)
        pairs_df  = compute_pair_scores(df, vec_pack, run_mode="Fast KNN")
        pairs_df  = label_pairs(pairs_df, similar_thr=0.6, duplicate_thr=0.85)
        clusters  = cluster_duplicates(pairs_df, id_col_a="row_i", id_col_b="row_j")
        cluster_summary = create_cluster_summary(clusters, df, pairs_df)
        state["dedup_pairs"]           = pairs_df
        state["dedup_clusters"]        = clusters
        state["dedup_cluster_summary"] = cluster_summary
        dup_count = int((pairs_df["label"] == "duplicate").sum()) if "label" in pairs_df.columns else len(pairs_df)
        return f"*Duplicate detection ran automatically ({dup_count:,} duplicate pairs found).*\n\n"
    except Exception as e:
        return f"*Could not auto-run duplicate detection: {e}*\n\n"


def _ensure_corpus_manager(state: Dict[str, Any]):
    """
    Get or lazily initialize the CorpusManager from state.
    Tries to connect to Redis; if not running, tries to start redis-server.
    Returns CorpusManager instance or None if unavailable.
    """
    cm = state.get("corpus_manager")
    if cm is not None:
        return cm

    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=2)
        r.ping()
    except Exception:
        # Try to start redis-server automatically
        try:
            import subprocess, time
            subprocess.Popen(
                ["redis-server", "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.5)
            import redis as redis_lib
            r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=2)
            r.ping()
        except Exception:
            return None  # Redis truly unavailable

    try:
        from core.corpus_manager import CorpusManager
        cm = CorpusManager(redis_client=r)
        state["corpus_manager"] = cm
        return cm
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────────────────

def build_context(state: Dict[str, Any], include_data: bool = False, n_rows: int = 20) -> str:
    """Build a concise summary of the current session state.

    Args:
        state: session state dict
        include_data: if True, append actual data rows to context
        n_rows: number of rows to include when include_data is True
    """
    lines = ["=== CURRENT SESSION CONTEXT ==="]

    df = state.get("df_raw")
    if isinstance(df, pd.DataFrame) and not df.empty:
        lines.append(f"Dataset loaded: {len(df):,} rows x {len(df.columns)} columns")
        lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    else:
        lines.append("No dataset loaded yet.")

    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty:
        total = len(report)
        rows_affected = report["row_id"].nunique() if "row_id" in report.columns else "unknown"
        lines.append(f"\nValidation report: {total:,} issues across {rows_affected} rows")

        if "issue" in report.columns:
            lines.append("Issue types:")
            for issue_type, count in report["issue"].value_counts().items():
                lines.append(f"  {issue_type}: {count}")

        if "severity" in report.columns:
            lines.append("Severity breakdown:")
            for sev, count in report["severity"].value_counts().items():
                lines.append(f"  {sev}: {count}")

        if "suggested_fix" in report.columns:
            fixable = report["suggested_fix"].notna().sum()
            lines.append(f"Auto-fixable issues: {fixable}")
    else:
        lines.append("\nNo validation report available — validation not yet run.")

    rules = state.get("rules_merged")
    if rules and isinstance(rules, dict):
        col_rules = rules.get("columns", {})
        lines.append(f"\nActive rules: {len(col_rules)} columns covered")

    # Dedupe status
    dedup_pairs = state.get("dedup_pairs")
    col_dups = state.get("column_duplicates")
    if isinstance(dedup_pairs, pd.DataFrame) and not dedup_pairs.empty:
        lines.append(f"\nDuplicate detection (row-based): DONE — {len(dedup_pairs):,} pairs found")
    elif isinstance(col_dups, pd.DataFrame) and not col_dups.empty:
        lines.append(f"\nDuplicate detection (column-based): DONE — {len(col_dups):,} column pairs")
    else:
        lines.append("\nDuplicate detection: NOT RUN — use get_duplicates tool if asked about duplicates")

    # Corpus status
    cm = state.get("corpus_manager")
    if cm is not None:
        try:
            corpora = cm.list_corpora()
            if corpora:
                names = ", ".join(c["corpus_name"] for c in corpora)
                lines.append(f"\nCorpus Manager: CONNECTED — {len(corpora)} corpus(es) loaded: {names}")
            else:
                lines.append("\nCorpus Manager: CONNECTED — no corpora loaded yet (use Corpus Manager panel to load)")
        except Exception:
            lines.append("\nCorpus Manager: CONNECTED (could not list corpora)")
    else:
        lines.append("\nCorpus Manager: NOT CONNECTED (Redis may not be running)")

    corpus_issues = state.get("corpus_issues_report")
    if isinstance(corpus_issues, pd.DataFrame) and not corpus_issues.empty:
        lines.append(f"Corpus issues found: {len(corpus_issues):,} (from last corpus check)")

    # ── Data context (toggle ON) ───────────────────────────────────
    if include_data:
        df = state.get("df_raw")
        if isinstance(df, pd.DataFrame) and not df.empty:
            n = min(n_rows, len(df))
            lines.append(f"\n=== ACTUAL DATA SAMPLE ({n} of {len(df):,} rows) ===")
            lines.append("Use this to answer specific questions about values, patterns, and content.")
            try:
                lines.append(df.head(n).to_markdown(index=True))
            except Exception:
                lines.append(df.head(n).to_string())

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Tool Handler Functions
# ─────────────────────────────────────────────────────────

def handle_run_validation(state: Dict[str, Any], use_ml: bool = False) -> str:
    """Run the validation pipeline and store results in state."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first (use the Load tool button)."

    try:
        from core.validator.validate import validate_and_anomaly_report
        rules = state.get("rules_merged") or {}
        df_ref = state.get("df_ref")

        # Auto-infer rules if none exist yet — so the user doesn't
        # need a separate "generate rules" step before validating.
        if not rules or not rules.get("columns"):
            try:
                from core.validator.discover import infer_rules_from_unclean
                rules = infer_rules_from_unclean(df)
                state["rules_merged"] = rules
            except Exception:
                rules = {}

        report = validate_and_anomaly_report(
            df,
            rules=rules,
            use_ml=use_ml,
            use_reference=(df_ref is not None),
            df_ref=df_ref,
        )

        # Attach fix suggestions (with corpus CSV support) so apply_fixes works
        try:
            from core.validator.validate import attach_suggestions
            report = attach_suggestions(report, df, df_ref, rules=rules)
        except Exception:
            pass  # suggestions are optional — validation still works without them

        # Attach human-readable issue descriptions (static, always runs)
        try:
            from core.validator.validate import attach_issue_descriptions
            report = attach_issue_descriptions(report)
        except Exception:
            pass

        # BERT explanations (optional, local model, toggled by user)
        _bert_count = 0
        if state.get("use_bert_explanations"):
            try:
                from core.enhanced_validator import EnhancedValidator
                ev = EnhancedValidator(use_bert=True)
                # Generate explanations for up to 50 issues (avoid slow runs on large reports)
                _max_explain = min(50, len(report))
                explanations = []
                for idx, row in report.iterrows():
                    if idx < _max_explain:
                        expl = ev.explainer.explain_validation_issue(
                            column=row.get("column", ""),
                            value=row.get("value"),
                            issue=row.get("issue", row.get("issue_type", "")),
                            detail=row.get("detail", ""),
                            expected=row.get("expected"),
                            row=row.get("row_id"),
                        )
                        explanations.append(expl)
                    else:
                        explanations.append(None)
                report["bert_explanation"] = explanations
                _bert_count = sum(1 for e in explanations if e)
            except Exception:
                pass  # BERT not available — continue without

        # ── Normalise column names so the tab UI can display results too ──
        _rename = {}
        if "issue" in report.columns and "issue_type" not in report.columns:
            _rename["issue"] = "issue_type"
        if "detail" in report.columns:
            _rename["detail"] = "detail_technical"
        if "rule" in report.columns and "source" not in report.columns:
            _rename["rule"] = "source"
        if _rename:
            report = report.rename(columns=_rename)

        # Add confidence column if missing (tab UI needs it)
        if "confidence" not in report.columns:
            _sev_conf = {"high": 0.90, "medium": 0.75, "low": 0.60}
            report["confidence"] = report["severity"].map(
                lambda s: _sev_conf.get(str(s).lower(), 0.70)
            )

        state["unified_issues_report"] = report
        state["validation_completed"] = True     # tab UI picks this up
        state["_chat_validation_done"] = True     # main panel visual trigger

        total = len(report)
        _issue_col = "issue_type" if "issue_type" in report.columns else "issue"
        rows_flagged = report["row_id"].nunique() if "row_id" in report.columns else "?"
        clean_rows = len(df) - (rows_flagged if isinstance(rows_flagged, int) else 0)

        lines = [
            f"Validation complete on {len(df):,} rows.",
            f"  Total issues found : {total:,}",
            f"  Rows flagged       : {rows_flagged}",
            f"  Clean rows         : {clean_rows:,}",
        ]

        if _issue_col in report.columns and total > 0:
            lines.append("\nTop issue types:")
            for itype, count in report[_issue_col].value_counts().head(5).items():
                lines.append(f"  {itype:<30} {count:,}")

        if "severity" in report.columns and total > 0:
            high = int((report["severity"] == "high").sum())
            med  = int((report["severity"] == "medium").sum())
            low  = int((report["severity"] == "low").sum())
            lines.append(f"\nSeverity:  High {high:,}  |  Medium {med:,}  |  Low {low:,}")

        if _bert_count > 0:
            lines.append(f"\nBERT explanations: {_bert_count} issues have contextual AI descriptions.")

        lines.append("\nResults saved. The colour-coded report is shown in the main panel.")
        lines.append("You can also switch to the Validate & Fix tab for the full view.")
        return "\n".join(lines)

    except Exception as e:
        return f"Validation error: {e}"


def handle_run_duplicate_detection(
    state: Dict[str, Any],
    duplicate_threshold: float = 0.85,
    similar_threshold: float = 0.6,
) -> str:
    """Run duplicate detection pipeline and store results in state."""
    # Pick best available dataframe
    df = None
    for key in ("df_trimmed", "df_cleaned", "df_clean", "df_validated", "df_raw"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            df = state[key]
            break

    if df is None:
        return "No dataset loaded. Please load data first (use the Load tool button)."

    if len(df) < 2:
        return "Dataset has fewer than 2 rows — nothing to deduplicate."

    if len(df) > 5000:
        return (
            f"Dataset has {len(df):,} rows — duplicate detection via chat is capped at 5,000 rows "
            f"to avoid crashes. Open the Dedupe panel for large datasets."
        )

    try:
        from core.vectorizers import build_vectorizers
        from core.matcher import compute_pair_scores, label_pairs
        from core.dedup_clustering import cluster_duplicates, create_cluster_summary

        # Use string/object columns for matching
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not str_cols:
            return "No text columns found in the dataset for duplicate matching."

        # Simple default vectorizer config
        vec_cfg = {
            "ngram_range": (1, 2),
            "max_features": 10000,
            "word_range": (1, 2),
            "svd_components": 0,
        }

        vec_pack = build_vectorizers(df, str_cols, vec_cfg)
        pairs_df  = compute_pair_scores(df, vec_pack, run_mode="Fast KNN")
        pairs_df  = label_pairs(pairs_df, similar_thr=similar_threshold, duplicate_thr=duplicate_threshold)

        clusters        = cluster_duplicates(pairs_df, id_col_a="row_i", id_col_b="row_j")
        cluster_summary = create_cluster_summary(clusters, df, pairs_df)

        # Store results so Dedupe panel and future AI queries can use them
        state["dedup_pairs"]           = pairs_df
        state["dedup_clusters"]        = clusters
        state["dedup_cluster_summary"] = cluster_summary
        state["dedup_config"] = {
            "used_cols": str_cols,
            "similar_threshold": similar_threshold,
            "duplicate_threshold": duplicate_threshold,
        }

        dup_count = int((pairs_df["label"] == "duplicate").sum()) if "label" in pairs_df.columns else len(pairs_df)
        sim_count = int((pairs_df["label"] == "similar").sum())  if "label" in pairs_df.columns else 0

        lines = [
            f"Duplicate detection complete on {len(df):,} rows.",
            f"  Columns checked     : {', '.join(str_cols[:5])}{'...' if len(str_cols) > 5 else ''}",
            f"  Duplicate threshold : {duplicate_threshold}",
            f"  Duplicate pairs     : {dup_count:,}",
            f"  Similar pairs       : {sim_count:,}",
            f"  Clusters found      : {len(clusters):,}",
        ]

        if cluster_summary is not None and not cluster_summary.empty and len(cluster_summary) > 0:
            lines.append(f"\nTop duplicate clusters:")
            for i, (_, row) in enumerate(cluster_summary.head(5).iterrows()):
                size = row.get("cluster_size", row.get("size", "?"))
                lines.append(f"  Cluster {i+1}: {size} records")

        lines.append("\nResults saved — open the Dedupe panel to review and merge them.")
        return "\n".join(lines)

    except Exception as e:
        return f"Duplicate detection error: {e}"


def handle_duplicates(state: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Return duplicate detection results if available.
    Auto-runs dedup if not yet done, then returns results.
    Returns (result_text, open_tool_name_or_None)
    """
    auto_note = _ensure_dedup(state)
    lines = [auto_note] if auto_note else []
    found_anything = False

    # ── Row-based duplicates (dedup_pairs / dedup_clusters) ────────
    pairs = state.get("dedup_pairs")
    clusters = state.get("dedup_clusters")
    cluster_summary = state.get("dedup_cluster_summary")

    if isinstance(pairs, pd.DataFrame) and not pairs.empty:
        found_anything = True
        lines.append(f"Row-based duplicate pairs found: {len(pairs):,}")
        if "score" in pairs.columns:
            avg_score = pairs["score"].mean()
            lines.append(f"  Average similarity score: {avg_score:.3f}")
        if "label" in pairs.columns:
            match_count = int((pairs["label"] == "match").sum())
            lines.append(f"  Confirmed matches: {match_count:,}")

    if isinstance(cluster_summary, pd.DataFrame) and not cluster_summary.empty:
        found_anything = True
        lines.append(f"\nDuplicate clusters: {len(cluster_summary):,}")
        lines.append(f"  (Each cluster = group of records that are likely duplicates)")

    # ── Column-based duplicates ─────────────────────────────────────
    col_dups = state.get("column_duplicates")
    if isinstance(col_dups, pd.DataFrame) and not col_dups.empty:
        found_anything = True
        lines.append(f"\nColumn-level duplicates/near-duplicates: {len(col_dups):,} column pairs")
        if "similarity" in col_dups.columns:
            top = col_dups.nlargest(5, "similarity")
            lines.append("  Top similar column pairs:")
            for _, row in top.iterrows():
                c1 = row.get("col1") or row.get("column1") or "?"
                c2 = row.get("col2") or row.get("column2") or "?"
                sim = row.get("similarity", "?")
                lines.append(f"    {c1} <-> {c2}  (similarity: {sim:.3f})")

    # ── Duplicate issues from validation report ─────────────────────
    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty and "issue" in report.columns:
        dup_issues = report[report["issue"].str.contains("dup", case=False, na=False)]
        if not dup_issues.empty:
            found_anything = True
            lines.append(f"\nDuplicate-type issues in validation report: {len(dup_issues):,}")
            rows_affected = dup_issues["row_id"].nunique() if "row_id" in dup_issues.columns else "?"
            lines.append(f"  Rows affected: {rows_affected}")

    if found_anything:
        return "\n".join(lines), None

    # ── Nothing found — guide user to run dedupe ────────────────────
    return (
        "Duplicate detection has not been run yet.\n\n"
        "Opening the Dedupe panel for you — run detection there first, "
        "then ask me again and I'll summarize the results."
    ), "Dedupe"


def handle_show_data(n: int, state: Dict[str, Any]) -> str:
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load tool."
    n = min(max(1, n), 100)
    head = df.head(n)
    try:
        return f"First {n} rows of the dataset:\n\n{head.to_markdown()}"
    except Exception:
        return f"First {n} rows of the dataset:\n\n```\n{head.to_string()}\n```"


def handle_get_row(row_number: int, state: Dict[str, Any]) -> str:
    """Show all field values for a specific row (1-based row number)."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load tool."
    total = len(df)
    if row_number < 1 or row_number > total:
        return f"Row {row_number} is out of range. The dataset has {total:,} rows (1 to {total})."
    row = df.iloc[row_number - 1]  # convert to 0-based
    lines = [f"**Row {row_number}** (index {row.name}):\n"]
    for col, val in row.items():
        lines.append(f"- **{col}**: {val}")
    return "\n".join(lines)


def handle_lookup_value(
    search_value: str,
    return_field: str,
    search_column: str,
    state: Dict[str, Any],
) -> str:
    """Look up a specific field for a named record."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load tool."

    search_value = search_value.strip()
    if not search_value:
        return "Please specify a name or value to look up."

    # Find matching rows
    if search_column and search_column in df.columns:
        mask = df[search_column].astype(str).str.contains(search_value, case=False, na=False)
    else:
        # Search all object/string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) == 0:
            str_cols = df.columns
        mask = df[str_cols].astype(str).apply(
            lambda col: col.str.contains(search_value, case=False, na=False)
        ).any(axis=1)

    matches = df[mask]
    if matches.empty:
        return f"No record found matching **'{search_value}'** in the dataset."

    # If a specific field is requested, find the best-matching column
    if return_field:
        field_lower = return_field.lower().replace(" ", "")
        # Match column by substring (e.g. "phone number" matches "phone_number")
        candidate_cols = [
            c for c in df.columns
            if field_lower in c.lower().replace("_", "").replace(" ", "")
            or c.lower().replace("_", "").replace(" ", "") in field_lower
        ]
        if candidate_cols:
            col = candidate_cols[0]
            rows_found = len(matches)
            if rows_found == 1:
                val = matches.iloc[0][col]
                return f"The **{col}** for **'{search_value}'** is: **{val}**"
            else:
                vals = matches[col].tolist()
                lines = [f"Found {rows_found} records matching '{search_value}':"]
                for i, (_, row) in enumerate(matches.iterrows()):
                    lines.append(f"  Row {row.name}: **{col}** = {row[col]}")
                return "\n".join(lines)
        else:
            available = ", ".join(df.columns[:15])
            return (
                f"Column matching **'{return_field}'** not found.\n"
                f"Available columns: {available}"
            )

    # No specific field — return all fields for the first match
    row = matches.iloc[0]
    lines = [f"Record for **'{search_value}'** (row {row.name}):\n"]
    for col, val in row.items():
        lines.append(f"- **{col}**: {val}")
    if len(matches) > 1:
        lines.append(f"\n_{len(matches) - 1} more record(s) match — ask for a specific field to narrow down._")
    return "\n".join(lines)


def handle_filter_rows(
    condition: str,
    column: str,
    value: str,
    state: Dict[str, Any],
) -> str:
    """Filter rows by a condition and return matching rows."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load tool."

    condition = condition.lower().strip()
    MAX_ROWS = 20  # cap display to avoid wall-of-text

    if condition == "missing":
        if column and column in df.columns:
            mask = df[column].isna() | (df[column].astype(str).str.strip() == "")
            label = f"missing '{column}'"
        else:
            mask = df.isna().any(axis=1)
            label = "at least one missing value"
        filtered = df[mask]
        if filtered.empty:
            return f"No rows found with {label}."
        snippet = filtered.head(MAX_ROWS).to_string()
        return f"Found **{len(filtered):,}** rows with {label} (showing up to {MAX_ROWS}):\n\n```\n{snippet}\n```"

    elif condition == "not_missing":
        if column and column in df.columns:
            mask = df[column].notna() & (df[column].astype(str).str.strip() != "")
            label = f"non-missing '{column}'"
        else:
            mask = df.notna().all(axis=1)
            label = "all fields filled (complete rows)"
        filtered = df[mask]
        snippet = filtered.head(MAX_ROWS).to_string()
        return f"Found **{len(filtered):,}** rows with {label} (showing up to {MAX_ROWS}):\n\n```\n{snippet}\n```"

    elif condition == "invalid":
        report = state.get("unified_issues_report")
        if not isinstance(report, pd.DataFrame) or report.empty:
            return "No validation results found. Run validation first, then I can filter invalid rows."
        if column and "column" in report.columns:
            col_issues = report[report["column"] == column]
            if col_issues.empty:
                return f"No invalid values found in '{column}'. All values passed validation."
            row_ids = col_issues["row_id"].unique() if "row_id" in col_issues.columns else []
        else:
            row_ids = report["row_id"].unique() if "row_id" in report.columns else []
        filtered = df[df.index.isin(row_ids)]
        if filtered.empty:
            return "No invalid rows found in the dataset."
        snippet = filtered.head(MAX_ROWS).to_string()
        label = f"invalid '{column}'" if column else "at least one invalid value"
        return f"Found **{len(filtered):,}** rows with {label} (showing up to {MAX_ROWS}):\n\n```\n{snippet}\n```"

    elif condition == "contains" and value:
        if column and column in df.columns:
            mask = df[column].astype(str).str.contains(value, case=False, na=False)
            label = f"'{column}' containing '{value}'"
        else:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(value, case=False, na=False)
            ).any(axis=1)
            label = f"any column containing '{value}'"
        filtered = df[mask]
        if filtered.empty:
            return f"No rows found with {label}."
        snippet = filtered.head(MAX_ROWS).to_string()
        return f"Found **{len(filtered):,}** rows with {label} (showing up to {MAX_ROWS}):\n\n```\n{snippet}\n```"

    elif condition == "equals" and value:
        if not column:
            return "Please specify a column name for an 'equals' filter."
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {', '.join(df.columns[:10])}."
        mask = df[column].astype(str).str.strip() == str(value).strip()
        filtered = df[mask]
        if filtered.empty:
            return f"No rows where '{column}' = '{value}'."
        snippet = filtered.head(MAX_ROWS).to_string()
        return f"Found **{len(filtered):,}** rows where '{column}' = '{value}' (showing up to {MAX_ROWS}):\n\n```\n{snippet}\n```"

    return (
        f"Unknown condition '{condition}'. "
        "Use: missing, not_missing, invalid, contains, equals."
    )


def handle_explain_row(row_id: int, state: Dict[str, Any]) -> str:
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No dataset loaded. Please load data first."
    if "row_id" not in report.columns:
        return "Validation report does not contain row IDs."
    row_issues = report[report["row_id"] == row_id]
    if row_issues.empty:
        return auto_note + f"Row {row_id} passed all validation checks — no issues recorded."
    lines = [f"Row {row_id} has {len(row_issues)} issue(s):\n"]
    for _, issue in row_issues.iterrows():
        field    = issue.get("column", "unknown field")
        itype    = issue.get("issue_type", issue.get("issue", "unknown"))
        value    = issue.get("value", "N/A")
        detail   = issue.get("detail_technical", issue.get("detail", ""))
        desc     = issue.get("description", "")
        severity = issue.get("severity", "")
        fix      = issue.get("suggested_fix", None)
        lines.append(f"  Field       : {field}")
        lines.append(f"  Issue       : {itype}  [{severity} severity]")
        lines.append(f"  Value       : {value}")
        if desc:
            lines.append(f"  Description : {desc}")
        if detail:
            lines.append(f"  Detail      : {detail}")
        if fix and str(fix) not in ["nan", "None", ""]:
            lines.append(f"  Suggested : {fix}")
        lines.append("")
    return auto_note + "\n".join(lines)


def handle_count_issues(issue_type: str, state: Dict[str, Any]) -> str:
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No dataset loaded. Please load data first."
    if issue_type == "all":
        total = len(report)
        rows = report["row_id"].nunique() if "row_id" in report.columns else "unknown"
        return auto_note + f"Total issues: {total:,} across {rows} rows."
    if "issue" in report.columns:
        mask  = report["issue"].str.contains(issue_type, case=False, na=False)
        count = int(mask.sum())
        return auto_note + f"Issues matching '{issue_type}': {count:,}"
    return "Could not determine issue counts."


def handle_worst_rows(state: Dict[str, Any], n: int = 10) -> str:
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No dataset loaded. Please load data first."
    if "row_id" not in report.columns:
        return auto_note + "Validation report does not contain row IDs."
    n = min(max(1, n), 50)
    row_counts = (
        report.groupby("row_id").size()
        .sort_values(ascending=False)
        .head(n)
    )
    lines = [f"Top {len(row_counts)} rows with the most issues:\n"]
    for row_id, count in row_counts.items():
        row_issues  = report[report["row_id"] == row_id]
        issue_types = ", ".join(row_issues["issue"].unique()) if "issue" in row_issues.columns else "unknown"
        severities  = ", ".join(row_issues["severity"].unique()) if "severity" in row_issues.columns else ""
        lines.append(f"  Row {row_id:>5}: {count} issue(s)  |  {issue_types}  |  severity: {severities}")
    return auto_note + "\n".join(lines)


def handle_column_issues(column: str, state: Dict[str, Any]) -> str:
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No dataset loaded. Please load data first."
    if "column" not in report.columns:
        return auto_note + "Validation report does not contain column information."
    mask       = report["column"].str.contains(column, case=False, na=False)
    col_issues = report[mask]
    if col_issues.empty:
        return auto_note + f"No issues found for column matching '{column}'."
    lines = [f"Issues in column '{column}': {len(col_issues):,} total\n"]
    if "issue" in col_issues.columns:
        for issue_type, count in col_issues["issue"].value_counts().items():
            lines.append(f"  {issue_type}: {count}")
    return auto_note + "\n".join(lines)


def handle_quality_score(state: Dict[str, Any]) -> str:
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."
    try:
        from core.quality_scores import compute_quality_scores
        scores = compute_quality_scores(df)
        lines  = ["Dataset Quality Scores (0-100):\n"]
        for dim, score in scores.items():
            if isinstance(score, (int, float)):
                filled = int(score / 10)
                bar    = "█" * filled + "░" * (10 - filled)
                lines.append(f"  {dim.capitalize():<15} {bar}  {score:.1f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not compute quality scores: {e}"


def handle_fixable(state: Dict[str, Any]) -> str:
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No dataset loaded. Please load data first."
    if "suggested_fix" not in report.columns:
        return auto_note + "No suggestion data found in the validation report."
    fixable   = report[report["suggested_fix"].notna()]
    total     = len(report)
    fix_count = len(fixable)
    lines = [f"Auto-fixable issues: {fix_count:,} out of {total:,} total issues\n"]
    if not fixable.empty and "issue" in fixable.columns:
        lines.append("By type:")
        for itype, count in fixable["issue"].value_counts().items():
            lines.append(f"  {itype}: {count}")
    return auto_note + "\n".join(lines)


def handle_summary(state: Dict[str, Any]) -> str:
    auto_note = _ensure_validation(state)
    lines = ["Data Quality Summary", "=" * 35]

    df = state.get("df_raw")
    if isinstance(df, pd.DataFrame) and not df.empty:
        lines.append(f"Dataset      : {len(df):,} rows, {len(df.columns)} columns")
    else:
        lines.append("No dataset loaded.")
        return "\n".join(lines)

    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty:
        total      = len(report)
        rows_aff   = report["row_id"].nunique() if "row_id" in report.columns else 0
        clean_rows = len(df) - rows_aff

        lines.append(f"Total issues : {total:,}")
        lines.append(f"Rows flagged : {rows_aff:,}  ({rows_aff/len(df)*100:.1f}%)")
        lines.append(f"Clean rows   : {clean_rows:,}  ({clean_rows/len(df)*100:.1f}%)")

        if "severity" in report.columns:
            high   = int((report["severity"] == "high").sum())
            medium = int((report["severity"] == "medium").sum())
            low    = int((report["severity"] == "low").sum())
            lines.append(f"\nSeverity:")
            lines.append(f"  High   : {high:,}")
            lines.append(f"  Medium : {medium:,}")
            lines.append(f"  Low    : {low:,}")

        if "issue" in report.columns:
            lines.append(f"\nTop issue types:")
            for itype, count in report["issue"].value_counts().head(6).items():
                lines.append(f"  {itype:<28} {count:,}")

        if "suggested_fix" in report.columns:
            fixable = int(report["suggested_fix"].notna().sum())
            lines.append(f"\nAuto-fixable : {fixable:,}")
    else:
        lines.append("Validation has not been run yet.")

    return auto_note + "\n".join(lines)


# ─────────────────────────────────────────────────────────
# New Action Handler Functions
# ─────────────────────────────────────────────────────────

def handle_run_data_profiling(state: Dict[str, Any]) -> str:
    """Run full data profiling and store in state."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load panel."
    try:
        from core.data_profiler import profile_dataset
        profile = profile_dataset(df)
        state["data_profile"] = profile

        summary = profile.get("dataset_summary", {})
        col_profiles = profile.get("column_profiles", {})
        outliers = profile.get("outliers", {})

        lines = [
            f"Data profiling complete on {len(df):,} rows x {len(df.columns)} columns.",
            "",
            "Dataset summary:",
            f"  Total cells      : {summary.get('total_cells', len(df) * len(df.columns)):,}",
            f"  Missing cells    : {summary.get('missing_cells', df.isnull().sum().sum()):,}",
            f"  Duplicate rows   : {summary.get('duplicate_rows', int(df.duplicated().sum())):,}",
            "",
            f"Columns profiled: {len(col_profiles)}",
        ]

        # Columns with outliers
        cols_with_outliers = [c for c, v in outliers.items() if v]
        if cols_with_outliers:
            lines.append(f"Columns with outliers: {', '.join(cols_with_outliers[:5])}"
                         + ("..." if len(cols_with_outliers) > 5 else ""))

        # Top missing columns
        missing_counts = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().any()}
        if missing_counts:
            top_missing = sorted(missing_counts.items(), key=lambda x: -x[1])[:5]
            lines.append("\nTop columns by missing values:")
            for col, cnt in top_missing:
                pct = cnt / len(df) * 100
                lines.append(f"  {col}: {cnt:,} missing ({pct:.1f}%)")

        lines.append("\nProfile saved — open the Data Profiling panel to explore charts and distributions.")
        return "\n".join(lines)
    except Exception as e:
        return f"Data profiling error: {e}"


def handle_generate_rules(state: Dict[str, Any]) -> str:
    """Auto-generate data quality rules using 3-source merge: JSON > Reference > Inferred.

    Priority (highest wins on conflict):
      1. JSON rules (uploaded by user)
      2. Reference rules (derived from clean reference dataset)
      3. Inferred rules (auto-detected from unclean data)
    """
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load panel."
    try:
        from core.validator.discover import merge_rules, rules_from_reference

        sources_used = []

        # ── Source 1: Infer rules from unclean data ──
        inferred_rules = None
        try:
            from core.rule_generator_enhanced import generate_rules_with_metadata
            rules_meta = generate_rules_with_metadata(df)
            inferred_rules = {"columns": {}}
            for col, rule_list in rules_meta.items():
                # merge_rules expects a flat dict per column (e.g. {"regex": "...", "type": "number"})
                # rule_list is a list of rich dicts; the actual rule lives inside "rule_data"
                flat = {}
                for r in rule_list:
                    if not r.get("selected", True):
                        continue  # skip low-quality rules not selected
                    rule_data = r.get("rule_data", {})
                    if isinstance(rule_data, dict):
                        flat.update(rule_data)
                    if "severity" in r:
                        flat["severity"] = r["severity"]
                if flat:
                    inferred_rules["columns"][col] = flat

            # Fallback: for columns with no regex, try to infer a simple pattern
            import re as _re
            for col in df.columns:
                col_rules = inferred_rules["columns"].get(col, {})
                if "regex" not in col_rules:
                    vals = df[col].dropna().astype(str).str.strip()
                    vals = vals[vals != ""]
                    if len(vals) < 5:
                        continue
                    sample = vals.head(50)
                    # Check if all values match a common pattern
                    patterns_to_try = [
                        (r'^[A-Z]\d{3}$', sample),           # e.g. C001
                        (r'^[A-Z]{2}\d{4}$', sample),        # e.g. AB1234
                        (r'^[A-Z]{2,4}-\d+$', sample),       # e.g. CUST-001
                        (r'^\d{8}$', sample),                 # 8-digit number
                        (r'^[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}$', sample),  # UK postcode
                    ]
                    for pat, s in patterns_to_try:
                        matches = s.str.fullmatch(pat, case=False)
                        coverage = matches.sum() / len(s) if len(s) > 0 else 0
                        if coverage >= 0.80:
                            if col not in inferred_rules["columns"]:
                                inferred_rules["columns"][col] = {}
                            inferred_rules["columns"][col]["regex"] = pat
                            break
        except Exception:
            try:
                from core.validator.discover import infer_rules_from_unclean
                inferred_rules = infer_rules_from_unclean(df)
            except Exception:
                pass
        if inferred_rules:
            sources_used.append("inferred (from data patterns)")

        # ── Source 2: Reference rules (from clean dataset) ──
        ref_rules = None
        df_ref = state.get("df_ref")
        if isinstance(df_ref, pd.DataFrame) and not df_ref.empty:
            try:
                ref_rules = rules_from_reference(df_ref)
                sources_used.append("reference (from clean dataset)")
            except Exception:
                pass

        # ── Source 3: JSON rules (uploaded by user) ──
        json_rules = state.get("rules_json")
        if json_rules and isinstance(json_rules, dict):
            sources_used.append("JSON (uploaded rules file)")

        # ── Merge: JSON > Reference > Inferred ──
        rules_merged = merge_rules(json_rules, ref_rules, inferred_rules)
        state["rules_merged"] = rules_merged

        col_rules = rules_merged.get("columns", {})
        total_rules = sum(
            len(v) if isinstance(v, list) else 1
            for v in col_rules.values()
        )

        lines = [
            f"Rules generated for {len(col_rules)} columns ({total_rules} rules total).",
            f"Sources: {', '.join(sources_used) if sources_used else 'none'}",
            "",
            "Columns covered:",
        ]
        for col in list(col_rules.keys())[:10]:
            rule_val = col_rules[col]
            n = len(rule_val) if isinstance(rule_val, list) else 1
            lines.append(f"  {col}: {n} rule(s)")
        if len(col_rules) > 10:
            lines.append(f"  ... and {len(col_rules) - 10} more columns")

        lines.append("\nRules saved and ready for validation. You can now run validation.")
        return "\n".join(lines)
    except Exception as e:
        return f"Rule generation error: {e}"


def handle_apply_fixes(mode: str, state: Dict[str, Any]) -> str:
    """Apply suggested fixes from the validation report."""
    df = state.get("df_raw")
    report = state.get("unified_issues_report")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No validation report available. Please run validation first."
    if "suggested_fix" not in report.columns:
        return "The validation report does not contain any suggested fixes."

    try:
        from core.validation_ui_helpers import apply_fixes_to_dataframe

        fixable = report[report["suggested_fix"].notna()].copy()

        if mode == "high_confidence":
            if "confidence" in fixable.columns:
                fixable = fixable[fixable["confidence"] >= 0.9]
            # If no confidence column, treat all fixes as high confidence

        if fixable.empty:
            return f"No fixes available for mode '{mode}'."

        selected_indices = fixable.index.tolist()
        df_corrected = apply_fixes_to_dataframe(df, report, selected_indices)
        state["df_corrected"] = df_corrected

        lines = [
            f"Fixes applied ({mode} mode):",
            f"  Fixes applied    : {len(selected_indices):,}",
            f"  Original rows    : {len(df):,}",
            f"  Corrected rows   : {len(df_corrected):,}",
        ]
        if "issue" in fixable.columns:
            lines.append("\nFix breakdown by issue type:")
            for itype, cnt in fixable["issue"].value_counts().head(6).items():
                lines.append(f"  {itype}: {cnt}")

        lines.append("\nCorrected dataset saved as df_corrected — open Validate & Fix to download it.")
        return "\n".join(lines)
    except Exception as e:
        return f"Apply fixes error: {e}"


def handle_run_basic_cleaning(
    lowercase: bool,
    strip_whitespace: bool,
    normalize_punctuation: bool,
    state: Dict[str, Any],
) -> str:
    """Run basic text cleaning on the dataset."""
    # Use best available dataframe
    df = None
    for key in ("df_corrected", "df_raw"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            df = state[key]
            break

    if df is None:
        return "No dataset loaded. Please load data first."

    try:
        from core.preprocess import basic_clean
        df_cleaned = basic_clean(
            df,
            lowercase=lowercase,
            strip_ws=strip_whitespace,
            normalize_punct=normalize_punctuation,
        )
        state["df_precleaned"] = df_cleaned

        # Count changes
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        changes = 0
        for col in str_cols:
            try:
                changes += int((df[col].astype(str) != df_cleaned[col].astype(str)).sum())
            except Exception:
                pass

        lines = [
            f"Basic text cleaning complete on {len(df):,} rows.",
            f"  Lowercase        : {'yes' if lowercase else 'no'}",
            f"  Strip whitespace : {'yes' if strip_whitespace else 'no'}",
            f"  Normalize punct  : {'yes' if normalize_punctuation else 'no'}",
            f"  Text columns     : {len(str_cols)}",
            f"  Cells changed    : {changes:,}",
            "",
            "Cleaned dataset saved — open Cleaning & EDA panel to download it.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Cleaning error: {e}"


def handle_apply_column_trimming(
    missing_threshold: float,
    correlation_threshold: float,
    state: Dict[str, Any],
) -> str:
    """Remove low-value columns from the dataset."""
    df = None
    for key in ("df_precleaned", "df_corrected", "df_raw"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            df = state[key]
            break

    if df is None:
        return "No dataset loaded. Please load data first."

    try:
        from core.trimming import compute_kept_columns
        kept_cols = compute_kept_columns(
            df,
            missing_thresh=missing_threshold,
            corr_thresh=correlation_threshold,
        )
        removed_cols = [c for c in df.columns if c not in kept_cols]
        df_trimmed = df[kept_cols].copy()
        state["df_trimmed"] = df_trimmed

        lines = [
            f"Column trimming complete.",
            f"  Original columns : {len(df.columns)}",
            f"  Kept columns     : {len(kept_cols)}",
            f"  Removed columns  : {len(removed_cols)}",
        ]
        if removed_cols:
            lines.append(f"\nRemoved: {', '.join(removed_cols[:10])}"
                         + ("..." if len(removed_cols) > 10 else ""))
        lines.append("\nTrimmed dataset saved — open Cleaning & EDA panel to download it.")
        return "\n".join(lines)
    except Exception as e:
        return f"Column trimming error: {e}"


def handle_merge_duplicates(strategy: str, state: Dict[str, Any]) -> str:
    """Merge duplicate clusters into a deduplicated dataset. Auto-runs dedup if needed."""
    # Auto-run dedup silently if not yet done
    auto_note = ""
    if not state.get("dedup_clusters"):
        auto_note = _ensure_dedup(state)

    clusters = state.get("dedup_clusters")
    if not clusters:
        return (
            (auto_note or "")
            + "No duplicate clusters found. "
            + "The dataset may not have duplicates, or detection could not run automatically."
        )

    df = None
    for key in ("df_trimmed", "df_precleaned", "df_corrected", "df_raw"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            df = state[key]
            break

    if df is None:
        return "No dataset loaded."

    try:
        from core.dedup_clustering import apply_deduplication

        # Auto-select first row in each cluster as master
        master_selections: Dict[int, int] = {}
        for cluster_id, row_ids in clusters.items():
            if row_ids:
                master_selections[cluster_id] = row_ids[0]

        df_dedup = apply_deduplication(
            df,
            clusters=clusters,
            master_selections=master_selections,
            merge_strategy=strategy,
        )
        state["df_deduplicated"] = df_dedup

        rows_removed = len(df) - len(df_dedup)
        lines = [
            f"Deduplication complete.",
            f"  Strategy         : {strategy}",
            f"  Clusters merged  : {len(clusters):,}",
            f"  Original rows    : {len(df):,}",
            f"  Deduplicated rows: {len(df_dedup):,}",
            f"  Rows removed     : {rows_removed:,} ({rows_removed/len(df)*100:.1f}%)",
            "",
            "Deduplicated dataset saved — open the Dedupe panel to download it.",
        ]
        return auto_note + "\n".join(lines)
    except Exception as e:
        return f"Merge duplicates error: {e}"


def handle_run_auto_report(state: Dict[str, Any]) -> str:
    """Run the full end-to-end pipeline: validate → clean → deduplicate → score."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first using the Load panel."

    results = []
    current_df = df.copy()

    # Step 1: Validate
    results.append("Step 1/4: Running validation...")
    try:
        from core.validator.validate import validate_and_anomaly_report
        rules = state.get("rules_merged") or {}
        report = validate_and_anomaly_report(current_df, rules=rules, use_ml=False)
        state["unified_issues_report"] = report
        total_issues = len(report)
        results.append(f"  Validation done — {total_issues:,} issues found.")
    except Exception as e:
        results.append(f"  Validation failed: {e}")
        report = None

    # Step 2: Basic cleaning
    results.append("Step 2/4: Running basic text cleaning...")
    try:
        from core.preprocess import basic_clean
        current_df = basic_clean(current_df, lowercase=True, strip_ws=True, normalize_punct=True)
        state["df_precleaned"] = current_df
        results.append(f"  Cleaning done — {len(current_df):,} rows.")
    except Exception as e:
        results.append(f"  Cleaning failed: {e}")

    # Step 3: Duplicate detection + merge
    results.append("Step 3/4: Running duplicate detection...")
    try:
        from core.vectorizers import build_vectorizers
        from core.matcher import compute_pair_scores, label_pairs
        from core.dedup_clustering import cluster_duplicates, apply_deduplication

        str_cols = current_df.select_dtypes(include=["object", "string"]).columns.tolist()
        if str_cols and len(current_df) >= 2:
            vec_cfg = {"ngram_range": (1, 2), "max_features": 10000,
                       "word_range": (1, 2), "svd_components": 0}
            vec_pack = build_vectorizers(current_df, str_cols, vec_cfg)
            pairs_df  = compute_pair_scores(current_df, vec_pack, run_mode="Fast KNN")
            pairs_df  = label_pairs(pairs_df, similar_thr=0.6, duplicate_thr=0.85)
            clusters  = cluster_duplicates(pairs_df, id_col_a="row_i", id_col_b="row_j")
            state["dedup_pairs"]    = pairs_df
            state["dedup_clusters"] = clusters

            dup_count = int((pairs_df["label"] == "duplicate").sum()) if "label" in pairs_df.columns else len(pairs_df)

            if clusters:
                master_selections = {cid: rids[0] for cid, rids in clusters.items() if rids}
                current_df = apply_deduplication(
                    current_df, clusters=clusters,
                    master_selections=master_selections, merge_strategy="keep_master"
                )
                state["df_deduplicated"] = current_df
                results.append(f"  Deduplicated — {dup_count} duplicate pairs, {len(current_df):,} rows remain.")
            else:
                results.append(f"  No duplicates found.")
        else:
            results.append(f"  Skipped — no text columns or too few rows.")
    except Exception as e:
        results.append(f"  Deduplication failed: {e}")

    # Step 4: Quality scoring
    results.append("Step 4/4: Computing quality scores...")
    try:
        from core.quality_scores import compute_quality_scores
        score_before = compute_quality_scores(df)       # original raw data
        score_after  = compute_quality_scores(current_df)  # after cleaning/dedup
        state["auto_scores"]    = {"before": score_before, "after": score_after}
        state["df_auto_after"]  = current_df
        scores = score_after

        avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(
            len([v for v in scores.values() if isinstance(v, (int, float))]), 1
        )
        results.append(f"  Quality score: {avg_score:.1f}/100")
    except Exception as e:
        results.append(f"  Scoring failed: {e}")
        avg_score = None

    # Summary
    results.append("")
    results.append("Auto report complete.")
    results.append(f"  Final dataset: {len(current_df):,} rows x {len(current_df.columns)} columns")
    if avg_score:
        results.append(f"  Overall quality score: {avg_score:.1f}/100")
    results.append("Open the Auto Report panel to download the full cleaned dataset and report.")

    return "\n".join(results)


def handle_execute_pipeline(pipeline_name: str, state: Dict[str, Any]) -> str:
    """Execute a saved pipeline by name, or list available pipelines."""
    try:
        from core.pipeline_manager import PipelineManager, PipelineExecutor

        pm = PipelineManager()
        available = pm.list_pipelines()

        if not pipeline_name:
            if not available:
                return "No saved pipelines found. Create one in the Pipeline Manager panel first."
            names = [p.get("name", "?") for p in available]
            return (
                f"Available pipelines ({len(names)}):\n"
                + "\n".join(f"  - {n}" for n in names)
                + "\n\nAsk me to run one by name."
            )

        # Find the pipeline
        pipeline = pm.load_pipeline(pipeline_name)
        if pipeline is None:
            names = [p.get("name", "?") for p in available]
            return (
                f"Pipeline '{pipeline_name}' not found.\n"
                f"Available: {', '.join(names) if names else 'none'}"
            )

        df = state.get("df_raw")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No dataset loaded. Please load data first."

        executor = PipelineExecutor(state)
        result = executor.execute_pipeline(pipeline, df)

        if result.get("df_output") is not None:
            state["df_pipeline_output"] = result["df_output"]

        steps_done = result.get("steps_executed", [])
        errors = result.get("errors", [])
        success = result.get("success", False)

        lines = [
            f"Pipeline '{pipeline_name}' execution {'complete' if success else 'finished with errors'}.",
            f"  Steps executed: {len(steps_done)}",
        ]
        if steps_done:
            for s in steps_done[:5]:
                lines.append(f"    - {s}")
        if errors:
            lines.append(f"  Errors: {len(errors)}")
            for e in errors[:3]:
                lines.append(f"    - {e}")
        if result.get("df_output") is not None:
            out_df = result["df_output"]
            lines.append(f"  Output dataset: {len(out_df):,} rows x {len(out_df.columns)} columns")
            lines.append("Open Pipeline Manager panel to download the output.")

        return "\n".join(lines)
    except Exception as e:
        return f"Pipeline execution error: {e}"


def handle_run_entity_resolution(threshold: float, state: Dict[str, Any]) -> str:
    """Find matching entities between the main dataset and the reference dataset."""
    df_main = state.get("df_raw")
    df_ref = state.get("df_ref")

    if not isinstance(df_main, pd.DataFrame) or df_main.empty:
        return "No main dataset loaded. Please load data first using the Load panel."
    if not isinstance(df_ref, pd.DataFrame) or df_ref.empty:
        return (
            "No reference dataset loaded. Entity resolution requires two datasets. "
            "Please load a reference dataset in the Load panel."
        )

    try:
        from core.entity_resolution import find_entity_matches

        datasets = {
            "main": df_main,
            "reference": df_ref,
        }
        pairs = find_entity_matches(datasets, threshold=threshold)
        state["er_pairs"]    = pairs
        state["er_datasets"] = datasets

        if pairs is None or (isinstance(pairs, pd.DataFrame) and pairs.empty):
            return (
                f"Entity resolution complete — no matches found above threshold {threshold}.\n"
                "Try lowering the threshold."
            )

        high_conf = int((pairs["score"] >= 0.9).sum()) if "score" in pairs.columns else 0
        avg_score = float(pairs["score"].mean()) if "score" in pairs.columns else 0.0

        lines = [
            f"Entity resolution complete.",
            f"  Threshold        : {threshold}",
            f"  Main dataset     : {len(df_main):,} rows",
            f"  Reference dataset: {len(df_ref):,} rows",
            f"  Matches found    : {len(pairs):,}",
            f"  High confidence  : {high_conf:,} (score >= 0.9)",
            f"  Avg match score  : {avg_score:.3f}",
            "",
            "Results saved — open the Entity Resolution panel to explore matches and download.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Entity resolution error: {e}"


def handle_get_profile(column: Optional[str], state: Dict[str, Any]) -> str:
    """Return profiling stats for a column or full dataset summary."""
    profile = state.get("data_profile")

    # Auto-run profiling if not done yet
    auto_note = ""
    if not isinstance(profile, dict):
        df = state.get("df_raw")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No dataset loaded. Please load data first."
        try:
            from core.data_profiler import profile_dataset
            profile = profile_dataset(df)
            state["data_profile"] = profile
            auto_note = "*Profiling ran automatically.*\n\n"
        except Exception as e:
            return f"Could not run profiling: {e}"

    col_profiles = profile.get("column_profiles", {})
    summary      = profile.get("dataset_summary", {})

    # ── No column specified → full dataset overview ────────────
    if not column or not column.strip():
        lines = [auto_note + "Dataset Profile Overview", "=" * 38]
        lines.append(f"  Rows              : {summary.get('total_rows', '?'):,}")
        lines.append(f"  Columns           : {summary.get('total_columns', '?')}")
        lines.append(f"  Missing cells     : {summary.get('missing_cells', '?'):,}  ({summary.get('missing_percentage', '?')}%)")
        lines.append(f"  Duplicate rows    : {summary.get('duplicate_rows', '?'):,}  ({summary.get('duplicate_percentage', '?')}%)")
        lines.append(f"  Memory usage      : {summary.get('memory_usage_mb', 0):.2f} MB")
        lines.append(f"  Numeric columns   : {summary.get('numeric_columns', '?')}")
        lines.append(f"  Categorical cols  : {summary.get('categorical_columns', '?')}")

        if col_profiles:
            lines.append("\nPer-column null rates:")
            for col_name, cp in col_profiles.items():
                pct = cp.get("missing_pct", 0)
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                lines.append(f"  {col_name:<22} {bar}  {pct:.1f}% missing")

        outliers = profile.get("outliers", {})
        if outliers.get("has_outliers") and outliers.get("outlier_columns"):
            lines.append(f"\nOutlier columns detected: {outliers['total_outlier_columns']}")
            for col_name, od in list(outliers["outlier_columns"].items())[:5]:
                lines.append(f"  {col_name}: {od['count']:,} outliers ({od['percentage']}%)")

        lines.append("\nAsk me about a specific column: 'what are the top values in company_name?'")
        return "\n".join(lines)

    # ── Specific column ────────────────────────────────────────
    col = column.strip()
    # Case-insensitive partial match
    matched_col = None
    for k in col_profiles:
        if col.lower() == k.lower() or col.lower() in k.lower():
            matched_col = k
            break

    if matched_col is None:
        available = ", ".join(col_profiles.keys())
        return (
            f"Column '{col}' not found in the profile.\n"
            f"Available columns: {available}"
        )

    cp    = col_profiles[matched_col]
    dtype = cp.get("dtype", "?")
    lines = [auto_note + f"Profile — '{matched_col}'  [{dtype}]", "─" * 38]

    # Core stats (all types)
    lines.append(f"  Count     : {cp.get('count', '?'):,}")
    lines.append(f"  Missing   : {cp.get('missing', '?'):,}  ({cp.get('missing_pct', '?')}%)")
    lines.append(f"  Unique    : {cp.get('unique', '?'):,}  ({cp.get('unique_pct', '?')}% of rows)")

    # Numeric extras
    if cp.get("mean") is not None:
        lines.append(f"  Min / Max : {cp.get('min')} / {cp.get('max')}")
        lines.append(f"  Mean      : {cp.get('mean'):.4g}")
        lines.append(f"  Median    : {cp.get('median'):.4g}")
        lines.append(f"  Std dev   : {cp.get('std'):.4g}")
        if cp.get("skewness") is not None:
            skew = cp["skewness"]
            skew_label = "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "approx. normal"
            lines.append(f"  Skewness  : {skew:.3f}  ({skew_label})")
        if cp.get("zeros", 0) > 0:
            lines.append(f"  Zeros     : {cp['zeros']:,}")
        if cp.get("negatives", 0) > 0:
            lines.append(f"  Negatives : {cp['negatives']:,}")

        # Outlier info if available
        outlier_cols = profile.get("outliers", {}).get("outlier_columns", {})
        if matched_col in outlier_cols:
            od = outlier_cols[matched_col]
            lines.append(f"  Outliers  : {od['count']:,}  ({od['percentage']}%)")
            lines.append(f"  IQR range : [{od['lower_bound']:.4g} – {od['upper_bound']:.4g}]")

    # String/categorical extras
    top_values = cp.get("top_values", [])
    if top_values:
        lines.append(f"\n  Mode      : '{cp.get('mode')}'  (appears {cp.get('mode_frequency', '?'):,}×)")
        lines.append(f"  Avg length: {cp.get('avg_length', '?')} chars  (min {cp.get('min_length')} / max {cp.get('max_length')})")
        lines.append(f"\n  Top values:")
        for item in top_values[:8]:
            val  = item.get("value", "?")
            cnt  = item.get("count", 0)
            pct  = item.get("percentage", 0)
            bar  = "█" * max(1, int(pct / 5))
            lines.append(f"    {bar:<20} {val!r:<28} {cnt:>5,}  ({pct:.1f}%)")

    # Detected patterns
    patterns = cp.get("detected_patterns", {})
    if patterns:
        lines.append(f"\n  Detected patterns:")
        for pname, pcount in patterns.items():
            lines.append(f"    {pname}: ~{pcount:,} values")

    return "\n".join(lines)


def handle_recommend_next_steps(state: Dict[str, Any]) -> str:
    """Analyse pipeline state and recommend the most logical next actions."""
    df = state.get("df_raw")

    # ── Step completion map ────────────────────────────────────
    done: Dict[str, bool] = {
        "load":          isinstance(df, pd.DataFrame) and not df.empty,
        "profile":       isinstance(state.get("data_profile"), dict),
        "rules":         bool(state.get("rules_merged")),
        "corpus_load":   state.get("corpus_manager") is not None,
        "corpus_check":  isinstance(state.get("corpus_issues_report"), pd.DataFrame),
        "validate":      isinstance(state.get("unified_issues_report"), pd.DataFrame)
                         and not state["unified_issues_report"].empty,
        "apply_fixes":   isinstance(state.get("df_corrected"), pd.DataFrame),
        "clean":         isinstance(state.get("df_precleaned"), pd.DataFrame),
        "standardize":   isinstance(state.get("df_standardized"), pd.DataFrame),
        "trim":          isinstance(state.get("df_trimmed"), pd.DataFrame),
        "dedup_detect":  isinstance(state.get("dedup_pairs"), pd.DataFrame),
        "dedup_merge":   isinstance(state.get("df_deduplicated"), pd.DataFrame),
        "score":         isinstance(state.get("auto_scores"), dict),
    }

    lines = ["Recommended Next Steps", "=" * 38]

    # ── Not started at all ─────────────────────────────────────
    if not done["load"]:
        lines.append("\nNo data loaded yet.")
        lines.append("\n  1. Open the Load Data panel and upload your dataset.")
        lines.append("     Say: 'open load' or use the Load button above.")
        return "\n".join(lines)

    # ── Build status summary ───────────────────────────────────
    n_cols = len(df.columns)
    n_rows = len(df)
    lines.append(f"\nDataset: {n_rows:,} rows x {n_cols} columns")

    status_lines = []
    status_lines.append(f"  {'Profiling':<26} {'Done' if done['profile'] else 'Not done'}")
    status_lines.append(f"  {'Rules generated':<26} {'Done' if done['rules'] else 'Not done'}")
    status_lines.append(f"  {'Corpus manager connected':<26} {'Yes' if done['corpus_load'] else 'No'}")
    status_lines.append(f"  {'Corpus check run':<26} {'Done' if done['corpus_check'] else 'Not done'}")
    status_lines.append(f"  {'Validation run':<26} {'Done' if done['validate'] else 'Not done'}")
    status_lines.append(f"  {'Fixes applied':<26} {'Done' if done['apply_fixes'] else 'Not done'}")
    status_lines.append(f"  {'Basic cleaning':<26} {'Done' if done['clean'] else 'Not done'}")
    status_lines.append(f"  {'Corpus standardization':<26} {'Done' if done['standardize'] else 'Not done'}")
    status_lines.append(f"  {'Column trimming':<26} {'Done' if done['trim'] else 'Not done'}")
    status_lines.append(f"  {'Duplicate detection':<26} {'Done' if done['dedup_detect'] else 'Not done'}")
    status_lines.append(f"  {'Duplicates merged':<26} {'Done' if done['dedup_merge'] else 'Not done'}")
    status_lines.append(f"  {'Quality scored':<26} {'Done' if done['score'] else 'Not done'}")
    lines.append("\nPipeline status:")
    lines.extend(status_lines)

    # ── Recommend the next 2-3 actions ────────────────────────
    recs = []

    if not done["profile"]:
        recs.append(("Profile the data",
                     "Understand column types, null rates, and distributions before anything else.",
                     "Say: 'profile the data'"))

    if not done["rules"]:
        recs.append(("Generate rules",
                     "Auto-infer validation rules from your data (patterns, ranges, formats).",
                     "Say: 'generate rules'"))

    if done["rules"] and not done["validate"]:
        recs.append(("Run validation",
                     "Check your data against the generated rules to find issues.",
                     "Say: 'validate the data'"))

    if done["validate"] and not done["apply_fixes"]:
        report = state.get("unified_issues_report")
        if isinstance(report, pd.DataFrame) and "suggested_fix" in report.columns:
            fixable = int(report["suggested_fix"].notna().sum())
            if fixable > 0:
                recs.append(("Apply fixes",
                             f"{fixable:,} auto-fixable issues found — apply them to clean the data.",
                             "Say: 'apply high confidence fixes'"))

    if not done["clean"]:
        recs.append(("Basic text cleaning",
                     "Strip whitespace, normalize punctuation, lowercase — quick wins.",
                     "Say: 'clean the data'"))

    if done["corpus_load"] and not done["corpus_check"]:
        try:
            cm = state.get("corpus_manager")
            corpora = cm.list_corpora() if cm else []
        except Exception:
            corpora = []
        if corpora:
            recs.append(("Run corpus check",
                         f"{len(corpora)} corpus(es) loaded — validate values against reference data.",
                         "Say: 'check corpus'"))

    if done["corpus_check"] and not done["standardize"]:
        recs.append(("Apply corpus standardization",
                     "Replace non-canonical values with their canonical forms.",
                     "Say: 'apply corpus standardization'"))

    if not done["trim"] and n_cols > 10:
        recs.append(("Trim low-value columns",
                     f"Dataset has {n_cols} columns — remove highly missing or correlated ones.",
                     "Say: 'trim columns'"))

    if not done["dedup_detect"]:
        recs.append(("Detect duplicates",
                     "Find near-duplicate rows that should be merged.",
                     "Say: 'find duplicates'"))

    if done["dedup_detect"] and not done["dedup_merge"]:
        pairs = state.get("dedup_pairs")
        dup_count = 0
        if isinstance(pairs, pd.DataFrame) and "label" in pairs.columns:
            dup_count = int((pairs["label"] == "duplicate").sum())
        if dup_count > 0:
            recs.append(("Merge duplicates",
                         f"{dup_count:,} duplicate pairs found — merge clusters into single records.",
                         "Say: 'merge duplicates'"))

    if not done["score"] and (done["validate"] or done["clean"] or done["dedup_merge"]):
        recs.append(("Check quality score",
                     "See dimension scores (completeness, consistency, uniqueness, etc.).",
                     "Say: 'what is the quality score?'"))

    # All done path
    if done["validate"] and done["clean"] and done["dedup_detect"] and done["score"]:
        recs.append(("Compare before vs after",
                     "See how much the quality improved across all steps.",
                     "Say: 'compare before vs after'"))
        recs.append(("Run auto report",
                     "Generate a full downloadable HTML report with all results.",
                     "Say: 'run auto report'"))

    if not recs:
        lines.append("\nAll major steps are complete!")
        lines.append("  Say 'compare before vs after' to see the quality improvement.")
        lines.append("  Say 'run auto report' to generate a downloadable report.")
        return "\n".join(lines)

    lines.append(f"\nRecommended next {min(len(recs), 3)} action(s):")
    for i, (title, reason, prompt) in enumerate(recs[:3], 1):
        lines.append(f"\n  {i}. {title}")
        lines.append(f"     {reason}")
        lines.append(f"     {prompt}")

    return "\n".join(lines)


def handle_apply_corpus_standardization(state: Dict[str, Any]) -> str:
    """Apply alias corpus replacements to standardize values in the dataset."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."

    cm = _ensure_corpus_manager(state)
    if cm is None:
        return (
            "Redis is not running — corpus standardization requires a Redis connection.\n"
            "Start Redis with: redis-server"
        )

    try:
        corpora = cm.list_corpora()
    except Exception as e:
        return f"Could not connect to corpus data: {e}"

    # Only alias-type corpora do standardization
    alias_corpora = [c for c in corpora if c.get("corpus_type", "") == "alias"]
    if not alias_corpora:
        return (
            "No alias-type corpora loaded. Corpus standardization requires alias corpora.\n\n"
            "Currently loaded corpora: "
            + (", ".join(c["corpus_name"] for c in corpora) if corpora else "none")
            + "\n\nOpening the Corpus Manager panel — load an alias corpus there.",
            "Corpus Manager",
        )

    try:
        from core.validation_ui_helpers import auto_detect_corpus_mappings
        from core.corpus_validation import apply_corpus_standardization, validate_all_corpus_mappings

        # Use most-processed dataframe available as the base
        base_df = df
        for key in ("df_corrected", "df_precleaned", "df_raw"):
            if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
                base_df = state[key]
                break

        corpus_list = [
            {"corpus_name": c["corpus_name"], "corpus_type": c.get("corpus_type", "alias")}
            for c in alias_corpora
        ]

        # Auto-detect which columns map to which alias corpus
        mappings = auto_detect_corpus_mappings(base_df, corpus_list)
        if not mappings:
            corpus_names = ", ".join(c["corpus_name"] for c in alias_corpora)
            return (
                f"No columns could be automatically mapped to the alias corpora ({corpus_names}).\n\n"
                "Make sure your corpus names match the columns they validate "
                "(e.g. corpus 'company_names' maps to a 'company' or 'company_name' column)."
            )

        corpus_mappings = {
            col: {"corpus_name": info["corpus"], "corpus_type": "alias"}
            for col, info in mappings.items()
            if info.get("type") == "alias"
        }

        if not corpus_mappings:
            return "No alias mappings found for any column."

        df_standardized = apply_corpus_standardization(base_df, corpus_mappings, cm)
        state["df_standardized"] = df_standardized

        # Count how many values were actually changed
        changed_cells = 0
        changed_cols  = []
        for col in corpus_mappings:
            if col in base_df.columns and col in df_standardized.columns:
                n_changed = (base_df[col].astype(str) != df_standardized[col].astype(str)).sum()
                if n_changed > 0:
                    changed_cells += n_changed
                    changed_cols.append(f"{col} ({n_changed:,} values)")

        lines = [
            f"Corpus standardization complete.",
            f"  Columns processed : {', '.join(corpus_mappings.keys())}",
            f"  Values changed    : {changed_cells:,} cells",
        ]
        if changed_cols:
            lines.append("\nChanges by column:")
            for c in changed_cols:
                lines.append(f"  {c}")
        else:
            lines.append("  (All values were already in canonical form)")

        lines.append(
            "\nStandardized dataset saved as 'df_standardized'. "
            "You can now run 'validate the data' or 'compare before vs after' to see the improvement."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"Corpus standardization error: {e}"


def handle_compare_versions(state: Dict[str, Any]) -> str:
    """Compare original vs best available cleaned version of the dataset."""
    df_before = state.get("df_raw")
    if not isinstance(df_before, pd.DataFrame) or df_before.empty:
        return "No dataset loaded. Please load data first."

    # Find the best available 'after' dataframe (most processed wins)
    after_key = None
    for key in ("df_auto_after", "df_deduplicated", "df_corrected",
                "df_standardized", "df_trimmed", "df_precleaned"):
        if isinstance(state.get(key), pd.DataFrame) and not state[key].empty:
            after_key = key
            break

    lines = ["Before vs After — Data Quality Comparison", "=" * 45]

    # ── Row / column counts ──────────────────────────────
    lines.append("\nDataset dimensions:")
    lines.append(f"  Before : {len(df_before):>7,} rows  x  {len(df_before.columns)} columns")
    if after_key:
        df_after = state[after_key]
        label = {
            "df_auto_after":    "Auto Report output",
            "df_deduplicated":  "After deduplication",
            "df_corrected":     "After fix application",
            "df_standardized":  "After corpus standardization",
            "df_trimmed":       "After column trim",
            "df_precleaned":    "After basic cleaning",
        }.get(after_key, after_key)
        rows_removed = len(df_before) - len(df_after)
        cols_removed = len(df_before.columns) - len(df_after.columns)
        lines.append(f"  After  : {len(df_after):>7,} rows  x  {len(df_after.columns)} columns  ({label})")
        if rows_removed != 0:
            lines.append(f"  Rows removed  : {rows_removed:,} ({rows_removed / len(df_before) * 100:.1f}%)")
        if cols_removed != 0:
            lines.append(f"  Cols removed  : {cols_removed}")
    else:
        lines.append("  After  : no processed version found yet — run cleaning, validation or auto report first.")

    # ── Quality scores ────────────────────────────────────
    auto_scores = state.get("auto_scores")
    if isinstance(auto_scores, dict) and "before" in auto_scores and "after" in auto_scores:
        before_scores = auto_scores["before"]
        after_scores  = auto_scores["after"]
        lines.append("\nQuality dimension scores (0–100):")
        all_dims = sorted(set(list(before_scores.keys()) + list(after_scores.keys())))
        for dim in all_dims:
            b = before_scores.get(dim)
            a = after_scores.get(dim)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta = a - b
                arrow = "+" if delta >= 0 else ""
                bar_b = "█" * int(b / 10) + "░" * (10 - int(b / 10))
                bar_a = "█" * int(a / 10) + "░" * (10 - int(a / 10))
                lines.append(
                    f"  {dim.capitalize():<14}  {bar_b}  {b:>5.1f}  →  {bar_a}  {a:>5.1f}  ({arrow}{delta:.1f})"
                )
        avg_b = sum(v for v in before_scores.values() if isinstance(v, (int, float))) / max(len(before_scores), 1)
        avg_a = sum(v for v in after_scores.values()  if isinstance(v, (int, float))) / max(len(after_scores), 1)
        delta_avg = avg_a - avg_b
        lines.append(f"\n  Overall average:  {avg_b:.1f}  →  {avg_a:.1f}  (+{delta_avg:.1f})")
    else:
        # Compute on the fly if scores aren't cached
        try:
            from core.quality_scores import compute_quality_scores
            before_scores = compute_quality_scores(df_before)
            lines.append("\nQuality dimension scores (before only — run Auto Report for full before/after):")
            for dim, score in before_scores.items():
                if isinstance(score, (int, float)):
                    bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                    lines.append(f"  {dim.capitalize():<14}  {bar}  {score:.1f}")
            if after_key:
                after_scores = compute_quality_scores(state[after_key])
                lines.append("\nQuality dimension scores (after):")
                for dim, score in after_scores.items():
                    if isinstance(score, (int, float)):
                        b = before_scores.get(dim, 0)
                        delta = score - b
                        arrow = "+" if delta >= 0 else ""
                        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                        lines.append(f"  {dim.capitalize():<14}  {bar}  {score:.1f}  ({arrow}{delta:.1f})")
        except Exception as e:
            lines.append(f"\nCould not compute quality scores: {e}")

    # ── Issue counts ──────────────────────────────────────
    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty:
        total_issues = len(report)
        rows_flagged = report["row_id"].nunique() if "row_id" in report.columns else "?"
        pct = (rows_flagged / len(df_before) * 100) if isinstance(rows_flagged, int) else 0
        lines.append(f"\nValidation issues in original data: {total_issues:,} across {rows_flagged} rows ({pct:.1f}%)")

        if "suggested_fix" in report.columns:
            fixable = int(report["suggested_fix"].notna().sum())
            lines.append(f"  Auto-fixable: {fixable:,}")

    # ── Null rate improvement ─────────────────────────────
    if after_key:
        df_after = state[after_key]
        # Only compare columns present in both
        common_cols = [c for c in df_before.columns if c in df_after.columns]
        if common_cols:
            null_before = df_before[common_cols].isna().mean().mean() * 100
            null_after  = df_after[common_cols].isna().mean().mean() * 100
            delta_null  = null_before - null_after
            if abs(delta_null) > 0.1:
                arrow = "-" if delta_null > 0 else "+"
                lines.append(
                    f"\nNull rate (avg across columns): {null_before:.1f}%  →  {null_after:.1f}%  ({arrow}{abs(delta_null):.1f}%)"
                )

    lines.append(
        "\nTip: Run 'run auto report' for a full pipeline and downloadable before/after comparison."
    )
    return "\n".join(lines)


def handle_list_corpora(state: Dict[str, Any]) -> str:
    """List all corpora currently loaded in Redis."""
    cm = _ensure_corpus_manager(state)
    if cm is None:
        return (
            "Redis is not running — cannot list corpora.\n\n"
            "Start Redis with:  redis-server\n"
            "Or on macOS:       brew services start redis"
        )
    try:
        corpora = cm.list_corpora()
    except Exception as e:
        return f"Could not list corpora: {e}"

    if not corpora:
        return (
            "No corpora loaded in Redis yet.\n\n"
            "Opening the Corpus Manager panel — upload reference data there "
            "(e.g. canonical company names, valid postcodes, email domains). "
            "Then ask me again.",
            "Corpus Manager",
        )

    lines = [f"Loaded corpora: {len(corpora)}\n"]
    for c in corpora:
        name      = c.get("corpus_name", "?")
        ctype     = c.get("corpus_type", "?")
        total     = c.get("total_rows", c.get("count", "?"))
        loaded_at = c.get("loaded_at", "")
        lines.append(f"  {name}")
        lines.append(f"    Type    : {ctype}  |  Records: {total}" + (f"  |  Loaded: {loaded_at}" if loaded_at else ""))

    lines.append(
        "\nTo validate your data against these corpora, ask: 'check corpus' or 'check corpus for the <column> column'."
    )
    return "\n".join(lines)


def handle_list_rules(column: Optional[str], state: Dict[str, Any]) -> str:
    """List active validation rules, optionally filtered to one column."""
    rules = state.get("rules_merged")
    if not rules or not isinstance(rules, dict):
        # Auto-generate rules from the data so the user gets an immediate answer
        df = state.get("df_raw")
        if isinstance(df, pd.DataFrame) and not df.empty:
            gen_result = handle_generate_rules(state)
            rules = state.get("rules_merged")
            if not rules or not isinstance(rules, dict):
                return f"Could not auto-generate rules.\n{gen_result}"
            # Fall through to display the newly generated rules
        else:
            return (
                "No rules are active yet and no dataset is loaded.\n\n"
                "Load a dataset first, then ask me to 'generate rules' or "
                "open the Rules panel to create them manually."
            )

    col_rules: Dict[str, Any] = rules.get("columns", rules)  # handle both schemas

    if not col_rules:
        return "Rules are loaded but contain no column-level rules."

    if column and column.strip():
        # Filter to one column — case-insensitive partial match
        col = column.strip().lower()
        matched = {k: v for k, v in col_rules.items() if col in k.lower()}
        if not matched:
            available = ", ".join(col_rules.keys())
            return (
                f"No rules found for column matching '{column}'.\n"
                f"Columns with rules: {available}"
            )
        col_rules = matched

    # Sort columns by number of rules descending (frequency order)
    sorted_cols = sorted(
        col_rules.items(),
        key=lambda kv: len(kv[1]) if isinstance(kv[1], list) else 1,
        reverse=True,
    )

    def _count_rules(v):
        if isinstance(v, list):
            return len(v)
        if isinstance(v, dict):
            return sum(1 for k in ("regex", "required", "type", "allowed_values", "min", "max") if k in v)
        return 1
    total_rules = sum(_count_rules(v) for _, v in sorted_cols)
    lines = [f"Validation rules: {total_rules} rules across {len(col_rules)} column(s)\n"]

    for col_name, col_rule in sorted_cols:
        # col_rule can be a flat dict {"regex": "...", "type": "number"} or a list of rule dicts
        if isinstance(col_rule, dict):
            # Flat dict — show each key-value as a separate rule line
            lines.append(f"\n{col_name}")
            severity = col_rule.get("severity", "medium")
            if "regex" in col_rule:
                lines.append(f"  [{severity}] regex  pattern: `{col_rule['regex']}`")
            if "required" in col_rule and col_rule["required"]:
                lines.append(f"  [{severity}] required  (must not be empty)")
            if "type" in col_rule:
                ctype = col_rule["type"]
                if ctype == "number":
                    bounds = ""
                    if "min" in col_rule:
                        bounds += f"  min={col_rule['min']}"
                    if "max" in col_rule:
                        bounds += f"  max={col_rule['max']}"
                    lines.append(f"  [{severity}] type_numeric{bounds}")
                elif ctype == "date":
                    lines.append(f"  [{severity}] type_date")
                else:
                    lines.append(f"  [{severity}] type={ctype}")
            if "allowed_values" in col_rule:
                vals = col_rule["allowed_values"]
                if isinstance(vals, list):
                    sample = ", ".join(str(v) for v in vals[:6])
                    lines.append(f"  [{severity}] allowed_values: {sample}" + (" ..." if len(vals) > 6 else ""))
            if "min" in col_rule and "type" not in col_rule:
                lines.append(f"  [{severity}] min={col_rule['min']}")
            if "max" in col_rule and "type" not in col_rule:
                lines.append(f"  [{severity}] max={col_rule['max']}")
            # Count rules shown for this column
            rule_count = sum(1 for k in ("regex", "required", "type", "allowed_values") if k in col_rule)
            if rule_count == 0:
                # Show raw keys as fallback
                extras = {k: v for k, v in col_rule.items() if k not in ("severity", "metadata")}
                for k, v in list(extras.items())[:5]:
                    lines.append(f"  [{severity}] {k}={v}")
        elif isinstance(col_rule, list):
            lines.append(f"\n{col_name}  ({len(col_rule)} rule(s))")
            for r in col_rule[:8]:
                rtype    = r.get("type", r.get("rule_type", "?"))
                severity = r.get("severity", "medium")
                detail   = ""
                regex_val = r.get("pattern") or r.get("regex")
                if regex_val:
                    rtype = "regex"
                    detail = f"  pattern: `{regex_val}`"
                elif rtype in ("min", "max", "range"):
                    lo = r.get("min", r.get("value", ""))
                    hi = r.get("max", "")
                    detail = f"  [{lo} – {hi}]" if hi else f"  min={lo}"
                elif rtype in ("not_null", "required"):
                    detail = "  (required — must not be empty)"
                elif rtype in ("type_numeric", "type_date"):
                    detail = f"  (column must be {rtype.replace('type_', '')})"
                elif rtype == "allowed_values" and "values" in r:
                    vals = r["values"]
                    sample = ", ".join(str(v) for v in vals[:6])
                    detail = f"  allowed: {sample}" + (" ..." if len(vals) > 6 else "")
                if not detail:
                    extras = {k: v for k, v in r.items() if k not in ("type", "rule_type", "severity", "metadata")}
                    if extras:
                        detail = "  " + ", ".join(f"{k}={v}" for k, v in list(extras.items())[:3])
                lines.append(f"  [{severity}] {rtype}{detail}")
        if isinstance(col_rule, list) and len(col_rule) > 8:
            lines.append(f"  ... and {len(col_rule) - 8} more rules")

    lines.append(
        "\nRules are now active. Ask 'run validation' to apply them, or 'explain the phone rule' for details on any rule."
    )
    return "\n".join(lines)


def handle_check_corpus(column: Optional[str], state: Dict[str, Any]) -> str:
    """Run corpus/canonical value validation against the loaded dataset."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."

    cm = _ensure_corpus_manager(state)
    if cm is None:
        return (
            "Redis is not running — corpus validation requires a Redis server.\n\n"
            "Start it with:\n"
            "  redis-server\n"
            "Or on macOS with Homebrew:\n"
            "  brew services start redis\n\n"
            "Then try again. If Redis is running but corpora aren't loaded, "
            "open the Corpus Manager panel to load reference data."
        )

    try:
        corpora = cm.list_corpora()
    except Exception as e:
        return f"Could not connect to corpus data: {e}"

    if not corpora:
        return (
            "No corpora loaded in Redis yet.\n\n"
            "Opening the Corpus Manager panel — load reference data there "
            "(e.g. canonical company names, valid postcodes, email domains). "
            "Then ask me to run a corpus check again.",
            "Corpus Manager",
        )

    try:
        from core.validation_ui_helpers import auto_detect_corpus_mappings
        from core.corpus_validation import validate_all_corpus_mappings, validate_with_corpus

        corpus_list = [
            {"corpus_name": c["corpus_name"], "corpus_type": c.get("corpus_type", "alias")}
            for c in corpora
        ]

        if column and column.strip():
            # ── Check a specific column ────────────────────────────
            col = column.strip()
            mappings = auto_detect_corpus_mappings(df, corpus_list)
            col_mapping = mappings.get(col)

            if col_mapping:
                report = validate_with_corpus(
                    df, col,
                    col_mapping["corpus"], cm,
                    col_mapping.get("type", "alias"),
                )
            else:
                # Try direct name match: find corpus whose name contains the column name
                matching = [c for c in corpora if col.lower() in c["corpus_name"].lower()]
                if matching:
                    report = validate_with_corpus(
                        df, col,
                        matching[0]["corpus_name"], cm,
                        matching[0].get("corpus_type", "alias"),
                    )
                else:
                    corpus_names = ", ".join(c["corpus_name"] for c in corpora)
                    return (
                        f"No corpus mapping found for column '{col}'.\n\n"
                        f"Available corpora: {corpus_names}\n"
                        f"Dataset columns  : {', '.join(df.columns.tolist())}\n\n"
                        "Tip: Open the Corpus Manager panel to load a corpus for this column."
                    )
            checked_cols = [col]
        else:
            # ── Auto-detect and check all mappable columns ─────────
            mappings = auto_detect_corpus_mappings(df, corpus_list)
            if not mappings:
                corpus_names = ", ".join(c["corpus_name"] for c in corpora)
                return (
                    f"No automatic corpus mappings detected for any column.\n\n"
                    f"Loaded corpora   : {corpus_names}\n"
                    f"Dataset columns  : {', '.join(df.columns.tolist()[:10])}\n\n"
                    "Tip: Name your corpus to match the column it validates "
                    "(e.g. corpus 'email_domains' maps to an 'email' column). "
                    "Or ask me to check a specific column directly: "
                    "'check corpus for the company_name column'."
                )
            corpus_mappings = {
                col: {
                    "corpus_name": info["corpus"],
                    "corpus_type": info.get("type", "alias"),
                }
                for col, info in mappings.items()
            }
            report = validate_all_corpus_mappings(df, corpus_mappings, cm)
            checked_cols = list(mappings.keys())

        # ── Summarise results ──────────────────────────────────────
        if not isinstance(report, pd.DataFrame) or report.empty:
            return (
                f"Corpus validation complete — no issues found.\n"
                f"All values in {', '.join(checked_cols)} match the reference corpora."
            )

        # Normalise column names to match unified_issues_report schema
        if "issue_type" in report.columns and "issue" not in report.columns:
            report = report.rename(columns={"issue_type": "issue"})
        if "source" not in report.columns:
            report["source"] = "corpus"

        state["corpus_issues_report"] = report

        total        = len(report)
        cols_aff     = report["column"].nunique()   if "column"  in report.columns else "?"
        rows_aff     = report["row_id"].nunique()   if "row_id"  in report.columns else "?"
        fixable      = int(report["suggested_fix"].notna().sum()) if "suggested_fix" in report.columns else 0

        lines = [
            f"Corpus validation complete.",
            f"  Columns checked  : {', '.join(checked_cols)}",
            f"  Issues found     : {total:,}",
            f"  Columns affected : {cols_aff}",
            f"  Rows affected    : {rows_aff}",
            f"  Auto-fixable     : {fixable:,} (canonical replacements available)",
        ]

        if "column" in report.columns:
            lines.append("\nBreakdown by column:")
            for col_name, count in report["column"].value_counts().items():
                lines.append(f"  {col_name}: {count:,} issues")

        if "issue" in report.columns:
            lines.append("\nIssue types:")
            for itype, count in report["issue"].value_counts().items():
                lines.append(f"  {itype}: {count:,}")

        # Merge corpus results into unified_issues_report so apply_fixes picks them up
        existing = state.get("unified_issues_report")
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            if "source" in existing.columns:
                existing_no_corpus = existing[existing["source"] != "corpus"]
            else:
                existing_no_corpus = existing
            try:
                state["unified_issues_report"] = pd.concat(
                    [existing_no_corpus, report], ignore_index=True
                )
                lines.append("\nResults merged into validation report — ask me to 'apply fixes' to standardize values.")
            except Exception:
                lines.append("\nResults saved separately — open Validate & Fix to apply corrections.")
        else:
            state["unified_issues_report"] = report
            lines.append("\nResults saved — ask me to 'apply fixes' to apply canonical replacements.")

        return "\n".join(lines)

    except Exception as e:
        return f"Corpus check error: {e}"


# ─────────────────────────────────────────────────────────
# NEW HANDLERS — BERT, Vector, Pipeline, Scorecard
# ─────────────────────────────────────────────────────────

def handle_bert_validation(state: Dict[str, Any], explain_sample: int = 10) -> str:
    """Run BERT-enhanced validation with human-friendly explanations."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."
    try:
        from core.enhanced_validator import EnhancedValidator
        ev = EnhancedValidator(use_bert=True)
        rules = state.get("rules_merged") or {}
        report = ev.validate_with_explanations(
            df, rules=rules if rules else None,
            explain_all=False, explain_sample=explain_sample,
        )
        state["bert_report"] = report

        total = len(report)
        explained = report["explanation"].notna().sum() if "explanation" in report.columns else 0
        lines = [
            f"BERT-enhanced validation complete.",
            f"Total issues: {total:,}",
            f"Issues with AI explanations: {explained:,}",
            "",
        ]
        # Show explained issues
        if "explanation" in report.columns:
            explained_rows = report[report["explanation"].notna()].head(explain_sample)
            for _, row in explained_rows.iterrows():
                col = row.get("column", "?")
                val = row.get("value", "?")
                issue = row.get("issue", "?")
                expl = row.get("explanation", "")
                lines.append(f"Column: {col} | Value: {val} | Issue: {issue}")
                lines.append(f"  Explanation: {expl}")
                lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"BERT validation failed: {e}"


def handle_build_vector_index(column: str, collection_name: str, state: Dict[str, Any]) -> str:
    """Build a vector index from a dataset column."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {', '.join(df.columns.tolist())}"
    try:
        from core.vector_db_manager import VectorDBManager
        vdb = state.get("vector_db_manager")
        if vdb is None:
            vdb = VectorDBManager()
            state["vector_db_manager"] = vdb

        coll_name = collection_name or column
        values = df[column].dropna().astype(str).unique().tolist()

        vdb.create_collection(coll_name)
        vdb.add_to_collection(
            collection_name=coll_name,
            documents=values,
            ids=[f"doc_{i}" for i in range(len(values))],
        )
        return (
            f"Vector index '{coll_name}' built successfully.\n"
            f"Indexed {len(values):,} unique values from column '{column}'.\n"
            f"You can now search it with: search_vector_index"
        )
    except Exception as e:
        return f"Failed to build vector index: {e}"


def handle_search_vector_index(query: str, collection_name: str, n_results: int, state: Dict[str, Any]) -> str:
    """Search a vector index for similar values."""
    try:
        from core.vector_db_manager import VectorDBManager
        vdb = state.get("vector_db_manager")
        if vdb is None:
            vdb = VectorDBManager()
            state["vector_db_manager"] = vdb

        results = vdb.query_collection(
            collection_name=collection_name,
            query_texts=[query],
            n_results=n_results,
        )
        if not results or not results.get("documents"):
            return f"No results found in collection '{collection_name}' for query '{query}'."

        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results.get("distances") else []

        lines = [f"Top {len(docs)} results for '{query}' in '{collection_name}':", ""]
        for i, doc in enumerate(docs):
            dist = f" (distance: {distances[i]:.3f})" if i < len(distances) else ""
            lines.append(f"  {i+1}. {doc}{dist}")
        return "\n".join(lines)
    except Exception as e:
        return f"Vector search failed: {e}"


def handle_save_pipeline(name: str, steps: list, state: Dict[str, Any]) -> str:
    """Save a named pipeline for later re-use."""
    try:
        from core.pipeline_manager import PipelineManager

        pm = PipelineManager()
        default_steps = ["validate", "clean", "deduplicate", "score"]
        pipeline_steps = steps if steps else default_steps

        valid_steps = {"validate", "clean", "deduplicate", "corpus_standardize",
                       "column_trim", "entity_resolution", "score"}
        invalid = [s for s in pipeline_steps if s not in valid_steps]
        if invalid:
            return f"Invalid step(s): {', '.join(invalid)}. Valid: {', '.join(sorted(valid_steps))}"

        pipeline_config = {
            "name": name,
            "steps": [{"name": s, "enabled": True} for s in pipeline_steps],
        }
        pm.save_pipeline(pipeline_config)
        return (
            f"Pipeline '{name}' saved with {len(pipeline_steps)} steps:\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(pipeline_steps))
            + "\n\nRun it anytime with: 'run pipeline " + name + "'"
        )
    except Exception as e:
        return f"Failed to save pipeline: {e}"


def handle_scorecard(state: Dict[str, Any]) -> str:
    """Get detailed quality scorecard with per-column breakdown."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please load data first."
    try:
        from core.quality_scores import compute_quality_scores
        from core.row_quality_scorer import score_rows

        scores = compute_quality_scores(df)
        lines = ["QUALITY SCORECARD", "=" * 50, ""]

        # Overall dimension scores
        lines.append("Overall Dimension Scores:")
        total = 0
        count = 0
        for dim, score in scores.items():
            if isinstance(score, (int, float)):
                filled = int(score / 10)
                bar = "█" * filled + "░" * (10 - filled)
                lines.append(f"  {dim.capitalize():<20} {bar}  {score:.1f}/100")
                total += score
                count += 1
        if count:
            lines.append(f"\n  {'OVERALL':<20} {'':>10}  {total/count:.1f}/100")

        # Per-column completeness
        lines.append("\nPer-Column Completeness:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            filled = int(pct / 10)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"  {col:<25} {bar}  {pct:.0f}%")

        # Before/after if available
        auto_scores = state.get("auto_scores")
        if auto_scores and "before" in auto_scores and "after" in auto_scores:
            lines.append("\nBefore vs After Cleaning:")
            before = auto_scores["before"]
            after = auto_scores["after"]
            for dim in before:
                if isinstance(before.get(dim), (int, float)) and isinstance(after.get(dim), (int, float)):
                    diff = after[dim] - before[dim]
                    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                    lines.append(f"  {dim.capitalize():<20} {before[dim]:.1f} → {after[dim]:.1f}  {arrow}{abs(diff):.1f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Could not compute scorecard: {e}"


def handle_edit_rule(action: str, column: str, rule_type: str, pattern: str,
                     severity: str, min_value, max_value, state: Dict[str, Any]) -> str:
    """Add, modify, or delete a validation rule."""
    rules = state.get("rules_merged") or {}
    if "columns" not in rules:
        rules["columns"] = {}
    col_rules = rules["columns"]

    if action == "delete":
        if column not in col_rules:
            return f"No rules found for column '{column}'. Nothing to delete."
        if rule_type:
            # Delete specific rule type
            existing = col_rules[column]
            if isinstance(existing, list):
                col_rules[column] = [r for r in existing if r.get("rule_type") != rule_type]
                if not col_rules[column]:
                    del col_rules[column]
            elif isinstance(existing, dict) and existing.get("rule_type") == rule_type:
                del col_rules[column]
            else:
                del col_rules[column]
        else:
            del col_rules[column]
        state["rules_merged"] = rules
        return f"Rule(s) deleted for column '{column}' (type: {rule_type or 'all'})."

    elif action in ("add", "modify"):
        new_rule = {"severity": severity or "medium"}
        if rule_type:
            new_rule["rule_type"] = rule_type
        if pattern:
            new_rule["regex"] = pattern
            if not rule_type:
                new_rule["rule_type"] = "regex"
        if min_value is not None:
            new_rule["min"] = min_value
        if max_value is not None:
            new_rule["max"] = max_value
        if min_value is not None or max_value is not None:
            if not rule_type:
                new_rule["rule_type"] = "range"

        if action == "modify" and column in col_rules:
            existing = col_rules[column]
            if isinstance(existing, list):
                # Replace rule of same type or append
                replaced = False
                for i, r in enumerate(existing):
                    if r.get("rule_type") == new_rule.get("rule_type"):
                        existing[i] = new_rule
                        replaced = True
                        break
                if not replaced:
                    existing.append(new_rule)
            else:
                col_rules[column] = [new_rule]
        else:
            if column in col_rules:
                existing = col_rules[column]
                if isinstance(existing, list):
                    existing.append(new_rule)
                else:
                    col_rules[column] = [existing, new_rule]
            else:
                col_rules[column] = [new_rule]

        state["rules_merged"] = rules
        action_word = "Modified" if action == "modify" else "Added"
        return (
            f"{action_word} rule for column '{column}':\n"
            f"  Type: {new_rule.get('rule_type', 'custom')}\n"
            f"  Pattern: {new_rule.get('regex', 'N/A')}\n"
            f"  Severity: {new_rule.get('severity', 'medium')}\n"
            + (f"  Range: {new_rule.get('min', '')} - {new_rule.get('max', '')}\n" if min_value is not None or max_value is not None else "")
            + f"\nRules updated. Run 'list rules for {column}' to see all rules."
        )
    else:
        return f"Unknown action '{action}'. Use 'add', 'modify', or 'delete'."


def handle_review_fixes(column: str, start_row: int, limit: int, state: Dict[str, Any]) -> str:
    """Show pending fixes for row-by-row review."""
    auto_note = _ensure_validation(state)
    report = state.get("unified_issues_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "No validation report available. Run validation first."
    if "suggested_fix" not in report.columns:
        return auto_note + "No fix suggestions in the validation report."

    fixable = report[report["suggested_fix"].notna()].copy()
    if column:
        fixable = fixable[fixable["column"] == column]
    if fixable.empty:
        return auto_note + f"No fixable issues found{' for column ' + column if column else ''}."

    total = len(fixable)
    subset = fixable.iloc[start_row:start_row + limit]

    lines = [
        auto_note + f"Pending fixes: showing {len(subset)} of {total} "
        f"(rows {start_row}-{start_row + len(subset) - 1})",
        "",
    ]
    for idx, (_, row) in enumerate(subset.iterrows()):
        row_id = row.get("row_id", "?")
        col = row.get("column", "?")
        current = row.get("value", "?")
        fix = row.get("suggested_fix", "?")
        issue = row.get("issue", "?")
        confidence = row.get("confidence", "?")
        lines.append(
            f"  Fix #{start_row + idx + 1}: Row {row_id}, {col}\n"
            f"    Current : {current}\n"
            f"    Fix to  : {fix}\n"
            f"    Issue   : {issue}\n"
            f"    Confidence: {confidence}\n"
        )

    lines.append(
        "To apply a specific fix, say: 'edit row X column_name to new_value'\n"
        "To apply all fixes: 'apply fixes'\n"
        f"To see more: 'review fixes from row {start_row + limit}'"
    )
    return "\n".join(lines)


def handle_map_corpus(column: str, corpus_name: str, state: Dict[str, Any]) -> str:
    """Map a dataset column to a loaded corpus."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded."
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {', '.join(df.columns.tolist())}"

    corpus_mgr = state.get("corpus_manager")
    if corpus_mgr is None:
        return "No corpus manager available. Start Redis and reload the app."

    corpora_list = corpus_mgr.list_corpora()
    corpus_names = [c.get("corpus_name", "") for c in corpora_list]
    if corpus_name not in corpus_names:
        return f"Corpus '{corpus_name}' not found. Available: {', '.join(corpus_names)}"

    # Find corpus details
    corpus_info = next(c for c in corpora_list if c.get("corpus_name") == corpus_name)
    corpus_type = corpus_info.get("corpus_type", "validation")

    # Store mapping
    if "corpus_mappings" not in state:
        state["corpus_mappings"] = {}
    state["corpus_mappings"][column] = {
        "corpus": corpus_name,
        "type": corpus_type,
    }

    return (
        f"Column '{column}' mapped to corpus '{corpus_name}' (type: {corpus_type}).\n"
        f"This mapping will be used when you run 'check corpus' or 'apply corpus standardization'.\n"
        f"\nCurrent mappings:\n"
        + "\n".join(f"  {col} -> {m['corpus']} ({m['type']})"
                    for col, m in state.get("corpus_mappings", {}).items())
    )


def handle_configure_validation(ml_models, contamination, bert_model, use_api, state: Dict[str, Any]) -> str:
    """Configure advanced validation parameters."""
    config = state.get("validation_config", {})
    changes = []

    if ml_models is not None:
        valid_models = {"isolation_forest", "lof", "one_class_svm"}
        invalid = [m for m in ml_models if m not in valid_models]
        if invalid:
            return f"Invalid ML model(s): {', '.join(invalid)}. Valid: {', '.join(sorted(valid_models))}"
        config["ml_models"] = ml_models
        changes.append(f"ML models: {', '.join(ml_models)}")

    if contamination is not None:
        if not 0.01 <= contamination <= 0.5:
            return f"Contamination must be between 0.01 and 0.5. Got: {contamination}"
        config["contamination"] = contamination
        changes.append(f"Contamination rate: {contamination}")

    if bert_model is not None:
        config["bert_model"] = bert_model
        changes.append(f"BERT model: {bert_model}")

    if use_api is not None:
        config["use_api"] = use_api
        changes.append(f"API validation: {'enabled' if use_api else 'disabled'}")

    state["validation_config"] = config

    if not changes:
        # Show current config
        lines = ["Current validation configuration:"]
        lines.append(f"  ML models: {config.get('ml_models', ['isolation_forest', 'lof', 'one_class_svm'])}")
        lines.append(f"  Contamination: {config.get('contamination', 0.05)}")
        lines.append(f"  BERT model: {config.get('bert_model', 'all-MiniLM-L6-v2')}")
        lines.append(f"  API validation: {config.get('use_api', False)}")
        return "\n".join(lines)

    return "Validation configuration updated:\n" + "\n".join(f"  {c}" for c in changes) + "\n\nThese settings will be used on the next 'run validation' or 'run BERT validation'."


def handle_show_entity_graph(max_clusters: int, state: Dict[str, Any]) -> str:
    """Show entity resolution results as text graph."""
    er_pairs = state.get("er_pairs")
    if not isinstance(er_pairs, pd.DataFrame) or er_pairs.empty:
        return "No entity resolution results found. Run 'run entity resolution' first."

    try:
        from core.entity_resolution import analyze_clusters
        analysis = analyze_clusters(er_pairs)

        clusters = analysis.get("clusters", {})
        lines = [
            f"Entity Resolution Graph Summary",
            f"=" * 45,
            f"Total matched pairs: {len(er_pairs):,}",
            f"Total clusters: {len(clusters)}",
            "",
        ]

        avg_score = er_pairs["score"].mean() if "score" in er_pairs.columns else 0
        lines.append(f"Average match score: {avg_score:.3f}")
        lines.append("")

        for i, (cluster_id, members) in enumerate(list(clusters.items())[:max_clusters]):
            lines.append(f"Cluster {cluster_id} ({len(members)} members):")
            for member in members[:5]:
                lines.append(f"  - {member}")
            if len(members) > 5:
                lines.append(f"  ... and {len(members) - 5} more")
            lines.append("")

        if len(clusters) > max_clusters:
            lines.append(f"... and {len(clusters) - max_clusters} more clusters")

        lines.append("\nFor interactive visualization, say 'open entity graph'.")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not display entity graph: {e}"


# ─────────────────────────────────────────────────────────
# Agentic Workflow Handlers
# ─────────────────────────────────────────────────────────

def handle_get_batch_history(state: Dict[str, Any], last_n: int = 10, column: str = "") -> str:
    """Get historical batch quality reports from database."""
    try:
        import sqlite3
        db_path = "./output/dq_investigator.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT batch_id, timestamp, rows_processed, overall_score, pass, issues_count "
            "FROM batch_runs ORDER BY timestamp DESC LIMIT ?",
            (min(last_n, 50),),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No batch history found in the database. Run validation first to build history."

        lines = ["**Batch History** (most recent first)\n"]
        lines.append("| Batch | Timestamp | Rows | Score | Pass | Issues |")
        lines.append("|---|---|---|---|---|---|")
        for batch_id, ts, rows_proc, score, passed, issues in rows:
            pass_str = "PASS" if passed else "FAIL"
            lines.append(f"| {batch_id} | {ts[:19]} | {rows_proc} | {score:.1f}% | {pass_str} | {issues} |")

        # Trend analysis
        if len(rows) >= 2:
            latest_score = rows[0][3]
            prev_score = rows[1][3]
            delta = latest_score - prev_score
            trend = "improved" if delta > 0 else "declined" if delta < 0 else "unchanged"
            lines.append(f"\n**Trend:** Score {trend} by {abs(delta):.1f}% from previous batch.")

            # Check for sudden drops
            scores = [r[3] for r in rows]
            for i in range(len(scores) - 1):
                if scores[i] - scores[i + 1] > 15:
                    lines.append(
                        f"\n**Alert:** Significant drop of {scores[i] - scores[i+1]:.1f}% "
                        f"between {rows[i+1][0]} and {rows[i][0]}."
                    )
                    break

        return "\n".join(lines)
    except Exception as e:
        return f"Could not retrieve batch history: {e}"


def handle_compare_batches(state: Dict[str, Any], batch_a: str = "latest", batch_b: str = "previous") -> str:
    """Compare two batch validation results."""
    try:
        import sqlite3
        db_path = "./output/dq_investigator.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Resolve 'latest' and 'previous'
        cursor.execute("SELECT batch_id, overall_score, issues_count, rows_processed, timestamp FROM batch_runs ORDER BY timestamp DESC LIMIT 2")
        recent = cursor.fetchall()

        if len(recent) < 2:
            conn.close()
            return "Need at least 2 batch runs to compare. Only found " + str(len(recent)) + " batch(es)."

        if batch_a == "latest":
            a = recent[0]
        else:
            cursor.execute("SELECT batch_id, overall_score, issues_count, rows_processed, timestamp FROM batch_runs WHERE batch_id = ?", (batch_a,))
            a = cursor.fetchone()

        if batch_b == "previous":
            b = recent[1]
        else:
            cursor.execute("SELECT batch_id, overall_score, issues_count, rows_processed, timestamp FROM batch_runs WHERE batch_id = ?", (batch_b,))
            b = cursor.fetchone()

        if not a or not b:
            conn.close()
            return "Could not find one or both batches in the database."

        # Get issue breakdowns
        def _get_issues(bid):
            cursor.execute(
                "SELECT column_name, COUNT(*) as cnt FROM issues WHERE batch_id = ? GROUP BY column_name ORDER BY cnt DESC LIMIT 10",
                (bid,),
            )
            return cursor.fetchall()

        issues_a = _get_issues(a[0])
        issues_b = _get_issues(b[0])
        conn.close()

        lines = [f"**Batch Comparison**\n"]
        lines.append(f"| Metric | {b[0]} | {a[0]} | Change |")
        lines.append("|---|---|---|---|")

        score_delta = a[1] - b[1]
        issue_delta = a[2] - b[2]
        lines.append(f"| Quality Score | {b[1]:.1f}% | {a[1]:.1f}% | {score_delta:+.1f}% |")
        lines.append(f"| Total Issues | {b[2]} | {a[2]} | {issue_delta:+d} |")
        lines.append(f"| Rows Processed | {b[3]} | {a[3]} | {a[3] - b[3]:+d} |")

        if score_delta < -10:
            lines.append(f"\n**Significant decline** of {abs(score_delta):.1f}% detected.")
        elif score_delta > 10:
            lines.append(f"\n**Significant improvement** of {score_delta:.1f}%.")

        # Column-level comparison
        cols_a = {col: cnt for col, cnt in issues_a}
        cols_b = {col: cnt for col, cnt in issues_b}
        all_cols = set(list(cols_a.keys()) + list(cols_b.keys()))

        if all_cols:
            lines.append(f"\n**Issues by Column:**")
            lines.append(f"| Column | Before | After | Change |")
            lines.append("|---|---|---|---|")
            for col in sorted(all_cols, key=lambda c: abs(cols_a.get(c, 0) - cols_b.get(c, 0)), reverse=True)[:10]:
                before = cols_b.get(col, 0)
                after = cols_a.get(col, 0)
                delta = after - before
                lines.append(f"| {col} | {before} | {after} | {delta:+d} |")

        return "\n".join(lines)
    except Exception as e:
        return f"Could not compare batches: {e}"


def handle_save_results_to_s3(state: Dict[str, Any], include_corrected: bool = True) -> str:
    """Save current results to S3."""
    s3_bucket = state.get("s3_output_bucket", "")
    s3_region = state.get("s3_region", "us-east-1")

    if not s3_bucket:
        return "No S3 output bucket configured. Go to Settings → S3 connector tab and set an output bucket."

    try:
        from core.storage.s3 import write_csv_to_s3, write_json_to_s3
        from datetime import datetime, timezone
        import numpy as np

        def _ser(obj):
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        paths = []

        # Save issue report
        report = state.get("unified_issues_report")
        if isinstance(report, pd.DataFrame) and not report.empty:
            uri = write_csv_to_s3(report, s3_bucket, f"reports/{batch_id}_issues.csv", s3_region)
            paths.append(f"Issues: {uri}")

        # Save corrected dataset
        if include_corrected:
            df_corrected = state.get("df_corrected") or state.get("df_cleaned") or state.get("df_raw")
            if df_corrected is not None:
                uri = write_csv_to_s3(df_corrected, s3_bucket, f"corrected/{batch_id}_corrected.csv", s3_region)
                paths.append(f"Corrected: {uri}")

        # Save quality report
        result = state.get("validation_result")
        if result:
            df_raw = state.get("df_raw")
            total_cells = len(df_raw) * len(df_raw.columns) if df_raw is not None else 1
            report_data = {
                "batch_id": batch_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "issues_count": result.total_issues,
                "overall_score": round(100 - (result.total_issues / max(total_cells, 1) * 100), 2),
                "step_timings": {k: round(v, 3) for k, v in result.step_timings.items()},
            }
            uri = write_json_to_s3(report_data, s3_bucket, f"reports/{batch_id}_report.json", s3_region, _ser)
            paths.append(f"Report: {uri}")

        if paths:
            return "**Results saved to S3:**\n" + "\n".join(f"- {p}" for p in paths)
        else:
            return "No results to save. Run validation first."
    except Exception as e:
        return f"Failed to save to S3: {e}"


def handle_investigate(state: Dict[str, Any], auto_fix: bool = False) -> str:
    """Run a full autonomous investigation pipeline."""
    lines = ["**Autonomous Investigation Report**\n"]
    lines.append("---\n")

    df_raw = state.get("df_raw")
    if df_raw is None:
        return "No dataset loaded. Go to Load Data first."

    lines.append(f"**Dataset:** {len(df_raw)} rows x {len(df_raw.columns)} columns\n")

    # Step 1: Run validation
    lines.append("### Step 1: Validation")
    val_result = handle_run_validation(state, use_ml=False)
    lines.append(val_result)
    lines.append("")

    result = state.get("validation_result")
    if not result:
        lines.append("Validation did not produce results. Cannot continue investigation.")
        return "\n".join(lines)

    # Step 2: Identify worst columns
    lines.append("### Step 2: Worst Columns")
    report = state.get("unified_issues_report")
    if isinstance(report, pd.DataFrame) and not report.empty and "column" in report.columns:
        col_counts = report["column"].value_counts().head(5)
        lines.append("| Column | Issues |")
        lines.append("|---|---|")
        for col, cnt in col_counts.items():
            lines.append(f"| {col} | {cnt} |")
        worst_col = col_counts.index[0] if len(col_counts) > 0 else None
    else:
        lines.append("No column-level issues found.")
        worst_col = None
    lines.append("")

    # Step 3: Profile worst column
    if worst_col:
        lines.append(f"### Step 3: Deep Profile — {worst_col}")
        profile_result = handle_get_profile(worst_col, state)
        lines.append(profile_result)
        lines.append("")

    # Step 4: Check batch history
    lines.append("### Step 4: Batch History")
    history_result = handle_get_batch_history(state, last_n=5)
    lines.append(history_result)
    lines.append("")

    # Step 5: Fixable issues
    lines.append("### Step 5: Fixable Issues")
    fixable_result = handle_fixable(state)
    lines.append(fixable_result)
    lines.append("")

    # Step 6: Summary and recommendation
    total_issues = result.total_issues
    total_cells = len(df_raw) * len(df_raw.columns)
    score = 100 - (total_issues / max(total_cells, 1) * 100)

    lines.append("### Step 6: Summary")
    lines.append(f"- **Quality Score:** {score:.1f}%")
    lines.append(f"- **Total Issues:** {total_issues}")
    lines.append(f"- **Pass Threshold:** 85%")
    lines.append(f"- **Status:** {'PASS' if score >= 85 else 'FAIL'}")
    lines.append("")

    if score < 85:
        lines.append("**Recommendation:** Apply high-confidence fixes to improve the score. "
                      "Type `apply fixes` or `yes` to proceed.")
        if auto_fix:
            lines.append("\n### Auto-fix Applied")
            fix_result = handle_apply_fixes("high_confidence", state)
            lines.append(fix_result)
    else:
        lines.append("**Recommendation:** Data quality is acceptable. Safe to proceed with downstream use.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────────────────

_DESTRUCTIVE_TOOLS = {"apply_fixes", "merge_duplicates", "edit_cell"}

def _dispatch_tool(
    fn_name: str,
    fn_args: Dict[str, Any],
    state: Dict[str, Any],
    force: bool = False,
) -> Tuple[str, Optional[str]]:
    """Execute a tool call. Returns (result_text, open_tool_name)."""
    open_tool_name = None

    # ── Confirmation gate for destructive tools ──
    if fn_name in _DESTRUCTIVE_TOOLS and not force:
        # Build a human-readable description of what will happen
        if fn_name == "apply_fixes":
            mode = fn_args.get("mode", "high_confidence")
            desc = f"Apply auto-fixes (mode: {mode}) to the dataset"
        elif fn_name == "merge_duplicates":
            strategy = fn_args.get("strategy", "keep_master")
            desc = f"Merge duplicate rows (strategy: {strategy})"
        elif fn_name == "edit_cell":
            row = fn_args.get("row", "?")
            col = fn_args.get("column", "?")
            val = fn_args.get("new_value", "?")
            desc = f"Edit row {row}, column \'{col}\' → \'{val}\'"
        else:
            desc = f"Execute {fn_name}"

        state["_pending_action"] = {"tool": fn_name, "args": fn_args}
        return (
            f"**Confirmation required:**\n\n"
            f"> {desc}\n\n"
            f"This will modify your data. Type **yes** or **confirm** to proceed, "
            f"or **no** / **cancel** to abort."
        ), None

    if fn_name == "run_duplicate_detection":
        dup_thr = float(fn_args.get("duplicate_threshold", 0.85))
        sim_thr = float(fn_args.get("similar_threshold", 0.6))
        result = handle_run_duplicate_detection(state, duplicate_threshold=dup_thr, similar_threshold=sim_thr)
        return result, None
    elif fn_name == "run_validation":
        _use_ml = bool(fn_args.get("use_ml", False))
        result = handle_run_validation(state, use_ml=_use_ml)
        return result, None
    elif fn_name == "get_duplicates":
        result, open_tool_name = handle_duplicates(state)
        return result, open_tool_name
    elif fn_name == "get_summary":
        result = handle_summary(state)
    elif fn_name == "count_issues":
        result = handle_count_issues(str(fn_args.get("issue_type", "all")), state)
    elif fn_name == "get_worst_rows":
        result = handle_worst_rows(state, n=int(fn_args.get("n", 10)))
    elif fn_name == "explain_row":
        result = handle_explain_row(int(fn_args.get("row_id", 0)), state)
    elif fn_name == "get_column_issues":
        result = handle_column_issues(str(fn_args.get("column", "")), state)
    elif fn_name == "get_quality_score":
        result = handle_quality_score(state)
    elif fn_name == "get_fixable_issues":
        result = handle_fixable(state)
    elif fn_name == "show_data":
        result = handle_show_data(int(fn_args.get("n", 10)), state)
    elif fn_name == "get_row":
        result = handle_get_row(int(fn_args.get("row_number", 1)), state)
    elif fn_name == "lookup_value":
        result = handle_lookup_value(
            search_value=str(fn_args.get("search_value", "")),
            return_field=str(fn_args.get("return_field", "")),
            search_column=str(fn_args.get("search_column", "")),
            state=state,
        )
    elif fn_name == "filter_rows":
        result = handle_filter_rows(
            condition=str(fn_args.get("condition", "missing")),
            column=str(fn_args.get("column", "")),
            value=str(fn_args.get("value", "")),
            state=state,
        )
    elif fn_name == "open_tool":
        open_tool_name = str(fn_args.get("tool_name", ""))
        result = f"Opening {open_tool_name} panel..."
    # ── New action tools ───────────────────────────────────
    elif fn_name == "run_data_profiling":
        result = handle_run_data_profiling(state)
    elif fn_name == "generate_rules":
        result = handle_generate_rules(state)
    elif fn_name == "apply_fixes":
        mode = str(fn_args.get("mode", "high_confidence"))
        result = handle_apply_fixes(mode, state)
    elif fn_name == "run_basic_cleaning":
        result = handle_run_basic_cleaning(
            lowercase=bool(fn_args.get("lowercase", True)),
            strip_whitespace=bool(fn_args.get("strip_whitespace", True)),
            normalize_punctuation=bool(fn_args.get("normalize_punctuation", True)),
            state=state,
        )
    elif fn_name == "apply_column_trimming":
        result = handle_apply_column_trimming(
            missing_threshold=float(fn_args.get("missing_threshold", 0.8)),
            correlation_threshold=float(fn_args.get("correlation_threshold", 0.9)),
            state=state,
        )
    elif fn_name == "merge_duplicates":
        strategy = str(fn_args.get("strategy", "keep_master"))
        result = handle_merge_duplicates(strategy, state)
    elif fn_name == "run_auto_report":
        result = handle_run_auto_report(state)
    elif fn_name == "execute_pipeline":
        pipeline_name = str(fn_args.get("pipeline_name", ""))
        result = handle_execute_pipeline(pipeline_name, state)
    elif fn_name == "run_entity_resolution":
        threshold = float(fn_args.get("threshold", 0.8))
        result = handle_run_entity_resolution(threshold, state)
    elif fn_name == "get_profile":
        col = fn_args.get("column") or ""
        result = handle_get_profile(col if col else None, state)
    elif fn_name == "recommend_next_steps":
        result = handle_recommend_next_steps(state)
    elif fn_name == "apply_corpus_standardization":
        _r = handle_apply_corpus_standardization(state)
        if isinstance(_r, tuple):
            result, open_tool_name = _r
        else:
            result = _r
    elif fn_name == "compare_versions":
        result = handle_compare_versions(state)
    elif fn_name == "list_corpora":
        _r = handle_list_corpora(state)
        if isinstance(_r, tuple):
            result, open_tool_name = _r
        else:
            result = _r
    elif fn_name == "list_rules":
        col = fn_args.get("column") or ""
        result = handle_list_rules(col if col else None, state)
    elif fn_name == "check_corpus":
        col = fn_args.get("column") or ""
        _r = handle_check_corpus(col if col else None, state)
        if isinstance(_r, tuple):
            result, open_tool_name = _r
        else:
            result = _r
    elif fn_name == "query_data":
        expression = str(fn_args.get("expression", ""))
        result = handle_query_data(expression, state)
    elif fn_name == "edit_cell":
        result = handle_edit_cell(
            row=int(fn_args.get("row", 1)),
            column=str(fn_args.get("column", "")),
            new_value=str(fn_args.get("new_value", "")),
            state=state,
            end_row=int(fn_args["end_row"]) if fn_args.get("end_row") else None,
        )
    elif fn_name == "download_data":
        result = handle_download_data(
            version=str(fn_args.get("version", "corrected")),
            state=state,
        )
    elif fn_name == "create_chart":
        config = {k: v for k, v in fn_args.items() if k != "chart_type"}
        result = handle_create_chart(
            chart_type=str(fn_args.get("chart_type", "")),
            config=config,
            state=state,
        )
    # ── New tools ──
    elif fn_name == "run_bert_validation":
        _n = int(fn_args.get("explain_sample", 10))
        result = handle_bert_validation(state, explain_sample=_n)
    elif fn_name == "build_vector_index":
        result = handle_build_vector_index(
            column=str(fn_args.get("column", "")),
            collection_name=str(fn_args.get("collection_name", "")),
            state=state,
        )
    elif fn_name == "search_vector_index":
        result = handle_search_vector_index(
            query=str(fn_args.get("query", "")),
            collection_name=str(fn_args.get("collection_name", "")),
            n_results=int(fn_args.get("n_results", 10)),
            state=state,
        )
    elif fn_name == "save_pipeline":
        result = handle_save_pipeline(
            name=str(fn_args.get("name", "")),
            steps=fn_args.get("steps", []),
            state=state,
        )
    elif fn_name == "get_scorecard":
        result = handle_scorecard(state)
    elif fn_name == "edit_rule":
        result = handle_edit_rule(
            action=str(fn_args.get("action", "")),
            column=str(fn_args.get("column", "")),
            rule_type=fn_args.get("rule_type"),
            pattern=fn_args.get("pattern"),
            severity=fn_args.get("severity"),
            min_value=fn_args.get("min_value"),
            max_value=fn_args.get("max_value"),
            state=state,
        )
    elif fn_name == "review_fixes":
        result = handle_review_fixes(
            column=fn_args.get("column"),
            start_row=int(fn_args.get("start_row", 0)),
            limit=int(fn_args.get("limit", 10)),
            state=state,
        )
    elif fn_name == "map_corpus":
        result = handle_map_corpus(
            column=str(fn_args.get("column", "")),
            corpus_name=str(fn_args.get("corpus_name", "")),
            state=state,
        )
    elif fn_name == "configure_validation":
        result = handle_configure_validation(
            ml_models=fn_args.get("ml_models"),
            contamination=fn_args.get("contamination"),
            bert_model=fn_args.get("bert_model"),
            use_api=fn_args.get("use_api"),
            state=state,
        )
    elif fn_name == "show_entity_graph":
        result = handle_show_entity_graph(
            max_clusters=int(fn_args.get("max_clusters", 10)),
            state=state,
        )
    # ── Agentic workflow tools ──
    elif fn_name == "get_batch_history":
        result = handle_get_batch_history(
            state=state,
            last_n=int(fn_args.get("last_n", 10)),
            column=str(fn_args.get("column", "")),
        )
    elif fn_name == "compare_batches":
        result = handle_compare_batches(
            state=state,
            batch_a=str(fn_args.get("batch_a", "latest")),
            batch_b=str(fn_args.get("batch_b", "previous")),
        )
    elif fn_name == "save_results_to_s3":
        result = handle_save_results_to_s3(
            state=state,
            include_corrected=bool(fn_args.get("include_corrected", True)),
        )
    elif fn_name == "investigate":
        result = handle_investigate(
            state=state,
            auto_fix=bool(fn_args.get("auto_fix", False)),
        )
    else:
        result = f"Unknown tool: {fn_name}"

    return result, open_tool_name


def _extract_tool_calls(response):
    """
    Extract tool calls from an Ollama response.
    Handles both object API (newer ollama library) and dict API (older).
    Returns list of (fn_name, fn_args) tuples.
    """
    # Get message from response
    if hasattr(response, "message"):
        msg = response.message
        raw_tool_calls = getattr(msg, "tool_calls", None)
    elif isinstance(response, dict):
        msg = response.get("message", {})
        if isinstance(msg, dict):
            raw_tool_calls = msg.get("tool_calls")
        else:
            raw_tool_calls = getattr(msg, "tool_calls", None)
    else:
        return []

    if not raw_tool_calls:
        return []

    parsed = []
    for tc in raw_tool_calls:
        try:
            if hasattr(tc, "function"):
                # Object API
                fn = tc.function
                fn_name = fn.name
                fn_args = fn.arguments if isinstance(fn.arguments, dict) else {}
            elif isinstance(tc, dict):
                # Dict API
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                fn_args = raw_args if isinstance(raw_args, dict) else {}
            else:
                continue
            parsed.append((fn_name, fn_args))
        except Exception:
            continue

    return parsed


def _get_response_content(response) -> str:
    """Extract text content from an Ollama response."""
    try:
        if hasattr(response, "message"):
            msg = response.message
            return getattr(msg, "content", "") or ""
        elif isinstance(response, dict):
            msg = response.get("message", {})
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            return getattr(msg, "content", "") or ""
    except Exception:
        pass
    return ""


def handle_query_data(expression: str, state: Dict[str, Any]) -> str:
    """Safely evaluate a pandas expression against the loaded dataframe."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please upload a file first."

    # Safety: block anything that could access the filesystem or execute code
    _BLOCKED = [
        "import ", "__", "open(", "os.", "sys.", "exec(", "eval(",
        "subprocess", "shutil", "pathlib", ".write", ".to_csv", ".to_excel",
        "globals(", "locals(", "vars(", "getattr(", "setattr(",
    ]
    expr_check = expression.lower().replace(" ", "")
    for blocked in _BLOCKED:
        if blocked.replace(" ", "") in expr_check:
            return f"Expression blocked for safety: '{blocked.strip()}' is not permitted."

    try:
        import numpy as np

        # Auto-coerce columns that look numeric but are stored as strings.
        # Threshold: if >50% of non-null values convert cleanly, treat as numeric.
        df_eval = df.copy()
        for col in df_eval.columns:
            if df_eval[col].dtype == object:
                converted = pd.to_numeric(df_eval[col], errors="coerce")
                non_null = df_eval[col].notna().sum()
                if non_null > 0 and converted.notna().sum() / non_null > 0.5:
                    df_eval[col] = converted

        from collections import Counter
        namespace = {
            "df": df_eval, "pd": pd, "np": np,
            # Safe builtins needed for common pandas expressions
            "str": str, "int": int, "float": float, "bool": bool,
            "len": len, "list": list, "dict": dict, "tuple": tuple,
            "range": range, "round": round, "abs": abs, "sum": sum,
            "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
            "print": print, "Counter": Counter,
        }
        namespace["__builtins__"] = {}
        result = eval(expression, namespace)  # noqa: S307

        # Format result sensibly
        if isinstance(result, pd.DataFrame):
            n = len(result)
            out = result.head(20).to_string()
            return out + (f"\n\n*Showing first 20 of {n:,} rows.*" if n > 20 else "")
        elif isinstance(result, pd.Series):
            n = len(result)
            out = result.head(20).to_string()
            return out + (f"\n\n*Showing first 20 of {n:,} entries.*" if n > 20 else "")
        else:
            return str(result)

    except Exception as e:
        return (
            f"Could not evaluate expression.\n"
            f"Expression: `{expression}`\n"
            f"Error: {e}"
        )


def handle_edit_cell(
    row: int,
    column: str,
    new_value: str,
    state: Dict[str, Any],
    end_row: Optional[int] = None,
) -> str:
    """Edit one or more cell values in the dataset."""
    df = state.get("df_raw")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "No dataset loaded. Please upload a file first."

    # Validate column exists (case-insensitive match)
    col_match = None
    for c in df.columns:
        if c.lower() == column.lower():
            col_match = c
            break
    if col_match is None:
        available = ", ".join(df.columns.tolist())
        return f"Column '{column}' not found.\nAvailable columns: {available}"

    # Convert 1-based row to 0-based index
    start_idx = row - 1
    end_idx = (end_row - 1) if end_row else start_idx

    # Validate row range
    if start_idx < 0 or end_idx >= len(df):
        return f"Row out of range. Dataset has {len(df)} rows (1 to {len(df)})."
    if end_idx < start_idx:
        return f"end_row ({end_row}) must be >= row ({row})."

    # Record old values for confirmation
    if start_idx == end_idx:
        old_val = df.at[df.index[start_idx], col_match]
        df.at[df.index[start_idx], col_match] = new_value
        state["df_raw"] = df
        # Show updated row for visual confirmation
        updated_row = df.iloc[start_idx].to_frame().T.to_string()
        return (
            f"Row {row}, column '{col_match}' updated.\n"
            f"  Old: {old_val}\n"
            f"  New: {new_value}\n\n"
            f"**Updated row {row}:**\n{updated_row}"
        )
    else:
        n_rows = end_idx - start_idx + 1
        old_vals = df.iloc[start_idx:end_idx + 1][col_match].tolist()
        for i in range(start_idx, end_idx + 1):
            df.at[df.index[i], col_match] = new_value
        state["df_raw"] = df

        # Show a sample of old values
        sample = old_vals[:5]
        sample_str = ", ".join(str(v) for v in sample)
        if len(old_vals) > 5:
            sample_str += f" ... ({len(old_vals) - 5} more)"

        # Show updated rows for visual confirmation
        updated_rows = df.iloc[start_idx:end_idx + 1].to_string()
        return (
            f"Rows {row}-{end_row}, column '{col_match}' updated ({n_rows} cells).\n"
            f"  Old values (sample): {sample_str}\n"
            f"  New value: {new_value}\n\n"
            f"**Updated rows:**\n{updated_rows}"
        )


def handle_create_chart(chart_type: str, config: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate a chart and store the image bytes in state for the UI to display."""
    import io
    import base64

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        return "matplotlib is not installed. Run: pip install matplotlib"

    df = state.get("df_raw")
    report = state.get("unified_issues_report")
    chart_type = (chart_type or "").lower().strip()

    # Color palette
    COLORS = ["#4C6EF5", "#F76707", "#37B24D", "#E03131", "#AE3EC9",
              "#1098AD", "#F59F00", "#D6336C", "#0CA678", "#845EF7"]
    SEVERITY_COLORS = {"high": "#E03131", "medium": "#F59F00", "low": "#1098AD"}
    BG_COLOR = "#0f1117"
    TEXT_COLOR = "#e0e8ff"
    GRID_COLOR = "#252a3a"

    def _style_chart(fig, ax):
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, alpha=0.2, color=GRID_COLOR)

    def _save_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        state["_chat_chart"] = buf.getvalue()
        return "Chart generated and displayed in the main panel."

    try:
        # ========== DATA QUALITY CHARTS ==========

        if chart_type == "issues_by_column":
            if not isinstance(report, pd.DataFrame) or report.empty:
                return "No validation report. Run validation first."
            counts = report["column"].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            _style_chart(fig, ax)
            bars = ax.barh(counts.index[::-1], counts.values[::-1], color=COLORS[0], edgecolor="none")
            ax.set_xlabel("Issue Count")
            ax.set_title("Issues by Column")
            for bar, val in zip(bars, counts.values[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", color=TEXT_COLOR, fontsize=9)
            return _save_chart(fig)

        elif chart_type == "issues_by_type":
            if not isinstance(report, pd.DataFrame) or report.empty:
                return "No validation report. Run validation first."
            counts = report["issue"].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            _style_chart(fig, ax)
            colors = [COLORS[i % len(COLORS)] for i in range(len(counts))]
            ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="none")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Count")
            ax.set_title("Issues by Type")
            return _save_chart(fig)

        elif chart_type == "issues_by_severity":
            if not isinstance(report, pd.DataFrame) or report.empty:
                return "No validation report. Run validation first."
            counts = report["severity"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor(BG_COLOR)
            colors = [SEVERITY_COLORS.get(s, COLORS[0]) for s in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, textprops={"color": TEXT_COLOR, "fontsize": 10})
            ax.set_title("Issues by Severity", color=TEXT_COLOR)
            return _save_chart(fig)

        elif chart_type == "missing_heatmap":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            missing = df.isnull().astype(int)
            fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.02)))
            _style_chart(fig, ax)
            cax = ax.imshow(missing.values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Row Index")
            ax.set_title("Missing Values Heatmap (red = missing)")
            fig.colorbar(cax, ax=ax, shrink=0.5, label="Missing")
            return _save_chart(fig)

        elif chart_type == "completeness_by_column":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            completeness = ((df.notna().sum() / len(df)) * 100).sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            _style_chart(fig, ax)
            colors = ["#E03131" if v < 80 else "#F59F00" if v < 95 else "#37B24D" for v in completeness.values]
            ax.barh(completeness.index, completeness.values, color=colors, edgecolor="none")
            ax.set_xlabel("Completeness %")
            ax.set_title("Data Completeness by Column")
            ax.set_xlim(0, 105)
            for i, v in enumerate(completeness.values):
                ax.text(v + 0.5, i, f"{v:.1f}%", va="center", color=TEXT_COLOR, fontsize=8)
            return _save_chart(fig)

        elif chart_type == "quality_score_breakdown":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            try:
                from core.quality_scores import compute_quality_scores
                scores = compute_quality_scores(df)
                dims = {k: v for k, v in scores.items()
                        if isinstance(v, (int, float)) and k != "overall"}
                if "overall" in scores:
                    dims["OVERALL"] = scores["overall"]
            except Exception as e:
                return f"Could not compute quality scores: {e}"
            fig, ax = plt.subplots(figsize=(8, 5))
            _style_chart(fig, ax)
            names = list(dims.keys())
            vals = list(dims.values())
            colors = ["#37B24D" if v >= 80 else "#F59F00" if v >= 60 else "#E03131" for v in vals]
            ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="none")
            ax.set_xlabel("Score (0-100)")
            ax.set_title("Data Quality Score Breakdown")
            ax.set_xlim(0, 105)
            for i, v in enumerate(vals[::-1]):
                ax.text(v + 0.5, i, f"{v:.1f}", va="center", color=TEXT_COLOR, fontsize=9)
            return _save_chart(fig)

        elif chart_type == "issues_severity_by_column":
            if not isinstance(report, pd.DataFrame) or report.empty:
                return "No validation report. Run validation first."
            pivot = report.groupby(["column", "severity"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 5))
            _style_chart(fig, ax)
            bottom = None
            for sev in ["high", "medium", "low"]:
                if sev in pivot.columns:
                    vals = pivot[sev].values
                    ax.bar(range(len(pivot)), vals, bottom=bottom or [0]*len(pivot),
                           label=sev, color=SEVERITY_COLORS[sev], edgecolor="none")
                    bottom = [b + v for b, v in zip(bottom or [0]*len(pivot), vals)]
            ax.set_xticks(range(len(pivot)))
            ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Issue Count")
            ax.set_title("Issues by Column & Severity")
            ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            return _save_chart(fig)

        elif chart_type == "top_issue_rows":
            if not isinstance(report, pd.DataFrame) or report.empty:
                return "No validation report. Run validation first."
            n = int(config.get("n", 15))
            row_counts = report["row_id"].value_counts().head(n)
            fig, ax = plt.subplots(figsize=(10, 5))
            _style_chart(fig, ax)
            ax.bar(range(len(row_counts)), row_counts.values, color=COLORS[3], edgecolor="none")
            ax.set_xticks(range(len(row_counts)))
            ax.set_xticklabels([f"Row {r}" for r in row_counts.index], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Issue Count")
            ax.set_title(f"Top {n} Rows by Issue Count")
            return _save_chart(fig)

        # ========== DATA EXPLORATION CHARTS ==========

        elif chart_type == "histogram":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            col = config.get("column", "")
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
            fig, ax = plt.subplots(figsize=(8, 5))
            _style_chart(fig, ax)
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            if data.empty:
                return f"Column '{col}' has no numeric data for a histogram."
            ax.hist(data, bins=int(config.get("bins", 30)), color=COLORS[0], edgecolor=BG_COLOR, alpha=0.9)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {col}")
            return _save_chart(fig)

        elif chart_type == "bar_chart":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            col = config.get("column", "")
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
            n = int(config.get("n", 10))
            counts = df[col].value_counts().head(n)
            fig, ax = plt.subplots(figsize=(10, 5))
            _style_chart(fig, ax)
            ax.barh(counts.index[::-1].astype(str), counts.values[::-1], color=COLORS[0], edgecolor="none")
            ax.set_xlabel("Count")
            ax.set_title(f"Top {n} Values in {col}")
            return _save_chart(fig)

        elif chart_type == "pie_chart":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            col = config.get("column", "")
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
            counts = df[col].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.patch.set_facecolor(BG_COLOR)
            colors = [COLORS[i % len(COLORS)] for i in range(len(counts))]
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index.astype(str), autopct="%1.1f%%",
                colors=colors, textprops={"color": TEXT_COLOR, "fontsize": 9})
            ax.set_title(f"Distribution of {col}", color=TEXT_COLOR)
            return _save_chart(fig)

        elif chart_type == "scatter":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            x_col = config.get("x", "")
            y_col = config.get("y", "")
            if x_col not in df.columns or y_col not in df.columns:
                return f"Column(s) not found. Available: {', '.join(df.columns)}"
            fig, ax = plt.subplots(figsize=(8, 6))
            _style_chart(fig, ax)
            x_data = pd.to_numeric(df[x_col], errors="coerce")
            y_data = pd.to_numeric(df[y_col], errors="coerce")
            mask = x_data.notna() & y_data.notna()
            ax.scatter(x_data[mask], y_data[mask], c=COLORS[0], alpha=0.6, s=20, edgecolors="none")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            return _save_chart(fig)

        elif chart_type == "box_plot":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            col = config.get("column", "")
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
            fig, ax = plt.subplots(figsize=(6, 5))
            _style_chart(fig, ax)
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                           boxprops=dict(facecolor=COLORS[0], color=TEXT_COLOR),
                           medianprops=dict(color="#F59F00", linewidth=2),
                           whiskerprops=dict(color=TEXT_COLOR),
                           capprops=dict(color=TEXT_COLOR),
                           flierprops=dict(marker="o", markerfacecolor=COLORS[3], markersize=4))
            ax.set_xticklabels([col])
            ax.set_title(f"Box Plot of {col}")
            return _save_chart(fig)

        elif chart_type == "correlation_matrix":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return "Not enough numeric columns for a correlation matrix."
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(8, 7))
            _style_chart(fig, ax)
            cax = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_yticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                            color="white" if abs(corr.values[i,j]) > 0.5 else TEXT_COLOR, fontsize=8)
            ax.set_title("Correlation Matrix")
            fig.colorbar(cax, ax=ax, shrink=0.8)
            return _save_chart(fig)

        elif chart_type == "time_series":
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "No dataset loaded."
            date_col = config.get("date_column", "")
            value_col = config.get("value_column", "")
            if date_col not in df.columns:
                return f"Date column '{date_col}' not found. Available: {', '.join(df.columns)}"
            fig, ax = plt.subplots(figsize=(10, 5))
            _style_chart(fig, ax)
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if value_col and value_col in df.columns:
                vals = pd.to_numeric(df[value_col], errors="coerce")
                mask = dates.notna() & vals.notna()
                sorted_idx = dates[mask].argsort()
                ax.plot(dates[mask].iloc[sorted_idx], vals[mask].iloc[sorted_idx],
                        color=COLORS[0], linewidth=1.5)
                ax.set_ylabel(value_col)
                ax.set_title(f"{value_col} over {date_col}")
            else:
                counts = dates.dt.to_period("D").value_counts().sort_index()
                ax.bar(range(len(counts)), counts.values, color=COLORS[0], edgecolor="none")
                ax.set_xticks(range(0, len(counts), max(1, len(counts)//10)))
                ax.set_xticklabels([str(p) for p in counts.index[::max(1, len(counts)//10)]],
                                   rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Count")
                ax.set_title(f"Records over {date_col}")
            ax.set_xlabel(date_col)
            return _save_chart(fig)

        else:
            available = (
                "Data Quality: issues_by_column, issues_by_type, issues_by_severity, "
                "missing_heatmap, completeness_by_column, quality_score_breakdown, "
                "issues_severity_by_column, top_issue_rows\n"
                "Data Exploration: histogram, bar_chart, pie_chart, scatter, "
                "box_plot, correlation_matrix, time_series"
            )
            return f"Unknown chart type '{chart_type}'.\nAvailable charts:\n{available}"

    except Exception as e:
        return f"Chart generation failed: {e}"


def handle_download_data(version: str, state: Dict[str, Any]) -> str:
    """Prepare dataset for download — returns a special marker the UI picks up."""
    version = (version or "corrected").lower().strip()
    _VERSIONS = {
        "raw": "df_raw",
        "corrected": "df_corrected",
        "cleaned": "df_cleaned",
    }
    key = _VERSIONS.get(version, "df_corrected")
    df = state.get(key)
    if not isinstance(df, pd.DataFrame) or df.empty:
        # Fallback to raw
        df = state.get("df_raw")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No dataset loaded. Please upload a file first."
        version = "raw"

    csv_str = df.to_csv(index=False)
    # Store CSV in state so the UI can create a download button
    state["_chat_download"] = {
        "csv": csv_str,
        "version": version,
        "rows": len(df),
        "cols": len(df.columns),
    }
    return (
        f"**{version.title()} dataset ready for download** ({len(df):,} rows x {len(df.columns)} columns).\n\n"
        f"A download button has appeared in the main area."
    )


# ─────────────────────────────────────────────────────────
# Ollama Calls
# ─────────────────────────────────────────────────────────

def _build_messages(
    user_message: str,
    context: str,
    history: List[Dict[str, str]],
):
    system_prompt = (
        "You are an expert data quality analyst assistant embedded in a data quality pipeline tool.\n"
        "You help users understand validation results, explain data issues, and recommend actions.\n"
        "Use the available tools to answer data-specific questions. For general conversation, respond directly.\n"
        "Be concise and practical. Never make up data — always use the tools to get real information.\n\n"
        "RESPONSE FORMATTING (STRICT):\n"
        "- NEVER use emojis in your responses. No unicode symbols like checkmarks or warning signs.\n"
        "- Use professional, business-style formatting only.\n"
        "- Use markdown tables for structured data. Use bullet points for lists.\n"
        "- Use bold for emphasis. Use code backticks for column names, values, and tool names.\n"
        "- Keep responses concise — no filler, no motivational language, no 'Great question!'.\n"
        "- Do not repeat tool results verbatim — summarise key findings in 2-5 bullet points.\n"
        "- For numbers, use commas for thousands (e.g. 1,234) and round to 2 decimal places.\n\n"
        "IMPORTANT RULES:\n"
        "- Always use a tool to answer — never make up data.\n"
        "- Never say 'please run the tool yourself' — if a tool exists for it, call it.\n"
        "- Never output SQL, Python code, or pseudocode.\n\n"
        "ACTION TOOLS (these EXECUTE operations on the data):\n"
        "- run_validation          → validate the data for issues\n"
        "- run_duplicate_detection → find duplicate/similar rows\n"
        "- run_data_profiling      → profile columns (stats, distributions, outliers)\n"
        "- generate_rules          → auto-infer validation rules from the data\n"
        "- apply_fixes             → apply suggested fixes from validation report\n"
        "- run_basic_cleaning      → strip whitespace, normalize text, lowercase\n"
        "- apply_column_trimming   → remove low-info, high-missing, correlated columns\n"
        "- merge_duplicates        → merge duplicate clusters into one record each\n"
        "- run_auto_report         → full end-to-end pipeline (validate+clean+deduplicate+score)\n"
        "- execute_pipeline        → run a saved pipeline by name\n"
        "- run_entity_resolution   → match entities across main and reference datasets\n"
        "- check_corpus            → validate values against Redis-backed reference corpora (canonical names, valid postcodes, etc.)\n"
        "- edit_cell               → change a specific cell value (row + column + new value)\n"
        "- download_data           → prepare dataset (raw/corrected/cleaned) for download\n"
        "- run_bert_validation     → BERT-enhanced validation with AI explanations of issues\n"
        "- build_vector_index      → build semantic vector index from a column\n"
        "- search_vector_index     → search vector index for semantically similar values\n"
        "- save_pipeline           → save current steps as a reusable named pipeline\n"
        "- edit_rule               → add, modify, or delete a validation rule for a column\n"
        "- map_corpus              → map a dataset column to a loaded corpus for validation\n"
        "- configure_validation    → configure ML models, contamination rate, BERT model, API settings\n\n"
        "READ TOOLS (these QUERY existing results):\n"
        "- get_summary, count_issues, get_worst_rows, explain_row, get_column_issues\n"
        "- get_quality_score, get_fixable_issues, get_duplicates, show_data\n"
        "- list_corpora         → list all corpora loaded in Redis\n"
        "- list_rules           → list active validation rules (optionally for a specific column)\n"
        "- get_profile          → null rate, top values, stats for a specific column or full dataset\n"
        "- recommend_next_steps → analyse pipeline state and suggest what to do next\n"
        "- query_data           → evaluate any pandas expression on df for ad-hoc data questions\n"
        "- get_scorecard        → detailed quality scorecard with per-column breakdown and before/after comparison\n"
        "- review_fixes         → show pending fix suggestions for row-by-row review before applying\n"
        "- show_entity_graph    → text-based entity resolution graph summary with clusters and scores\n\n"
        "ROUTING RULES:\n"
        "- User asks to find/detect duplicates → run_duplicate_detection\n"
        "- User asks to merge/remove duplicates → merge_duplicates\n"
        "- User asks to validate/check data → run_validation\n"
        "- User asks to profile/explore data → run_data_profiling\n"
        "- User asks to generate/infer rules → generate_rules\n"
        "- User asks to list/show rules but no rules exist yet → call generate_rules first, then list_rules\n"
        "- User asks to validate then show rules → run_validation first, then generate_rules, then list_rules\n"
        "- User asks to apply fixes/corrections → apply_fixes\n"
        "- User asks to clean/tidy/normalize text → run_basic_cleaning\n"
        "- User asks to trim/remove columns → apply_column_trimming\n"
        "- User asks for full report/run everything → run_auto_report\n"
        "- User asks to run a pipeline → execute_pipeline\n"
        "- User asks to match/link/resolve entities → run_entity_resolution\n"
        "- User asks to check canonical/corpus/reference/standardize values → check_corpus\n"
        "- User asks about null rate, missing rate, unique count, data type, column stats → get_profile\n"
        "- User asks most common/frequent value, top N values, value counts → query_data (NOT get_profile)\n"
        "- User asks to change/update/fix/set/correct a cell value → edit_cell\n"
        "- User asks what to do next, where to start, what's left → recommend_next_steps\n"
        "- User asks any ad-hoc question about specific values, frequencies, filters, averages → query_data\n"
        "  (e.g. 'which customer appears most', 'average age', 'rows where city is London', 'top 5 products')\n"
        "- User asks to download/export/save the dataset → download_data\n"
        "- User asks for BERT explanations, AI explanations, or WHY issues were flagged → run_bert_validation\n"
        "- User asks for anomaly detection, ML validation, or advanced validation → run_validation with use_ml=true\n"
        "- User asks to build a vector index, create embeddings, set up semantic search → build_vector_index\n"
        "- User asks to search semantically, find similar values by meaning → search_vector_index\n"
        "- User asks to save a pipeline, create a reusable workflow → save_pipeline\n"
        "- User asks for scorecard, detailed quality breakdown, per-column scores → get_scorecard\n"
        "- User asks to add/change/delete a validation rule → edit_rule\n"
        "- User asks to review fixes before applying, approve/reject per row → review_fixes\n"
        "- User asks to map a column to a corpus → map_corpus\n"
        "- User asks to configure ML models, contamination, BERT model, API settings → configure_validation\n"
        "- User asks about entity graph, entity clusters, connected entities → show_entity_graph\n"
        "- Only call open_tool if user explicitly asks to OPEN or GO TO a panel.\n\n"
        "MULTI-STEP CHAINING:\n"
        "You can chain multiple tool calls across turns. After each set of tool results,\n"
        "you will be called again and can call additional tools based on what you learned.\n"
        "Example workflows:\n"
        "- 'validate and show worst rows' -> call run_validation, see results, then call get_worst_rows\n"
        "- 'find duplicates and merge them' -> call run_duplicate_detection, then merge_duplicates\n"
        "- 'validate, show worst 5 rows, then fix them' -> run_validation -> get_worst_rows -> apply_fixes\n"
        "When you need results from one step to decide the next, call tools one step at a time.\n"
        "When tools are independent, call multiple in the same turn.\n"
        "When all steps are complete, respond with a final text summary of everything done.\n"
        "Do NOT re-call tools you already called unless the user explicitly asks to re-run them.\n\n"
        "CRITICAL FORMATTING REMINDER (MUST FOLLOW):\n"
        "- Absolutely NO emojis anywhere in your response. Not a single one.\n"
        "- No unicode symbols (no checkmarks, warning signs, arrows, colored circles).\n"
        "- Use only plain text, markdown bold, markdown tables, bullet points, and code backticks.\n"
        "- Professional tone. No 'Great question!', 'Sure!', 'Absolutely!', or similar filler.\n\n"
        + context
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def call_ollama(
    user_message: str,
    context: str,
    history: List[Dict[str, str]],
    model: str = "llama3.1",
) -> str:
    """Simple Ollama call without tools (used as fallback)."""
    try:
        import ollama
    except ImportError:
        return "Ollama Python package is not installed. Run: pip install ollama"

    messages = _build_messages(user_message, context, history)

    try:
        response = ollama.chat(model=model, messages=messages)
        return _get_response_content(response)
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            return (
                "Cannot connect to Ollama. Make sure it is running:\n\n"
                "  ollama serve\n\nThen try again."
            )
        if "not found" in err.lower() or "model" in err.lower():
            return (
                f"Model '{model}' not found locally.\n"
                f"Pull it first:\n\n  ollama pull {model}"
            )
        return f"Ollama error: {err}"


def _extract_args_str(text: str, open_pos: int) -> str:
    """
    Given text and the position just after an opening '(', scan forward
    tracking depth and quotes to find the matching ')'. Returns the args string.
    """
    depth = 1
    i = open_pos
    in_dq = False  # inside double-quoted string
    in_sq = False  # inside single-quoted string
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '"' and not in_sq:
            in_dq = not in_dq
        elif c == "'" and not in_dq:
            in_sq = not in_sq
        elif not in_dq and not in_sq:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
        i += 1
    return text[open_pos: i - 1] if depth == 0 else text[open_pos:]


def _parse_text_tool_call(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Fallback: detect when the LLM writes a tool call as plain text instead of
    invoking it via the API (e.g. 'get_row(row_number=2)' or 'query_data(expression="...")').
    Returns a list of (tool_name, args) tuples, same format as _extract_tool_calls.
    """
    import re
    import json

    known_tools = {t["function"]["name"] for t in TOOLS}
    results = []

    # Pattern 1: tool_name(...) — Python-style call, depth-aware arg extraction
    tool_name_pat = re.compile(r'\b(' + '|'.join(re.escape(n) for n in known_tools) + r')\s*\(')
    for m in tool_name_pat.finditer(text):
        tool_name = m.group(1)
        args_str = _extract_args_str(text, m.end()).strip()
        args: Dict[str, Any] = {}
        # Parse key="value", key='value', key=123 pairs
        for pair in re.finditer(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|([^\s,)]+))', args_str):
            key = pair.group(1)
            val: Any = pair.group(2) if pair.group(2) is not None else (
                       pair.group(3) if pair.group(3) is not None else
                       (pair.group(4) or "").rstrip(",").strip())
            try:
                val = int(val)
            except (ValueError, TypeError):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
            args[key] = val
        results.append((tool_name, args))

    if results:
        return results

    # Pattern 2: {"name": "tool_name", "arguments": {...}} — JSON-style
    json_pattern = re.compile(
        r'\{"name"\s*:\s*"(' + '|'.join(re.escape(n) for n in known_tools) + r')".*?"arguments"\s*:\s*(\{.*?\})',
        re.DOTALL,
    )
    for m in json_pattern.finditer(text):
        tool_name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except Exception:
            args = {}
        results.append((tool_name, args))

    return results


def call_ollama_with_tools(
    user_message: str,
    context: str,
    history: List[Dict[str, str]],
    model: str = "llama3.1",
    state: Optional[Dict[str, Any]] = None,
    include_data: bool = False,
    n_rows: int = 20,
) -> Dict[str, Any]:
    """
    Call Ollama with tool calling enabled — agentic loop.
    The LLM can chain multiple tool calls across iterations.
    Returns {"response": str, "open_tool": str | None, "tools_called": [str]}
    """
    if state is None:
        state = {}
    if include_data:
        context = build_context(state, include_data=True, n_rows=n_rows)

    try:
        import ollama
    except ImportError:
        return {
            "response": "Ollama Python package is not installed. Run: pip install ollama",
            "open_tool": None,
        }

    messages = _build_messages(user_message, context, history)

    # ── Accumulators across iterations ──
    all_tools_called: List[str] = []
    all_results: List[str] = []  # raw results for the user
    open_tool_name: Optional[str] = None
    _NEEDS_SUMMARY = {"open_tool", "recommend_next_steps"}

    # ═══ AGENTIC LOOP ═══
    for iteration in range(MAX_AGENT_ITERATIONS):
        try:
            response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                return {
                    "response": "Cannot connect to Ollama. Make sure it is running:\n\n  ollama serve",
                    "open_tool": None,
                }
            if iteration == 0:
                return {
                    "response": call_ollama(user_message, context, history, model),
                    "open_tool": None,
                }
            break  # subsequent iteration failed — return what we have

        tool_calls = _extract_tool_calls(response)

        # Text-based fallback only on first iteration
        if not tool_calls and iteration == 0:
            raw_text = _get_response_content(response)
            tool_calls = _parse_text_tool_call(raw_text)
            if not tool_calls:
                hint = (
                    "\n\n---\n*No tool was executed for this query. "
                    "Try being more specific, e.g.: 'show me the first 10 rows', "
                    "'run validation', 'list rules', 'edit row 5 email to x@y.com'*"
                )
                return {"response": raw_text + hint, "open_tool": None, "tools_called": []}

        # No tool calls on later iterations → LLM is done
        if not tool_calls:
            final_text = _get_response_content(response)
            if all_results:
                combined = "\n\n".join(all_results)
                if final_text:
                    combined += "\n\n" + final_text
                final_text = combined
            return {
                "response": final_text or "(No results.)",
                "open_tool": open_tool_name,
                "tools_called": all_tools_called,
            }

        # ── Dispatch all tool calls in this iteration ──
        tool_result_messages = []
        for fn_name, fn_args in tool_calls:
            all_tools_called.append(fn_name)
            result_text, _open = _dispatch_tool(fn_name, fn_args, state)
            if _open:
                open_tool_name = _open
            if fn_name not in _NEEDS_SUMMARY:
                all_results.append(result_text)
            tool_result_messages.append({
                "role": "tool",
                "content": _truncate_result(result_text),
                "tool_name": fn_name,
            })

        # ── Feed results back into messages for next iteration ──
        # Append the full response message (preserves tool_calls structure)
        messages.append(response.message)
        messages.extend(tool_result_messages)
        # Loop continues — LLM sees results and decides next step

    # ── Max iterations reached — return everything we have ──
    final_text = "\n\n".join(all_results) if all_results else "(Reached maximum tool-chaining depth.)"
    return {
        "response": final_text,
        "open_tool": open_tool_name,
        "tools_called": all_tools_called,
    }



# ─────────────────────────────────────────────────────────
# Main Entry Point
def _tools_to_anthropic_format() -> List[Dict[str, Any]]:
    """Convert the shared TOOLS list from OpenAI/Ollama format to Anthropic format."""
    converted = []
    for tool in TOOLS:
        fn = tool["function"]
        params = fn.get("parameters", {"type": "object", "properties": {}, "required": []})
        converted.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": params,
        })
    return converted


def call_claude_with_tools(
    user_message: str,
    context: str,
    history: List[Dict[str, str]],
    model: str = "claude-sonnet-4-6",
    state: Optional[Dict[str, Any]] = None,
    api_key: str = "",
    **_kwargs,
) -> Dict[str, Any]:
    """
    Call Claude (Anthropic API) with tool calling — agentic loop.
    The LLM can chain multiple tool calls across iterations.
    Returns {"response": str, "open_tool": str | None, "tools_called": [str]}
    """
    try:
        import anthropic
    except ImportError:
        return {
            "response": "Anthropic package not installed. Run: pip install anthropic",
            "open_tool": None,
        }

    if not api_key:
        return {
            "response": "No Anthropic API key provided. Add it in Model settings.",
            "open_tool": None,
        }

    client = anthropic.Anthropic(api_key=api_key)
    claude_tools = _tools_to_anthropic_format()

    # Build the system prompt — include routing rules + chaining instructions
    # (same as Ollama path gets via _build_messages)
    system_prompt = (
        "You are an expert data quality analyst AI agent embedded in an enterprise data quality platform.\n"
        "You don't just answer questions — you actively investigate, diagnose, and fix data quality issues.\n\n"
        "CAPABILITIES:\n"
        "- Run validation to find issues\n"
        "- Profile columns for deep analysis\n"
        "- Check batch history to spot trends and regressions\n"
        "- Compare batches to identify what changed\n"
        "- Apply fixes (with user confirmation for destructive actions)\n"
        "- Save results to S3\n"
        "- Run a full autonomous investigation pipeline\n\n"
        "AGENTIC BEHAVIOR:\n"
        "When a user asks you to investigate or analyse data, be proactive:\n"
        "1. Run validation if it hasn't been run yet\n"
        "2. Identify the worst columns and drill into them\n"
        "3. Check batch history for trends\n"
        "4. Suggest specific fixes with evidence\n"
        "5. Ask for confirmation before applying changes\n\n"
        "MULTI-STEP CHAINING:\n"
        "You can chain multiple tool calls across turns. After each set of tool results,\n"
        "you will be called again and can call additional tools based on what you learned.\n"
        "When you need results from one step to decide the next, call tools one step at a time.\n"
        "When all steps are complete, respond with a final text summary.\n\n"
        "Be concise and practical. Never make up data — always use the tools to get real information.\n\n"
        + context
    )

    # Build messages — Anthropic format uses a system param, not a system message
    messages: List[Dict[str, Any]] = []
    for h in history:
        role = h.get("role", "")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    # ── Accumulators across iterations ──
    all_tools_called: List[str] = []
    all_results: List[str] = []  # raw results for the user
    open_tool_name: Optional[str] = None
    _NEEDS_SUMMARY = {"open_tool", "recommend_next_steps"}

    # ═══ AGENTIC LOOP ═══
    for iteration in range(MAX_AGENT_ITERATIONS):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                tools=claude_tools,
                messages=messages,
            )
        except anthropic.AuthenticationError:
            return {"response": "Invalid Anthropic API key.", "open_tool": None}
        except anthropic.RateLimitError:
            return {"response": "Anthropic rate limit reached. Please wait.", "open_tool": None}
        except Exception as e:
            if iteration == 0:
                return {"response": f"Claude API error: {e}", "open_tool": None}
            break  # subsequent iteration failed — return what we have

        text_blocks = [b for b in response.content if b.type == "text"]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # ── No tool calls → LLM is done ──
        if not tool_use_blocks:
            final_text = "\n".join(b.text for b in text_blocks)
            if all_results:
                combined = "\n\n".join(all_results)
                if final_text:
                    combined += "\n\n" + final_text
                final_text = combined
            if not final_text and not all_results:
                final_text = (
                    "\n\n---\n*No tool was executed for this query. "
                    "Try being more specific.*"
                )
            return {
                "response": final_text or "(No results.)",
                "open_tool": open_tool_name,
                "tools_called": all_tools_called,
            }

        # ── Dispatch all tool calls in this iteration ──
        tool_result_contents = []
        for block in tool_use_blocks:
            fn_name = block.name
            fn_args = dict(block.input) if block.input else {}
            all_tools_called.append(fn_name)
            result_text, _open = _dispatch_tool(fn_name, fn_args, state or {})
            if _open:
                open_tool_name = _open
            if fn_name not in _NEEDS_SUMMARY:
                all_results.append(result_text)
            tool_result_contents.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _truncate_result(result_text),
            })

        # ── Feed results back into messages for next iteration ──
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_result_contents})

        # If stop_reason is end_turn (not tool_use), model is done
        if response.stop_reason == "end_turn":
            break
        # Loop continues — LLM sees results and decides next step

    # ── Max iterations reached — return everything we have ──
    final_text = "\n\n".join(all_results) if all_results else "(Reached maximum tool-chaining depth.)"
    return {
        "response": final_text,
        "open_tool": open_tool_name,
        "tools_called": all_tools_called,
    }


# ─────────────────────────────────────────────────────────

def answer(
    user_message: str,
    state: Dict[str, Any],
    history: List[Dict[str, str]],
    model: str = "llama3.1",
    include_data: bool = False,
    n_rows: int = 20,
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Answer a user's question.

    Routes to Claude (Anthropic API) when model starts with 'claude-',
    otherwise uses Ollama (local).

    Args:
        include_data: When True, actual data rows are sent to the LLM context.
        n_rows: Number of data rows to include when include_data is True.
        api_key: Anthropic API key (only needed for Claude models).

    Returns: {"response": str, "open_tool": str | None}
    """
    # ── Handle pending confirmation ──
    _confirm_words = {"yes", "y", "confirm", "go ahead", "do it", "proceed", "ok", "sure", "yep", "yeah"}
    _cancel_words = {"no", "n", "cancel", "abort", "stop", "nevermind", "nope", "don't"}
    _user_lower = user_message.strip().lower().rstrip(".!,")

    pending = state.get("_pending_action")
    if pending and _user_lower in _confirm_words:
        # Execute the stored destructive action
        fn_name = pending["tool"]
        fn_args = pending["args"]
        del state["_pending_action"]
        result_text, open_tool = _dispatch_tool(fn_name, fn_args, state, force=True)
        return {
            "response": result_text,
            "open_tool": open_tool,
            "tools_called": [fn_name],
        }
    elif pending and _user_lower in _cancel_words:
        del state["_pending_action"]
        return {
            "response": "Action cancelled. Your data was not modified.",
            "open_tool": None,
            "tools_called": [],
        }

    # Clear any stale pending action if user asks something new
    if pending:
        del state["_pending_action"]

    context = build_context(state, include_data=include_data, n_rows=n_rows)

    if model.startswith("claude-"):
        return call_claude_with_tools(
            user_message, context, history, model=model, state=state, api_key=api_key,
        )

    return call_ollama_with_tools(
        user_message, context, history, model, state,
        include_data=False,  # already baked into context above
        n_rows=n_rows,
    )
