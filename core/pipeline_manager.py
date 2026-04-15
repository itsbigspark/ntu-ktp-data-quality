# core/pipeline_manager.py
"""
Pipeline Manager - Save, load, and execute data quality pipelines
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class Pipeline:
    """Represents a data quality pipeline"""

    def __init__(
        self,
        name: str,
        description: str = "",
        steps: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.description = description
        self.steps = steps or []
        self.metadata = metadata or {}

        # Auto-populate metadata
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        if "version" not in self.metadata:
            self.metadata["version"] = "1.0"

    def add_step(self, step_type: str, config: Dict[str, Any], description: str = ""):
        """Add a pipeline step"""
        step = {
            "step_number": len(self.steps) + 1,
            "type": step_type,
            "description": description,
            "config": config,
            "enabled": True
        }
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        """Create pipeline from dictionary"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=data.get("steps", []),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        """Convert pipeline to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Pipeline':
        """Create pipeline from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class PipelineManager:
    """Manages pipeline storage and retrieval"""

    def __init__(self, pipelines_dir: str = "pipelines"):
        self.pipelines_dir = Path(pipelines_dir)
        self.pipelines_dir.mkdir(exist_ok=True)

    def save_pipeline(self, pipeline: Pipeline) -> str:
        """Save pipeline to file"""
        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in pipeline.name)
        filename = f"{safe_name}.json"
        filepath = self.pipelines_dir / filename

        # Add last modified timestamp
        pipeline.metadata["last_modified"] = datetime.now().isoformat()

        # Save to file
        with open(filepath, 'w') as f:
            f.write(pipeline.to_json())

        return str(filepath)

    def load_pipeline(self, name: str) -> Optional[Pipeline]:
        """Load pipeline by name"""
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
        filename = f"{safe_name}.json"
        filepath = self.pipelines_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            json_str = f.read()

        return Pipeline.from_json(json_str)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines"""
        pipelines = []

        for filepath in self.pipelines_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                pipelines.append({
                    "name": data["name"],
                    "description": data.get("description", ""),
                    "steps_count": len(data.get("steps", [])),
                    "created_at": data.get("metadata", {}).get("created_at", ""),
                    "last_modified": data.get("metadata", {}).get("last_modified", ""),
                    "version": data.get("metadata", {}).get("version", "1.0"),
                    "filepath": str(filepath)
                })
            except Exception as e:
                print(f"Error loading pipeline {filepath}: {e}")
                continue

        # Sort by last modified (newest first)
        pipelines.sort(key=lambda x: x.get("last_modified", ""), reverse=True)

        return pipelines

    def delete_pipeline(self, name: str) -> bool:
        """Delete pipeline by name"""
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
        filename = f"{safe_name}.json"
        filepath = self.pipelines_dir / filename

        if filepath.exists():
            filepath.unlink()
            return True

        return False

    def export_pipeline(self, name: str, export_path: str) -> bool:
        """Export pipeline to specified path"""
        pipeline = self.load_pipeline(name)
        if not pipeline:
            return False

        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        with open(export_file, 'w') as f:
            f.write(pipeline.to_json())

        return True

    def import_pipeline(self, filepath: str) -> Optional[Pipeline]:
        """Import pipeline from file"""
        try:
            with open(filepath, 'r') as f:
                json_str = f.read()

            pipeline = Pipeline.from_json(json_str)

            # Save to pipelines directory
            self.save_pipeline(pipeline)

            return pipeline
        except Exception as e:
            print(f"Error importing pipeline: {e}")
            return None


class PipelineExecutor:
    """Executes pipeline steps"""

    def __init__(self, streamlit_state):
        """
        Initialize with Streamlit session state

        Args:
            streamlit_state: Streamlit session state object
        """
        self.state = streamlit_state

    def execute_pipeline(
        self,
        pipeline: Pipeline,
        df: pd.DataFrame,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Execute entire pipeline

        Args:
            pipeline: Pipeline to execute
            df: Input dataframe
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with results and execution details
        """
        results = {
            "success": True,
            "df_output": df.copy(),
            "steps_executed": [],
            "errors": []
        }

        current_df = df.copy()

        for idx, step in enumerate(pipeline.steps):
            if not step.get("enabled", True):
                continue

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(pipeline.steps), step)

            try:
                # Execute step based on type
                step_result = self._execute_step(step, current_df)

                if step_result["success"]:
                    current_df = step_result["df_output"]
                    results["steps_executed"].append({
                        "step_number": step["step_number"],
                        "type": step["type"],
                        "description": step.get("description", ""),
                        "success": True,
                        "details": step_result.get("details", {})
                    })
                else:
                    results["errors"].append({
                        "step_number": step["step_number"],
                        "type": step["type"],
                        "error": step_result.get("error", "Unknown error")
                    })

                    # Stop execution on error
                    results["success"] = False
                    break

            except Exception as e:
                results["errors"].append({
                    "step_number": step["step_number"],
                    "type": step["type"],
                    "error": str(e)
                })
                results["success"] = False
                break

        results["df_output"] = current_df

        return results

    def _execute_step(self, step: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute a single pipeline step"""

        step_type = step["type"]
        config = step.get("config", {})

        result = {
            "success": False,
            "df_output": df.copy(),
            "details": {}
        }

        try:
            if step_type == "validate":
                result = self._execute_validate(df, config)

            elif step_type == "clean":
                result = self._execute_clean(df, config)

            elif step_type == "corpus_standardize":
                result = self._execute_corpus_standardize(df, config)

            elif step_type == "deduplicate":
                result = self._execute_deduplicate(df, config)

            elif step_type == "trim":
                result = self._execute_trim(df, config)

            else:
                result["error"] = f"Unknown step type: {step_type}"

        except Exception as e:
            result["error"] = str(e)

        return result

    def _execute_validate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation step"""
        from core.validator.validate import validate_df

        result = {"success": True, "df_output": df.copy(), "details": {}}

        # Get rules from config or session state
        rules = config.get("rules") or self.state.get("rules_merged")

        if not rules:
            # Skip validation if no rules available (instead of failing)
            result["success"] = True
            result["details"]["skipped"] = True
            result["details"]["reason"] = "No validation rules available. Create rules in the Rules tab first."
            result["details"]["issues_found"] = 0
            result["details"]["issues"] = []
            return result

        # Run validation
        issues_df = validate_df(df, rules)

        result["details"]["issues_found"] = len(issues_df)
        result["details"]["issues"] = issues_df.to_dict('records') if not issues_df.empty else []

        # Apply fixes if configured
        if config.get("apply_fixes", False) and not issues_df.empty:
            # TODO: Implement auto-fix logic
            result["details"]["fixes_applied"] = 0

        return result

    def _execute_clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cleaning step"""
        from core.preprocess import basic_clean

        df_cleaned = basic_clean(
            df,
            lowercase=config.get("lowercase", True),
            strip_ws=config.get("strip_whitespace", True),
            normalize_punct=config.get("normalize_punctuation", True)
        )

        changes = (df != df_cleaned).sum().sum()

        return {
            "success": True,
            "df_output": df_cleaned,
            "details": {
                "values_changed": int(changes)
            }
        }

    def _execute_corpus_standardize(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute corpus standardization step"""
        from core.corpus_validation import apply_corpus_standardization

        corpus_manager = self.state.get("corpus_manager")

        if not corpus_manager:
            # Skip if corpus manager not available
            return {
                "success": True,
                "df_output": df.copy(),
                "details": {
                    "skipped": True,
                    "reason": "Corpus manager not available. Configure corpus in the Corpus Manager tab first.",
                    "columns_standardized": 0,
                    "values_changed": 0
                }
            }

        mappings = config.get("mappings", {})

        if not mappings:
            # Skip if no mappings configured
            return {
                "success": True,
                "df_output": df.copy(),
                "details": {
                    "skipped": True,
                    "reason": "No corpus mappings provided. Configure corpus mappings in the Corpus Manager tab.",
                    "columns_standardized": 0,
                    "values_changed": 0
                }
            }

        df_standardized = apply_corpus_standardization(df, mappings, corpus_manager)

        changes = (df != df_standardized).sum().sum()

        return {
            "success": True,
            "df_output": df_standardized,
            "details": {
                "columns_standardized": len(mappings),
                "values_changed": int(changes)
            }
        }

    def _execute_deduplicate(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deduplication step"""
        from core.vectorizers import build_vectorizers
        from core.matcher import compute_pair_scores, label_pairs
        from core.dedup_clustering import cluster_duplicates, apply_deduplication

        # Get configuration
        columns = config.get("columns", [])
        threshold = config.get("threshold", 0.80)
        merge_strategy = config.get("merge_strategy", "fill_nulls")

        if not columns:
            # Skip if no columns specified - use all string columns as default
            string_columns = df.select_dtypes(include=['object']).columns.tolist()
            if not string_columns:
                return {
                    "success": True,
                    "df_output": df.copy(),
                    "details": {
                        "skipped": True,
                        "reason": "No string columns available for deduplication.",
                        "duplicates_found": 0,
                        "rows_merged": 0
                    }
                }
            columns = string_columns[:3]  # Use first 3 string columns by default

        # Build vectorizers
        vec_pack = build_vectorizers(
            df,
            columns,
            {
                "use_char": True,
                "use_word": False,
                "char_range": (3, 5),
                "word_range": (1, 2),
                "svd_components": 0
            }
        )

        # Compute similarity scores
        scores_df = compute_pair_scores(
            df=df,
            vec_pack=vec_pack,
            comparison_mode="Row-concatenated",
            run_mode="Fast KNN" if len(df) < 10000 else "Blocking + KNN",
            topk=20
        )

        # Label pairs
        pairs_df = label_pairs(scores_df, similar_thr=0.60, duplicate_thr=threshold)
        duplicate_pairs = pairs_df[pairs_df['label'] == 'duplicate'].copy()

        if duplicate_pairs.empty:
            return {
                "success": True,
                "df_output": df.copy(),
                "details": {
                    "duplicates_found": 0,
                    "duplicates_removed": 0
                }
            }

        # Cluster and merge
        clusters = cluster_duplicates(duplicate_pairs, id_col_a='row_i', id_col_b='row_j')
        master_selections = {cid: rows[0] for cid, rows in clusters.items()}

        df_deduped = apply_deduplication(
            df,
            clusters,
            master_selections,
            merge_strategy=merge_strategy
        )

        return {
            "success": True,
            "df_output": df_deduped,
            "details": {
                "duplicates_found": len(duplicate_pairs),
                "duplicates_removed": len(df) - len(df_deduped),
                "clusters": len(clusters)
            }
        }

    def _execute_trim(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute column trimming step"""
        from core.trimming import compute_kept_columns, apply_trimming

        # Get configuration (using correct parameter names matching compute_kept_columns signature)
        low_threshold = config.get("low_threshold", 0.01)
        high_threshold = config.get("high_threshold", 0.99)
        missing_threshold = config.get("missing_threshold", 0.80)
        correlation_threshold = config.get("correlation_threshold", 0.90)
        protected_columns = config.get("protected_columns", [])

        # Apply trimming (apply_trimming computes and returns kept_columns internally)
        df_trimmed, kept_cols = apply_trimming(
            df,
            low_thresh=low_threshold,
            high_thresh=high_threshold,
            missing_thresh=missing_threshold,
            corr_thresh=correlation_threshold,
            protect=protected_columns
        )

        return {
            "success": True,
            "df_output": df_trimmed,
            "details": {
                "columns_removed": len(df.columns) - len(df_trimmed.columns),
                "columns_kept": len(df_trimmed.columns)
            }
        }
