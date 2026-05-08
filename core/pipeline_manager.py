# core/pipeline_manager.py
"""
Pipeline Manager - Save, load, and execute data quality pipelines
"""

import json
import os
import io
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import pandas as pd

logger = logging.getLogger("dq_engine.pipeline_manager")


def _save_step_to_s3(pipeline_run_id: str, step_number: int, step_type: str,
                     df: pd.DataFrame, bucket: str, region: str = "us-east-1") -> str:
    """Save a step's output DataFrame to S3. Returns the S3 URI or empty string on failure."""
    try:
        import boto3
        key = f"pipeline_runs/{pipeline_run_id}/step_{step_number:02d}_{step_type}.csv"
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        boto3.client("s3", region_name=region).put_object(
            Bucket=bucket, Key=key, Body=buf.getvalue()
        )
        uri = f"s3://{bucket}/{key}"
        logger.info("Saved step output: %s", uri)
        return uri
    except Exception as exc:
        logger.warning("Could not save step to S3: %s", exc)
        return ""


def _save_pipeline_run_to_db(pipeline_run_id: str, pipeline_name: str,
                              steps_executed: list, source_rows: int,
                              output_rows: int, success: bool) -> None:
    """Persist a pipeline run summary + per-step records to PostgreSQL/SQLite."""
    try:
        from core.storage.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        _db_url = str(engine.url)
        is_pg = "postgresql" in _db_url or "postgres" in _db_url

        id_col = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

        with engine.connect() as conn:
            # Create tables if they don't exist
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id {id_col},
                    run_id TEXT NOT NULL,
                    pipeline_name TEXT,
                    started_at TIMESTAMP,
                    source_rows INTEGER,
                    output_rows INTEGER,
                    steps_count INTEGER,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS pipeline_step_results (
                    id {id_col},
                    run_id TEXT NOT NULL,
                    step_number INTEGER,
                    step_type TEXT,
                    description TEXT,
                    rows_in INTEGER,
                    rows_out INTEGER,
                    success BOOLEAN,
                    s3_uri TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Insert run summary
            conn.execute(text("""
                INSERT INTO pipeline_runs
                    (run_id, pipeline_name, started_at, source_rows, output_rows, steps_count, success)
                VALUES (:run_id, :name, :started_at, :src, :out, :steps, :ok)
            """), {
                "run_id": pipeline_run_id,
                "name": pipeline_name,
                "started_at": datetime.now(timezone.utc),
                "src": source_rows,
                "out": output_rows,
                "steps": len(steps_executed),
                "ok": success,
            })

            # Insert per-step records
            for s in steps_executed:
                conn.execute(text("""
                    INSERT INTO pipeline_step_results
                        (run_id, step_number, step_type, description,
                         rows_in, rows_out, success, s3_uri, details)
                    VALUES (:run_id, :num, :stype, :desc,
                            :rows_in, :rows_out, :ok, :uri, :details)
                """), {
                    "run_id": pipeline_run_id,
                    "num": s.get("step_number"),
                    "stype": s.get("type"),
                    "desc": s.get("description", ""),
                    "rows_in": s.get("rows_in", 0),
                    "rows_out": s.get("rows_out", 0),
                    "ok": s.get("success", True),
                    "uri": s.get("s3_uri", ""),
                    "details": json.dumps(s.get("details", {})),
                })
            conn.commit()
        logger.info("Pipeline run %s saved to DB", pipeline_run_id)
    except Exception as exc:
        logger.warning("Could not save pipeline run to DB: %s", exc)


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
        progress_callback=None,
        s3_bucket: str = None,
        s3_region: str = "us-east-1",
    ) -> Dict[str, Any]:
        """
        Execute entire pipeline, saving each step's output to S3 and DB.

        Args:
            pipeline: Pipeline to execute
            df: Input dataframe
            progress_callback: Optional callback for progress updates
            s3_bucket: If set, each step output is saved to S3
            s3_region: AWS region for S3 saves

        Returns:
            Dictionary with results, execution details, and per-step S3 URIs
        """
        pipeline_run_id = f"{pipeline.name.replace(' ', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

        results = {
            "success": True,
            "df_output": df.copy(),
            "steps_executed": [],
            "errors": [],
            "pipeline_run_id": pipeline_run_id,
        }

        current_df = df.copy()
        source_rows = len(df)

        for idx, step in enumerate(pipeline.steps):
            if not step.get("enabled", True):
                continue

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(pipeline.steps), step)

            rows_in = len(current_df)

            try:
                step_result = self._execute_step(step, current_df)

                if step_result["success"]:
                    current_df = step_result["df_output"]
                    rows_out = len(current_df)

                    # Save step output to S3 if bucket configured
                    s3_uri = ""
                    if s3_bucket:
                        s3_uri = _save_step_to_s3(
                            pipeline_run_id, step["step_number"],
                            step["type"], current_df, s3_bucket, s3_region,
                        )

                    results["steps_executed"].append({
                        "step_number": step["step_number"],
                        "type": step["type"],
                        "description": step.get("description", ""),
                        "success": True,
                        "rows_in": rows_in,
                        "rows_out": rows_out,
                        "s3_uri": s3_uri,
                        "details": step_result.get("details", {}),
                    })
                else:
                    results["errors"].append({
                        "step_number": step["step_number"],
                        "type": step["type"],
                        "error": step_result.get("error", "Unknown error"),
                    })
                    results["success"] = False
                    break

            except Exception as e:
                results["errors"].append({
                    "step_number": step["step_number"],
                    "type": step["type"],
                    "error": str(e),
                })
                results["success"] = False
                break

        results["df_output"] = current_df

        # Persist run summary + step records to DB (silently skips if DB unavailable)
        _save_pipeline_run_to_db(
            pipeline_run_id=pipeline_run_id,
            pipeline_name=pipeline.name,
            steps_executed=results["steps_executed"],
            source_rows=source_rows,
            output_rows=len(current_df),
            success=results["success"],
        )

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
