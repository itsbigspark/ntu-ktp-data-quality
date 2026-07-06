"""
MCP server — expose the DataQualify engine as tools any agent can call.

Wraps the source connectors, validation engine, and tracked batch runner as
Model Context Protocol tools, so an external agent (Claude Desktop, a pipeline
agent, anything MCP-aware) can drive data quality without touching the UI:

    validate(source)          # fetch a source, validate, summarise the issues
    run_batch(source)         # fetch, validate, record a tracked batch
    list_batches(limit)       # recent tracked batches
    infer_rules(source)       # inferred validation rules for a source

The tool *logic* lives in plain, importable ``tool_*`` functions (testable with
no MCP dependency); ``build_server`` registers them as MCP tools.

Run the server (requires the ``[mcp]`` extra)::

    pip install -e ".[mcp]"
    python -m dataqualify.mcp_server        # stdio transport

Point an MCP client (e.g. Claude Desktop) at that command to give the agent the
tools. The tools return deterministic engine results; the calling agent decides
what to do with them — the data-quality decisions stay in the engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _load(source: str) -> pd.DataFrame:
    from dataqualify.sources import parse_source
    return parse_source(source).fetch()


def tool_validate(source: str, reference: Optional[str] = None) -> Dict[str, Any]:
    """Validate a data source and summarise the issues found."""
    import dataqualify as dq

    df = _load(source)
    ref = _load(reference) if reference else None
    issues = dq.validate(df, reference=ref)
    by_issue = issues["issue"].value_counts().to_dict() if len(issues) else {}
    sample = (
        issues.head(25)[["row_id", "column", "issue", "detail", "severity"]]
        .to_dict("records")
        if len(issues) else []
    )
    return {
        "source": source,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "issues_count": int(len(issues)),
        "by_issue": {str(k): int(v) for k, v in by_issue.items()},
        "sample_issues": sample,
    }


def tool_run_batch(source: str, sink_dir: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a source, run the pipeline, and record a tracked batch."""
    import dataqualify as dq

    rec = dq.run_batch(dq.parse_source(source), sink_dir=sink_dir)
    return rec.to_dict()


def tool_list_batches(limit: int = 20) -> List[Dict[str, Any]]:
    """List recently tracked batches (id, source, status, score, timing)."""
    from dataqualify.batch import SQLiteBatchStore

    return [r.to_dict() for r in SQLiteBatchStore().list(limit=limit)]


def tool_infer_rules(source: str) -> Dict[str, Any]:
    """Infer validation rules from a data source (types, bounds, allowed values, presence)."""
    import dataqualify as dq

    rules = dq.infer_rules(_load(source))
    cols = {
        c: {k: v for k, v in (r or {}).items() if not k.startswith("_")}
        for c, r in (rules.get("columns") or {}).items()
    }
    return {"columns": cols, "uniqueness": rules.get("uniqueness", {})}


def build_server():
    """Construct the FastMCP server with the tools registered. Requires ``mcp``."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("dataqualify")

    @server.tool()
    def validate(source: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """Validate a data source (file path, s3://, https://, or companies-house:NUMBER)
        and summarise the issues found."""
        return tool_validate(source, reference)

    @server.tool()
    def run_batch(source: str, sink_dir: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a source, validate it, and record a tracked batch. Optionally write
        issues/corrected/report to sink_dir."""
        return tool_run_batch(source, sink_dir)

    @server.tool()
    def list_batches(limit: int = 20) -> List[Dict[str, Any]]:
        """List recently tracked batches."""
        return tool_list_batches(limit)

    @server.tool()
    def infer_rules(source: str) -> Dict[str, Any]:
        """Infer validation rules from a data source."""
        return tool_infer_rules(source)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
