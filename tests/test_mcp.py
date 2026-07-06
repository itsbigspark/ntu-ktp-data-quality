"""
Tests for the MCP tool layer.

The tool *logic* (tool_validate / tool_infer_rules) is tested directly with no
MCP dependency. The FastMCP server construction is tested only when the optional
``mcp`` package is installed.

Run:  ./.venv/bin/python -m pytest tests/test_mcp.py -v
"""
from __future__ import annotations

import warnings
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from dataqualify.mcp_server import tool_validate, tool_infer_rules


def test_tool_validate_summarises_issues(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({
        "ccy": ["GBP"] * 8 + ["gbp"],
        "email": [f"u{i}@x.com" for i in range(8)] + ["bad"],
    }).to_csv(p, index=False)

    out = tool_validate(str(p))
    assert out["rows"] == 9
    assert out["issues_count"] == 2               # gbp + bad email
    assert set(out["by_issue"]) == {"format_error", "not in allowed set"}
    assert isinstance(out["sample_issues"], list)


def test_tool_infer_rules_returns_columns(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"status": ["Active", "Dissolved"] * 10, "n": range(20)}).to_csv(p, index=False)
    out = tool_infer_rules(str(p))
    assert "status" in out["columns"] and "n" in out["columns"]
    # internal meta keys must not leak into the tool output
    assert all(not k.startswith("_") for col in out["columns"].values() for k in col)


def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    from dataqualify.mcp_server import build_server
    server = build_server()
    assert server.name == "dataqualify"
