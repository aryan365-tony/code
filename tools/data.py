from __future__ import annotations

import json
import re
import sqlite3
import pandas as pd

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import _safe_path

@tool
def read_csv_summary(path: str) -> str:
    """Read a CSV and return shape, dtypes, and head(5)."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: file not found: {target}"
    try:
        df = pd.read_csv(target)
        summary = [
            f"Shape: {df.shape}",
            f"Columns and Dtypes:\n{df.dtypes}",
            f"Head:\n{df.head(5).to_string()}"
        ]
        return "\n\n".join(summary)
    except Exception as e:
        return f"Error: {e}"

@tool
def query_csv(path: str, query: str, columns: str | None = None) -> str:
    """Query a CSV using pandas DataFrame.query."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: file not found: {target}"
    try:
        df = pd.read_csv(target)
        result = df.query(query, engine="python")
        if columns:
            cols = [c.strip() for c in columns.split(",")]
            result = result[cols]
        return result.head(50).to_string()
    except (pd.errors.UndefinedVariableError, SyntaxError) as e:
        return f"Error: invalid query: {e}"
    except Exception as e:
        return f"Error: {e}"

@tool
def json_query(path: str, key_path: str) -> str:
    """Load JSON and query using dot/bracket notation."""
    target = _safe_path(path)
    if not target.exists():
        return f"Error: file not found: {target}"
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        keys = re.findall(r'[^.\[\]]+|\[\d+\]', key_path)
        current = data
        for key in keys:
            if key.startswith("[") and key.endswith("]"):
                idx = int(key[1:-1])
                if not isinstance(current, list) or idx >= len(current):
                    return f"Error: path not found: {key_path}"
                current = current[idx]
            else:
                if not isinstance(current, dict) or key not in current:
                    return f"Error: path not found: {key_path}"
                current = current[key]
                
        return json.dumps(current, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool
def sqlite_query(db_path: str, sql: str) -> str:
    """Execute a read-only SELECT query on a SQLite database."""
    target = _safe_path(db_path)
    if not target.exists():
        return f"Error: database not found: {target}"
        
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed"
        
    try:
        uri = f"file:{target.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(50)
            if not rows:
                return "No results."
                
            col_names = [description[0] for description in cursor.description]
            output = ["\t".join(col_names)]
            for row in rows:
                output.append("\t".join(str(v) for v in row))
                
            return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(read_csv_summary, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(query_csv, metadata=ToolMetadata(risk_level="safe", timeout_s=20.0))
register_lc_tool(json_query, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(sqlite_query, metadata=ToolMetadata(risk_level="safe", timeout_s=15.0))
