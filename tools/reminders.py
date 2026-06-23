from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata

STORAGE_DIR = Path("./storage").expanduser().resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STORAGE_DIR / "reminders.db"

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY, text TEXT, due TEXT, done INTEGER)"
    )
    return conn

@tool
def set_reminder(text: str, due: str) -> str:
    """Set a reminder. `due` must be an ISO-8601 formatted date string."""
    try:
        datetime.fromisoformat(due)
    except ValueError:
        return "Error: due must be a valid ISO-8601 formatted string."
        
    try:
        with _get_db() as conn:
            conn.execute("INSERT INTO reminders (text, due, done) VALUES (?, ?, 0)", (text, due))
            return f"Reminder set for {due}."
    except Exception as e:
        return f"Error: {e}"

@tool
def list_reminders(include_done: bool = False) -> str:
    """List reminders. By default only shows pending ones."""
    try:
        with _get_db() as conn:
            cursor = conn.cursor()
            if include_done:
                cursor.execute("SELECT id, text, due, done FROM reminders ORDER BY due")
            else:
                cursor.execute("SELECT id, text, due, done FROM reminders WHERE done = 0 ORDER BY due")
                
            rows = cursor.fetchall()
            if not rows:
                return "No reminders found."
                
            output = []
            for row in rows:
                checkbox = "[x]" if row[3] else "[ ]"
                output.append(f"{checkbox} {row[2]} — {row[1]} (id: {row[0]})")
            return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(set_reminder, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(list_reminders, metadata=ToolMetadata(risk_level="safe"))
