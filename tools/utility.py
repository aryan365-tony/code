from __future__ import annotations

import ast
import math
import os
import platform
import shutil
import zoneinfo
from datetime import datetime, timezone

from langchain_core.tools import tool

from tools.base import register_lc_tool, ToolMetadata
from tools.filesystem import WORKSPACE_DIR

@tool
def get_current_time() -> str:
    """Return the current date and time in ISO-8601 format with timezone."""
    return datetime.now(timezone.utc).astimezone().isoformat()

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed_names = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "log": math.log, "floor": math.floor, "ceil": math.ceil,
        "pow": math.pow
    }
    
    try:
        tree = ast.parse(expression, mode="eval")
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Constant)):
                continue
            if isinstance(node, ast.Call):
                continue
            if isinstance(node, ast.Name) and node.id in allowed_names:
                continue
            return f"Error: invalid expression or unsupported function/operator: {type(node).__name__}"
            
        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div): return left / right
                if isinstance(node.op, ast.FloorDiv): return left // right
                if isinstance(node.op, ast.Mod): return left % right
                if isinstance(node.op, ast.Pow): return left ** right
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if isinstance(node.op, ast.USub): return -operand
                if isinstance(node.op, ast.UAdd): return operand
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in allowed_names:
                    args = [_eval(arg) for arg in node.args]
                    return allowed_names[node.func.id](*args)
            raise ValueError(f"Unsupported syntax")
            
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def convert_timezone(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert an ISO time string from one timezone to another."""
    try:
        dt = datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(from_tz))
        dt = dt.astimezone(zoneinfo.ZoneInfo(to_tz))
        return dt.isoformat()
    except zoneinfo.ZoneInfoNotFoundError as e:
        return f"Error: timezone not found: {e}"
    except ValueError as e:
        return f"Error: invalid time string: {e}"
    except Exception as e:
        return f"Error: {e}"

@tool
def system_info() -> str:
    """Return information about the system and workspace disk usage."""
    try:
        usage = shutil.disk_usage(WORKSPACE_DIR)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        
        info = [
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"CPU Count: {os.cpu_count()}",
            f"Workspace Disk Usage: {total_gb:.1f}GB total, {used_gb:.1f}GB used, {free_gb:.1f}GB free"
        ]
        return "\n".join(info)
    except Exception as e:
        return f"Error: {e}"

register_lc_tool(get_current_time, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(calculate, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(convert_timezone, metadata=ToolMetadata(risk_level="safe"))
register_lc_tool(system_info, metadata=ToolMetadata(risk_level="safe"))
