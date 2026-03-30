"""
Tool Implementations for the AI Agent

Each tool is a function that takes structured input and returns a string result.
These are the "hands" of the agent -- the things it can actually DO.

Tools:
    1. calculator: evaluate math expressions safely
    2. file_reader: read and return file contents
    3. database_query: run SQL against the sample SQLite database
    4. web_search: simulated web search (returns mock results for demo)
"""

import os
import sqlite3
import math


# ============================================================
# Tool 1: Calculator
# ============================================================

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, **, (), sqrt, sin, cos, pi, etc.
    Does NOT support: arbitrary code execution.
    """
    # Only allow safe math operations
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e, "pow": pow,
    }

    try:
        # Remove any non-math characters for safety
        for char in expression:
            if char not in "0123456789+-*/.() ,epilogsqrtaboundmaxcn":
                if not char.isalpha():
                    pass  # allow digits and operators
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ============================================================
# Tool 2: File Reader
# ============================================================

def file_reader(filepath: str) -> str:
    """Read and return the contents of a file.

    Limited to files in the current project directory for safety.
    """
    # Security: only allow reading files in the project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.abspath(filepath)

    if not abs_path.startswith(base_dir):
        return f"Error: access denied. Can only read files within {base_dir}"

    if not os.path.exists(abs_path):
        return f"Error: file not found: {filepath}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate very long files
        if len(content) > 5000:
            content = content[:5000] + f"\n... (truncated, {len(content)} total characters)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


# ============================================================
# Tool 3: Database Query
# ============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), "sample.db")


def database_query(sql: str) -> str:
    """Execute a read-only SQL query against the sample database.

    Only SELECT queries are allowed for safety.
    """
    # Security: only allow SELECT queries
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return "Error: only SELECT queries are allowed."

    if not os.path.exists(DB_PATH):
        return "Error: database not found. Run setup_db.py first."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned 0 rows."

        # Format as a readable table
        columns = rows[0].keys()
        header = " | ".join(columns)
        separator = "-+-".join("-" * len(c) for c in columns)
        lines = [header, separator]

        for row in rows[:50]:  # limit to 50 rows
            lines.append(" | ".join(str(row[c]) for c in columns))

        result = "\n".join(lines)
        if len(rows) > 50:
            result += f"\n... ({len(rows)} total rows, showing first 50)"

        return result

    except Exception as e:
        return f"SQL Error: {e}"


# ============================================================
# Tool 4: Web Search (Simulated)
# ============================================================

# Mock search results for demo purposes
MOCK_RESULTS = {
    "python": [
        {"title": "Python 3.12 Released", "snippet": "Python 3.12.0 was released on October 2, 2023. Python 1.0 was released in January 1994."},
        {"title": "Python Official Website", "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively."},
    ],
    "weather": [
        {"title": "Current Weather", "snippet": "Today's weather: 72°F (22°C), partly cloudy, humidity 45%."},
    ],
    "default": [
        {"title": "Search Result", "snippet": "This is a simulated search result. In production, connect to a real search API like Tavily or SerpAPI."},
    ],
}


def web_search(query: str) -> str:
    """Search the web for information.

    NOTE: This is a simulated search for demo purposes.
    In production, replace with a real search API (Tavily, SerpAPI, etc.)
    """
    # Find matching mock results
    query_lower = query.lower()
    results = MOCK_RESULTS.get("default")

    for keyword, mock_results in MOCK_RESULTS.items():
        if keyword in query_lower:
            results = mock_results
            break

    # Format results
    lines = [f"Search results for: '{query}'\n"]
    for i, r in enumerate(results):
        lines.append(f"{i+1}. {r['title']}")
        lines.append(f"   {r['snippet']}\n")

    lines.append("(Note: these are simulated results for demonstration)")
    return "\n".join(lines)


# ============================================================
# Tool Registry
# ============================================================

# Maps tool names to their implementations and Anthropic tool schemas
TOOL_REGISTRY = {
    "calculator": {
        "function": calculator,
        "schema": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /), exponentiation (**), and functions (sqrt, sin, cos, log). Use this for any math computation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '15 * 0.85' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    "file_reader": {
        "function": file_reader,
        "schema": {
            "name": "file_reader",
            "description": "Read the contents of a file. Use this to read documents, data files, or configuration files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    "database_query": {
        "function": database_query,
        "schema": {
            "name": "database_query",
            "description": "Execute a read-only SQL SELECT query against the sample database. Tables: customers (id, name, email, city, joined_date), orders (id, customer_id, product, amount, order_date), products (id, name, category, price, stock).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    "web_search": {
        "function": web_search,
        "schema": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date facts, news, or information not available in the database or files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
}


def get_tool_schemas() -> list[dict]:
    """Get all tool schemas for the Anthropic API."""
    return [tool["schema"] for tool in TOOL_REGISTRY.values()]


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool by name with the given input."""
    if name not in TOOL_REGISTRY:
        return f"Error: unknown tool '{name}'"

    func = TOOL_REGISTRY[name]["function"]

    # Map the input schema fields to function arguments
    try:
        if name == "calculator":
            return func(input_data["expression"])
        elif name == "file_reader":
            return func(input_data["filepath"])
        elif name == "database_query":
            return func(input_data["sql"])
        elif name == "web_search":
            return func(input_data["query"])
        else:
            return f"Error: no handler for tool '{name}'"
    except Exception as e:
        return f"Error executing {name}: {e}"
