"""
AI Agent with Tool Use

A complete agent implementation using Anthropic Claude's tool use API.
The agent can reason about which tools to use and execute multi-step tasks.

Tools available:
    1. calculator - evaluate math expressions
    2. file_reader - read file contents
    3. database_query - SQL queries on sample database
    4. web_search - simulated web search

Usage:
    python setup_db.py   # create the sample database first
    python agent.py      # run demo tasks

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import json
from tools import get_tool_schemas, execute_tool

MAX_ITERATIONS = 10  # Safety limit to prevent infinite loops


def run_agent(user_message: str, verbose: bool = True) -> str:
    """Run the agent loop for a single user message.

    The agent will:
    1. Send the message to Claude with available tools
    2. If Claude wants to use a tool, execute it and send the result back
    3. Repeat until Claude gives a final text response or we hit the iteration limit
    """
    try:
        import anthropic
    except ImportError:
        return "Error: pip install anthropic"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "Error: set ANTHROPIC_API_KEY environment variable"

    client = anthropic.Anthropic()
    tools = get_tool_schemas()

    messages = [{"role": "user", "content": user_message}]

    system_prompt = """You are a helpful assistant with access to tools. Use them when needed to answer questions accurately. Always show your reasoning.

When using the database_query tool, write clean SQL. The database has these tables:
- customers (id, name, email, city, joined_date)
- orders (id, customer_id, product, amount, order_date)
- products (id, name, category, price, stock)

When doing calculations, use the calculator tool instead of doing math in your head.
When you have enough information to answer, provide a clear final response."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Agent Task: {user_message}")
        print(f"{'='*60}")

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        if verbose:
            print(f"Stop reason: {response.stop_reason}")

        # Process the response
        if response.stop_reason == "tool_use":
            # Claude wants to use tools
            # The response may contain both text and tool_use blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    if verbose:
                        print(f"  Tool: {tool_name}")
                        print(f"  Input: {json.dumps(tool_input, indent=2)}")

                    # Execute the tool
                    result = execute_tool(tool_name, tool_input)

                    if verbose:
                        result_preview = result[:200] + "..." if len(result) > 200 else result
                        print(f"  Result: {result_preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result,
                    })

            # Send tool results back to Claude
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Claude is done -- extract final text
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            if verbose:
                print(f"\n{'='*60}")
                print(f"Final Answer: {final_text}")
                print(f"{'='*60}")

            return final_text

        else:
            return f"Unexpected stop reason: {response.stop_reason}"

    return "Error: agent reached maximum iteration limit without completing the task."


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    demo_tasks = [
        # Task 1: Simple calculation
        "What is 15% of 847.50?",

        # Task 2: Database query
        "How many customers are in the database?",

        # Task 3: Database + calculation (multi-step)
        "What is the total revenue from all orders? And what is the average order value?",

        # Task 4: Complex multi-step reasoning
        "Find the customer who has spent the most money. What city are they from, and what is their average order amount?",

        # Task 5: Web search + calculation
        "Search for when Python was first released, and calculate how many years ago that was from 2025.",
    ]

    print("AI AGENT DEMO")
    print("=" * 60)
    print(f"Tools available: calculator, file_reader, database_query, web_search")
    print(f"Max iterations per task: {MAX_ITERATIONS}")

    for i, task in enumerate(demo_tasks):
        print(f"\n\n{'#'*60}")
        print(f"# TASK {i+1}")
        print(f"{'#'*60}")

        result = run_agent(task, verbose=True)
