# AI Agent with Tool Use

**Time: ~8 hours | Files: `agent.py`, `tools.py`, `setup_db.py`**

---

## What is an AI Agent?

An LLM that can **use tools, make decisions, and take actions**. Not just Q&A -- it *does* things.

```
    Regular LLM:    Question ──> Answer

    Agent:          Question ──> Think ──> Use Tool ──> Observe ──> Think ──> Use Tool ──> Answer
                                  │                       │
                                  └── "I need to search   └── "The search returned X,
                                       the database"           now I need to calculate..."
```

## The Agent Loop (Memorize This)

```python
messages = [{"role": "user", "content": user_question}]

while iterations < MAX_ITERATIONS:
    response = llm(messages, tools=available_tools)

    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call.name, tool_call.input)
            messages.append(tool_result)
    else:
        return response.text  # Agent is done -- return final answer
```

This is the core pattern. The LLM decides what to do. If it needs a tool, it calls one. If it has enough information, it responds. The loop continues until the agent produces a final answer or hits a safety limit.

## Tool Use / Function Calling

LLMs call tools by generating structured JSON that matches a schema you provide:

```python
tools = [{
    "name": "calculate",
    "description": "Evaluate a mathematical expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g., '15 * 0.85'"}
        },
        "required": ["expression"]
    }
}]
```

The LLM reads the tool name, description, and parameter schema to decide when and how to use it. **Tool descriptions are prompts** -- write them clearly.

## Agent Architectures

| Architecture | How It Works | Best For |
|-------------|-------------|---------|
| **ReAct** | Think -> Act -> Observe -> Repeat | General-purpose tool use |
| **Plan-and-Execute** | Make full plan first, then execute steps | Complex multi-step tasks |
| **Router** | Classify query, route to specialist | High-volume with known categories |
| **Multi-Agent** | Orchestrator delegates to specialist agents | Complex systems with clear divisions |

## Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Hallucinated tool calls | Model invents tools that don't exist | Strict tool definitions, validation |
| Infinite loops | Agent keeps calling tools without converging | Max iteration limit |
| Wrong tool selection | Using search when calculator was needed | Better tool descriptions |
| Parameter hallucination | Making up file paths, URLs | Input validation, constrained params |
| Error cascading | Tool error confuses the agent | Error handling, clear error messages |

## Files in This Module

| File | What It Does |
|------|-------------|
| `agent.py` | Complete agent with tool use loop using Anthropic Claude |
| `tools.py` | 4 tool implementations (calculator, file reader, DB query, web search) |
| `setup_db.py` | Creates and populates a sample SQLite database |

## Exercises

1. Add a 5th tool and test if the agent discovers when to use it
2. Create a task that requires 3+ sequential tool calls
3. Intentionally break a tool and verify the agent handles the error gracefully
4. Add detailed logging to trace the agent's reasoning
5. Implement a maximum cost limit (stop if token usage exceeds threshold)
6. Build a "human-in-the-loop" mode that asks for confirmation before executing tools
7. Try the same multi-step task with different models and compare tool use accuracy
8. Implement a simple memory system (agent can remember facts across turns)
