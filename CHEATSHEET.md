# Cheatsheet

Copy-paste code patterns for every major task in this curriculum.

---

## Python Environment Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Phase 1 dependencies
pip install torch torchvision matplotlib numpy

# Phase 2 dependencies
pip install anthropic openai chromadb langchain llama-index

# Phase 3 dependencies
pip install fastapi uvicorn boto3 pydantic

# Phase 4 dependencies
pip install streamlit

# API keys (add to ~/.zshrc or ~/.bashrc)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

---

## PyTorch Essentials

### Training Loop (THE pattern)

```python
for epoch in range(epochs):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Define a Model

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### Save / Load Model

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## Anthropic Claude API

### Basic Call

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
```

### Streaming

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Tool Use

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)

# Check if model wants to use a tool
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")
```

---

## OpenAI API

### Basic Call

```python
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world", "Goodbye world"],
)
vectors = [item.embedding for item in response.data]
```

---

## RAG Patterns

### Cosine Similarity

```python
import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### RAG Prompt Template

```python
prompt = f"""Answer based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_chunks}

Question: {user_question}
Answer:"""
```

---

## Docker

```bash
# Build
docker build -t my-ai-app .

# Run
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY my-ai-app

# Docker Compose
docker-compose up        # start
docker-compose down      # stop
docker-compose up --build  # rebuild and start
```

---

## AWS Bedrock (boto3)

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
    inferenceConfig={"maxTokens": 512},
)

print(response["output"]["message"]["content"][0]["text"])
```

---

## FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def query(req: Query):
    answer = call_llm(req.question)  # your LLM call here
    return {"answer": answer}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}
```

---

## LLM-as-Judge Template

```python
judge_prompt = f"""Rate this answer on a scale of 1-5:
- Faithfulness: Is it grounded in the context?
- Relevance: Does it answer the question?

Question: {question}
Expected: {expected}
Actual: {actual}

Respond as JSON: {{"faithfulness": N, "relevance": N}}"""
```

---

## Git Workflow

```bash
# Start a new module
git checkout -b phase-1/mnist
# ... work ...
git add phase-1/mnist/
git commit -m "Complete MNIST module with training + evaluation"
git push -u origin phase-1/mnist
```
