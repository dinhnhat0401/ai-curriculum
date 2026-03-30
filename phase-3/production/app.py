"""
Production AI API Service

A FastAPI application with production features:
- Chat and query endpoints
- Structured logging
- Token and cost tracking
- Rate limiting
- Health check
- Error handling
- CORS configuration

Usage:
    pip install fastapi uvicorn pydantic
    uvicorn app:app --reload
    # or: python app.py
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Logging Setup (structured JSON)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # JSON will be the message itself
)
logger = logging.getLogger("ai-api")


def log_request(request_id: str, **kwargs):
    """Log a structured JSON entry for every request."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        **kwargs,
    }
    logger.info(json.dumps(entry))


# ============================================================
# Simple Rate Limiter (in-memory)
# ============================================================

class RateLimiter:
    """Simple in-memory rate limiter. Per-IP, sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Remove expired entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# ============================================================
# Cost Tracker
# ============================================================

class CostTracker:
    """Track token usage and estimated costs."""

    COST_PER_1M = {"input": 3.00, "output": 15.00}  # Claude Sonnet pricing

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    @property
    def total_cost(self) -> float:
        return (
            self.total_input_tokens * self.COST_PER_1M["input"] / 1_000_000
            + self.total_output_tokens * self.COST_PER_1M["output"] / 1_000_000
        )

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_per_request": round(self.total_cost / max(self.total_requests, 1), 6),
        }


cost_tracker = CostTracker()

# ============================================================
# Request/Response Models
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000, description="The question to answer")
    max_tokens: int = Field(default=1024, ge=1, le=4096)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    metadata: dict = {}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    cost_summary: dict

# ============================================================
# App Setup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(json.dumps({"event": "startup", "timestamp": datetime.utcnow().isoformat()}))
    yield
    logger.info(json.dumps({"event": "shutdown", "timestamp": datetime.utcnow().isoformat()}))


app = FastAPI(
    title="AI Query API",
    description="Production AI service with RAG-powered question answering",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (configure for your frontend domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Use for monitoring and load balancer checks."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        cost_summary=cost_tracker.summary(),
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    """Answer a question using the AI system."""
    request_id = f"req_{int(time.time()*1000)}"
    client_ip = request.client.host if request.client else "unknown"
    start = time.time()

    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        log_request(request_id, event="rate_limited", client_ip=client_ip)
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Input validation
    if len(req.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Call the LLM (replace with your RAG pipeline in production)
        answer, input_tokens, output_tokens = _generate_answer(req.question, req.max_tokens)

        latency_ms = (time.time() - start) * 1000
        cost_tracker.add(input_tokens, output_tokens)

        # Log the request
        log_request(
            request_id,
            event="query",
            question=req.question[:100],
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return QueryResponse(
            answer=answer,
            sources=[],
            metadata={
                "request_id": request_id,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": "claude-sonnet-4-20250514",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log_request(request_id, event="error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")


def _generate_answer(question: str, max_tokens: int) -> tuple[str, int, int]:
    """Generate an answer using Claude API. Returns (answer, input_tokens, output_tokens)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system="You are a helpful assistant. Answer concisely and accurately.",
            messages=[{"role": "user", "content": question}],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
    else:
        # Fallback for demo without API key
        return (
            f"(Demo mode - no API key) You asked: {question}",
            len(question.split()) * 2,
            20,
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
