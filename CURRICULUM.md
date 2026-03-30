# The 16-Week Curriculum

Detailed week-by-week study plan. Follow in order -- each week builds on the last.

---

## Phase 1: The Engine Room (Weeks 1-4)

> After this phase, you understand how neural networks learn. Not conceptually -- at the code level.

---

### Week 1: First Contact with Neural Networks

**Time: ~5 hours**

**Learning Objectives:**
- Train a neural network and understand every line of the code
- Explain what tensors, gradients, and loss functions are
- Internalize the training loop pattern (forward - loss - backward - update)
- Understand why we split data into train/test sets

**Watch (1.5h):**
- Karpathy "Intro to Large Language Models" (1h) -- the big picture of what LLMs are
- 3Blue1Brown "But what is a neural network?" (Ch 1) -- visual intuition

**Read (30min):**
- Karpathy "The Unreasonable Effectiveness of RNNs" blog post -- the precursor to everything

**Build (3h):**
- `phase-1/mnist/` -- Train a neural network on MNIST handwritten digits
- Start from the code in `hello.py`, then experiment:
  - Change hidden layer sizes (32, 128, 784, 2048)
  - Add a third layer
  - Try different learning rates (0.1, 0.001, 0.00001)
  - Remove normalization and see what breaks

**You know you're done when:**
- [ ] Your model achieves >95% accuracy on MNIST
- [ ] You can explain the training loop without looking at code
- [ ] You can explain what a gradient is to a non-technical person
- [ ] You modified the architecture and observed the effects

---

### Week 2: Backpropagation from Scratch (Micrograd)

**Time: ~5 hours**

**THIS IS THE MOST IMPORTANT WEEK IN THE ENTIRE CURRICULUM.**

**Learning Objectives:**
- Understand automatic differentiation at the code level
- Build a working autograd engine from scratch
- Understand the chain rule and how gradients flow backward
- Build a neural network on top of your autograd engine

**Watch (2.5h):**
- Karpathy "The spelled-out intro to neural networks and backpropagation: building micrograd"
- Code along in a Jupyter notebook as you watch

**Watch (1h):**
- 3Blue1Brown Ch 2 "Gradient descent", Ch 3 "Backpropagation", Ch 4 "Backpropagation calculus"

**Build (2h):**
- Close the video. Rebuild micrograd from memory.
- `phase-1/micrograd/` -- `engine.py` (Value class), `nn.py` (Neuron/Layer/MLP), `train.py`
- If you get stuck, look at the code briefly, then close it and try again

**Exercises:**
1. Add a ReLU activation function to `engine.py`
2. Add a sigmoid activation function
3. Trace a tiny network (2 inputs, 1 hidden, 1 output) forward and backward by hand with actual numbers
4. Compare your micrograd network's output with an equivalent PyTorch network

**You know you're done when:**
- [ ] You rebuilt micrograd without looking at the source
- [ ] You can explain: "What is a gradient? Why do we go backward?"
- [ ] You understand why `self.grad += out.grad * ...` uses `+=` not `=`
- [ ] You understand why we zero gradients before each backward pass

---

### Week 3: Language Modeling (Makemore)

**Time: ~6 hours**

**Learning Objectives:**
- Understand what a language model is (probability distribution over sequences)
- Build bigram and MLP language models
- Understand embeddings -- the most important concept for RAG and production AI
- Understand batch normalization and training dynamics

**Watch (4h):**
- Karpathy "makemore" Part 1: Bigram model
- Karpathy "makemore" Part 2: MLP (Bengio et al. 2003)
- Karpathy "makemore" Part 3: Activations, gradients, BatchNorm

**Build (2h):**
- `phase-1/makemore/bigram.py` -- Build a bigram model, then a neural bigram
- `phase-1/makemore/mlp.py` -- Build an MLP language model with embeddings

**Key Concept -- Embeddings:**
This is the concept that connects Phase 1 to everything else. An embedding is a learned dense vector representation of a discrete token. When you build RAG systems in Phase 2, embedding is how documents become searchable. Understand this deeply.

**Exercises:**
1. Train on names from different cultures -- do the models capture different patterns?
2. Visualize the 2D embedding space -- which characters cluster together?
3. Vary context window size (2, 3, 5, 8) and compare generated names
4. Implement learning rate decay and compare convergence

**You know you're done when:**
- [ ] Your model generates plausible-looking new names
- [ ] You can explain what an embedding is and why it matters
- [ ] You understand why bigram is a "model" (it assigns probabilities to sequences)
- [ ] You can explain batch normalization intuitively

---

### Week 4: GPT from Scratch (NanoGPT)

**Time: ~6 hours**

**Learning Objectives:**
- Understand the transformer architecture at the code level
- Implement self-attention, multi-head attention, and feed-forward networks
- Understand positional encoding, residual connections, and layer normalization
- Train a character-level GPT and generate text

**Watch (2h):**
- Karpathy "Let's build GPT: from scratch, in code, spelled out"
- 3Blue1Brown "Attention in transformers, visually explained"

**Read (2h):**
- "Attention Is All You Need" (Vaswani et al., 2017) -- first pass while watching the video
- Read again after building nanoGPT -- it will click now

**Build (2h):**
- `phase-1/nanogpt/gpt.py` -- Complete GPT implementation
- Train on Shakespeare, generate text
- Experiment with hyperparameters:
  - Change n_layer (1, 2, 4, 8)
  - Change n_head (1, 2, 4)
  - Change block_size (32, 64, 128, 256)

**Exercises:**
1. Train on a different text corpus (Python code, song lyrics, etc.)
2. Visualize attention patterns -- which tokens attend to which?
3. Remove positional embeddings -- what breaks?
4. Remove the causal mask -- what happens?

**You know you're done when:**
- [ ] nanoGPT generates coherent-ish text
- [ ] You can explain self-attention in your own words
- [ ] You can map every component in the paper to your code
- [ ] You understand: "The transformer is just embedding + [attention + FFN] x N + output"

**PHASE 1 COMPLETE.** You understand the engine. Everything from here is applied.

---

## Phase 2: Production AI Engineering (Weeks 5-8)

> After this phase, you can build everything clients pay for: RAG, agents, fine-tuning, evaluation.

---

### Week 5: The LLM Landscape + API Mastery

**Time: ~5 hours**

**Learning Objectives:**
- Understand the major LLM providers and their trade-offs
- Master LLM API patterns: streaming, tool use, structured output
- Learn prompt engineering fundamentals
- Build a multi-model comparison tool

**Read (1h):**
- Anthropic prompt engineering docs: https://docs.anthropic.com
- OpenAI API docs: https://platform.openai.com/docs

**Build (4h):**
- `phase-2/model-comparison/compare.py` -- Compare GPT-4, Claude, and open-source models
- `phase-2/model-comparison/prompt_engineering.py` -- 5 prompting strategies on the same task
- Test with 15+ diverse prompts, document results

**Key API Patterns to Master:**
1. Basic completion (synchronous)
2. Streaming (token-by-token for UX)
3. System prompts (setting behavior)
4. Temperature control (0 = deterministic, 1 = creative)
5. Structured output (JSON mode)
6. Tool use / function calling
7. Retry with exponential backoff

**You know you're done when:**
- [ ] You can call 3 different LLM APIs from memory
- [ ] You have a documented comparison of quality/speed/cost
- [ ] You can explain when to use which model and why
- [ ] You understand prompt engineering beyond "just ask nicely"

---

### Week 6: RAG -- The Most Important Build

**Time: ~8 hours**

**Learning Objectives:**
- Build a complete RAG system from scratch (no frameworks)
- Understand every component: chunking, embedding, retrieval, generation
- Then rebuild with LangChain/LlamaIndex to understand what frameworks hide
- Evaluate RAG quality with metrics

**Build Day 1 -- From Scratch (5h):**
- `phase-2/rag-scratch/chunker.py` -- 3 chunking strategies
- `phase-2/rag-scratch/embeddings.py` -- Embedding generation
- `phase-2/rag-scratch/vector_store.py` -- Simple vector store
- `phase-2/rag-scratch/rag.py` -- Complete pipeline
- `phase-2/rag-scratch/evaluate.py` -- Test with 20 questions

**Build Day 2 -- With Frameworks (3h):**
- `phase-2/rag-framework/langchain_rag.py` -- Same RAG with LangChain
- `phase-2/rag-framework/llamaindex_rag.py` -- Same RAG with LlamaIndex
- Compare: what's easier? what's harder? what did the framework hide?

**Read:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- Anthropic's guide on RAG best practices

**You know you're done when:**
- [ ] Your from-scratch RAG answers questions correctly on your test docs
- [ ] You can explain chunking strategy trade-offs to a client
- [ ] You understand the full pipeline: document -> chunk -> embed -> store -> retrieve -> generate
- [ ] You know when RAG works and when it fails

---

### Week 7: AI Agents with Tool Use

**Time: ~8 hours**

**Learning Objectives:**
- Build an agent that reasons about which tools to use
- Implement the agent loop (the core pattern)
- Handle multi-step tasks requiring multiple tool calls
- Understand agent failure modes and how to handle them

**Read (1h):**
- Anthropic "Building Effective Agents": https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems
- Anthropic tool use docs

**Build (7h):**
- `phase-2/agent/tools.py` -- 4 tool implementations
- `phase-2/agent/setup_db.py` -- Sample database
- `phase-2/agent/agent.py` -- Complete agent with tool use
- Test with 5+ multi-step tasks

**The Agent Loop (memorize this):**
```python
while not done:
    response = llm(messages)
    if response has tool_calls:
        results = execute_tools(response.tool_calls)
        messages.append(tool_results)
    else:
        return response.text  # done
```

**Exercises:**
1. Add a 5th tool and test if the agent discovers when to use it
2. Create a task that requires 3+ sequential tool calls
3. Intentionally break a tool and verify the agent handles the error
4. Add logging to see the agent's "reasoning" at each step

**You know you're done when:**
- [ ] Agent handles multi-step tasks with 3+ tools
- [ ] You've seen it fail and understand why
- [ ] You have error handling and a maximum iteration limit
- [ ] You can explain ReAct (Reason + Act) pattern

---

### Week 8: Fine-tuning + Evaluation Pipeline

**Time: ~8 hours**

**Learning Objectives:**
- Know when to fine-tune vs when to use RAG vs when to just prompt better
- Fine-tune a small model with LoRA
- Build an automated evaluation pipeline (the most important production tool)
- Evaluate with metrics + LLM-as-judge

**Build -- Fine-tuning (4h):**
- `phase-2/fine-tuning/prepare_dataset.py` -- Create and format training data
- `phase-2/fine-tuning/finetune_lora.py` -- LoRA fine-tuning with Hugging Face PEFT

**Build -- Evaluation (4h):**
- `phase-2/evaluation/metrics.py` -- Reusable metric functions
- `phase-2/evaluation/eval_pipeline.py` -- Complete eval framework with LLM-as-judge
- Run evaluation on your RAG system, agent, or fine-tuned model

**Key Decision Framework:**
```
Need specific knowledge?  → RAG
Need specific behavior?   → Fine-tuning
Need specific output?     → Prompt engineering
Need all three?           → RAG + fine-tuned model + good prompts
```

**You know you're done when:**
- [ ] You fine-tuned a model and it produces better task-specific output
- [ ] Your eval pipeline produces a clear pass/fail report
- [ ] You can answer: "How do I know it works?" with data

**PHASE 2 COMPLETE.** You can build the things clients pay for.

---

## Phase 3: Business Stack (Weeks 9-12)

> After this phase, you can ship enterprise AI on real infrastructure.

---

### Week 9: AWS Bedrock

**Time: ~8 hours**

**Learning Objectives:**
- Use AWS Bedrock for enterprise-grade AI deployment
- Build managed RAG with Bedrock Knowledge Bases
- Understand enterprise requirements: data residency, IAM, VPC
- Compare managed vs self-hosted approaches

**Setup (1h):**
- AWS account with Bedrock access enabled
- Install and configure AWS CLI + boto3

**Build (7h):**
- `phase-3/aws-bedrock/bedrock_chat.py` -- Call Claude on Bedrock
- `phase-3/aws-bedrock/bedrock_rag.py` -- Bedrock Knowledge Bases RAG
- Document comparison: your from-scratch RAG vs Bedrock managed RAG

**You know you're done when:**
- [ ] You can call models on Bedrock via boto3
- [ ] Knowledge Base works with your documents
- [ ] You have a documented comparison of self-hosted vs managed

---

### Week 10: Production Deployment

**Time: ~8 hours**

**Learning Objectives:**
- Containerize an AI application with Docker
- Build a production API with FastAPI
- Implement monitoring, guardrails, and error handling
- Set up CI/CD with GitHub Actions

**Build (8h):**
- `phase-3/production/app.py` -- FastAPI application with all production features
- `phase-3/production/Dockerfile` -- Multi-stage production build
- `phase-3/production/docker-compose.yml` -- Local development setup
- Tests, health checks, structured logging

**Production Checklist:**
- [ ] Containerized (Docker)
- [ ] API with proper error handling
- [ ] Request/response logging
- [ ] Token usage and cost tracking
- [ ] Rate limiting
- [ ] Input validation / prompt injection detection
- [ ] Health check endpoint
- [ ] CI/CD pipeline

**You know you're done when:**
- [ ] `docker-compose up` starts your entire service
- [ ] You can hit the API and get responses
- [ ] Every request is logged with latency and token count

---

### Week 11: Evaluation at Scale + Cost Optimization

**Time: ~6 hours**

**Learning Objectives:**
- Build production evaluation suite with regression detection
- Implement cost optimization strategies
- Calculate cost per query at scale

**Build (6h):**
- `phase-3/evaluation/eval_suite.py` -- Production eval with regression detection
- `phase-3/evaluation/cost_optimizer.py` -- Model routing, semantic caching, batch processing

**Cost Optimization Strategies:**
1. **Prompt caching** -- Anthropic supports native caching of system prompts
2. **Model routing** -- Haiku for easy queries, Sonnet for hard ones
3. **Batch processing** -- Async batch API for non-real-time tasks
4. **Response length control** -- Constrain output tokens where appropriate

**You know you're done when:**
- [ ] Eval suite catches regressions automatically
- [ ] At least 2 cost optimizations implemented and measured
- [ ] You can tell a client: "This costs $X per 1000 queries with Y% accuracy"

---

### Week 12: Certifications

**Time: ~10 hours**

**Certifications:**
1. **AWS Certified Generative AI Developer Professional** -- 85 questions, 205 min, $300
2. **Salesforce Agentforce Specialist** -- $200

**Prep:**
- Review all your builds from Phases 1-3
- AWS Skill Builder practice exams
- Salesforce Trailhead modules

**You know you're done when:**
- [ ] At least one certification passed
- [ ] Added to LinkedIn profile

**PHASE 3 COMPLETE.** You can deliver enterprise AI on real infrastructure.

---

## Phase 4: Your Weapon (Weeks 13-16)

> After this phase, you have a product, a demo, and a business.

---

### Weeks 13-14: Discovery + Demo Build

**Time: ~20 hours**

**Discovery (ongoing since Phase 2):**
- 5+ conversations with real companies documented
- Pain points identified and patterns found
- ONE workflow chosen as your target

**Build the Demo:**
- `phase-4/demo/streamlit_app.py` -- Working application
- Solves a real, validated pain point
- Works on real data
- Has evaluation metrics you can show
- Handles edge cases gracefully

**You know you're done when:**
- [ ] Someone who isn't you can understand what the demo does
- [ ] It works reliably, not just on cherry-picked examples
- [ ] You have metrics: accuracy, latency, cost per query

---

### Week 15: Package Your Work

**Time: ~6 hours**

**Deliverables:**
- [ ] 5-minute demo video (Loom)
- [ ] One-page case study
- [ ] 30-minute presale presentation
- [ ] LinkedIn content announcing your practice

---

### Week 16: Launch

**Time: ~5 hours**

- Send outreach to 10 companies
- Book 3+ meetings
- Close 1+ pilot engagement
- Start delivering

**PHASE 4 COMPLETE.** You have a cert, a demo, a pipeline, and a business.

---

## Ongoing Habits

**Daily (15 min):**
- Read Simon Willison's blog
- Scan Hacker News for AI/ML posts

**Weekly (2-3 hours):**
- One research paper or technical deep dive
- Ship one improvement to your demo or pipeline

**Monthly:**
- Rebuild one component with newer tools/models
- Re-run evaluation benchmarks
- Review and update your materials
