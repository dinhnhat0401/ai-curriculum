<p align="center">
  <img src="https://img.shields.io/badge/Duration-16_Weeks-blue?style=for-the-badge" alt="Duration">
  <img src="https://img.shields.io/badge/Level-Senior_Dev_→_AI_Engineer-green?style=for-the-badge" alt="Level">
  <img src="https://img.shields.io/badge/Projects-12+_Builds-orange?style=for-the-badge" alt="Projects">
  <img src="https://img.shields.io/badge/Code-100%25_From_Scratch-red?style=for-the-badge" alt="Code">
</p>

# From Software Engineer to AI Engineer

**The complete, hands-on curriculum for senior developers who want to build production AI systems.**

No hand-waving. No "just call the API." You build everything from scratch first -- autograd engines, GPT models, RAG pipelines, AI agents -- then learn the frameworks. By week 16, you can architect, build, evaluate, and ship enterprise AI.

---

## The Philosophy

```
Week 1:  You train your first neural network.
Week 4:  You build GPT from scratch and understand every line.
Week 8:  You build production RAG, agents, and evaluation pipelines.
Week 12: You deploy on AWS with monitoring, guardrails, and CI/CD.
Week 16: You have a product, a demo, and a business.
```

**Build first. Framework second. Ship third.**

Every module follows the same pattern: understand the theory, build it from scratch, then (and only then) use the tools that abstract it away. When something breaks in production at 2am, you'll know exactly where to look.

---

## The Roadmap

```
                         FROM SOFTWARE ENGINEER TO AI ENGINEER
                         =====================================

    PHASE 1                PHASE 2                PHASE 3                PHASE 4
    The Engine Room        Production AI          Business Stack         Your Weapon
    Weeks 1-4              Weeks 5-8              Weeks 9-12             Weeks 13-16
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │              │       │              │       │              │       │              │
    │  MNIST       │       │  Model       │       │  AWS         │       │  Discovery   │
    │  Micrograd   │──────>│  Comparison  │──────>│  Bedrock     │──────>│  Demo Build  │
    │  Makemore    │       │  RAG         │       │  Production  │       │  Materials   │
    │  NanoGPT     │       │  Agents      │       │  Evaluation  │       │  Launch      │
    │              │       │  Fine-tuning │       │  Cost Opt.   │       │              │
    └──────────────┘       │  Evaluation  │       └──────────────┘       └──────────────┘
                           └──────────────┘
    You understand          You can build          You can ship           You have a
    the engine.             what clients           enterprise AI.         business.
                            pay for.
```

---

## What You Will Build

| # | Project | What It Is | Key Concepts |
|---|---------|-----------|--------------|
| 1 | **MNIST Classifier** | Your "hello world" -- train a neural net to read handwritten digits | Tensors, forward pass, loss functions, SGD, batching |
| 2 | **Micrograd** | Autograd engine from scratch -- backpropagation in 100 lines | Computational graphs, chain rule, gradient descent |
| 3 | **Makemore** | Character-level language model -- generates new words | Embeddings, bigrams, MLPs, batch normalization |
| 4 | **NanoGPT** | GPT from scratch -- the transformer architecture, fully implemented | Self-attention, multi-head attention, positional encoding |
| 5 | **Model Comparator** | Test GPT-4, Claude, and open-source models side by side | API patterns, prompt engineering, cost/quality trade-offs |
| 6 | **RAG System** | Retrieval-Augmented Generation from scratch, then with frameworks | Chunking, embeddings, vector stores, retrieval, generation |
| 7 | **AI Agent** | Agent with tool use -- reasons, plans, and executes multi-step tasks | Function calling, ReAct loop, error handling |
| 8 | **Fine-tuning Pipeline** | Fine-tune an open-source model with LoRA | LoRA, dataset prep, training, evaluation |
| 9 | **Eval Pipeline** | Automated evaluation system for any AI application | Metrics, LLM-as-judge, regression testing |
| 10 | **AWS Bedrock Deploy** | Enterprise RAG on AWS managed services | Bedrock, Knowledge Bases, SageMaker |
| 11 | **Production API** | Dockerized, monitored, CI/CD-deployed AI service | FastAPI, Docker, monitoring, guardrails |
| 12 | **The Demo** | Your product -- solves a real problem for real users | Full stack AI application |

---

## Prerequisites

| Requirement | Level Needed | Why |
|------------|-------------|-----|
| **Python** | Intermediate+ | All code is Python. You should be comfortable with classes, decorators, generators. |
| **Git** | Basic | Version control. You'll commit every project. |
| **Command Line** | Comfortable | Environment setup, running scripts, Docker. |
| **Math** | High school calculus | Derivatives and chain rule. Linear algebra helps but isn't required -- we build intuition first. |
| **ML Experience** | None required | That's what this curriculum teaches. |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-curriculum.git
cd ai-curriculum

# Create a Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install Phase 1 dependencies
pip install torch torchvision matplotlib numpy

# Start with MNIST
cd phase-1/mnist
python hello.py
```

You just trained a neural network. Now go read `phase-1/mnist/README.md` to understand what happened.

---

## Directory Structure

```
ai-curriculum/
│
├── README.md                  # You are here
├── CURRICULUM.md              # Detailed 16-week study plan
├── GLOSSARY.md                # 80+ AI/ML terms defined
├── CHEATSHEET.md              # Copy-paste code patterns
├── RESOURCES.md               # Books, papers, videos, courses
│
├── phase-1/                   # THE ENGINE ROOM (Weeks 1-4)
│   ├── README.md              # Phase 1 overview and learning path
│   ├── mnist/                 # Project 1: Hello World neural network
│   │   ├── README.md          # Detailed guide with theory + walkthrough
│   │   └── hello.py           # Complete MNIST classifier
│   ├── micrograd/             # Project 2: Autograd engine from scratch
│   │   ├── README.md          # The most important guide in the curriculum
│   │   ├── engine.py          # Value class with automatic differentiation
│   │   ├── nn.py              # Neuron, Layer, MLP built on Value
│   │   └── train.py           # Training loop demonstration
│   ├── makemore/              # Project 3: Character-level language model
│   │   ├── README.md          # Embeddings, bigrams, MLPs, BatchNorm
│   │   ├── bigram.py          # Bigram model (counting + neural)
│   │   ├── mlp.py             # MLP language model (Bengio 2003)
│   │   └── names.txt          # Training dataset
│   └── nanogpt/               # Project 4: GPT from scratch
│       ├── README.md          # Transformer architecture deep dive
│       ├── gpt.py             # Complete GPT implementation
│       └── data/input.txt     # Shakespeare training data
│
├── phase-2/                   # PRODUCTION AI (Weeks 5-8)
│   ├── README.md              # Phase 2 overview
│   ├── model-comparison/      # Project 5: Multi-model benchmarking
│   │   ├── README.md          # LLM landscape, API patterns, prompting
│   │   ├── compare.py         # Side-by-side model comparison tool
│   │   └── prompt_engineering.py  # Prompting techniques demo
│   ├── rag-scratch/           # Project 6a: RAG from scratch
│   │   ├── README.md          # The definitive RAG guide
│   │   ├── chunker.py         # Document chunking strategies
│   │   ├── embeddings.py      # Embedding generation + similarity
│   │   ├── vector_store.py    # In-memory vector store
│   │   ├── rag.py             # Complete RAG pipeline
│   │   ├── evaluate.py        # RAG evaluation framework
│   │   └── sample_docs/       # Test documents
│   ├── rag-framework/         # Project 6b: RAG with LangChain + LlamaIndex
│   │   ├── README.md          # Framework comparison guide
│   │   ├── langchain_rag.py   # LangChain implementation
│   │   └── llamaindex_rag.py  # LlamaIndex implementation
│   ├── agent/                 # Project 7: AI agent with tool use
│   │   ├── README.md          # Agent architectures and patterns
│   │   ├── agent.py           # Complete agent with 4 tools
│   │   ├── tools.py           # Tool implementations
│   │   └── setup_db.py        # Sample database creation
│   ├── fine-tuning/           # Project 8: Model fine-tuning
│   │   ├── README.md          # When and how to fine-tune
│   │   ├── prepare_dataset.py # Dataset creation and formatting
│   │   └── finetune_lora.py   # LoRA fine-tuning with PEFT
│   └── evaluation/            # Project 9: Evaluation pipeline
│       ├── README.md          # Evaluation methodology guide
│       ├── eval_pipeline.py   # Complete eval framework
│       └── metrics.py         # Reusable metric functions
│
├── phase-3/                   # BUSINESS STACK (Weeks 9-12)
│   ├── README.md              # Phase 3 overview
│   ├── aws-bedrock/           # Project 10: AWS Bedrock deployment
│   │   ├── README.md          # Enterprise AI on AWS
│   │   ├── bedrock_chat.py    # Bedrock API usage
│   │   └── bedrock_rag.py     # Bedrock Knowledge Bases
│   ├── production/            # Project 11: Production deployment
│   │   ├── README.md          # Production engineering guide
│   │   ├── app.py             # FastAPI application
│   │   ├── Dockerfile         # Production container
│   │   └── docker-compose.yml # Local development setup
│   └── evaluation/            # Cost optimization + eval at scale
│       ├── README.md          # Cost optimization strategies
│       ├── cost_optimizer.py  # Model routing, caching, batching
│       └── eval_suite.py      # Production evaluation suite
│
├── phase-4/                   # YOUR WEAPON (Weeks 13-16)
│   ├── README.md              # Phase 4 overview
│   ├── discovery/             # Finding problems worth solving
│   │   ├── README.md          # Discovery methodology
│   │   ├── discovery_template.md  # Call template
│   │   └── opportunity_scorecard.md  # Scoring framework
│   ├── demo/                  # The product
│   │   ├── README.md          # Building demos that sell
│   │   └── streamlit_app.py   # Demo application template
│   └── materials/             # Sales materials
│       ├── README.md          # Packaging your work
│       ├── case_study_template.md   # Case study format
│       └── proposal_template.md     # Project proposal format
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | PyTorch |
| **LLM APIs** | Anthropic (Claude), OpenAI (GPT-4), Ollama (local models) |
| **Embeddings** | OpenAI text-embedding-3-small, sentence-transformers |
| **Vector Stores** | ChromaDB, FAISS (in-memory) |
| **RAG Frameworks** | LangChain, LlamaIndex |
| **Fine-tuning** | Hugging Face Transformers, PEFT, TRL |
| **API Framework** | FastAPI |
| **Frontend** | Streamlit |
| **Cloud** | AWS Bedrock, SageMaker, S3, Lambda |
| **Infrastructure** | Docker, GitHub Actions |
| **Evaluation** | Custom pipeline + LLM-as-judge |

---

## How to Use This Repo

**If you're following the full curriculum:**
1. Read `CURRICULUM.md` for the week-by-week plan
2. Work through each phase in order -- they build on each other
3. Each module has its own `README.md` with theory, code walkthrough, and exercises
4. Track your progress with GitHub Issues (run `setup.sh` to create them)

**If you're here for a specific topic:**
- Want to understand transformers? Start at `phase-1/nanogpt/`
- Want to build RAG? Start at `phase-2/rag-scratch/`
- Want to build agents? Start at `phase-2/agent/`
- Want production deployment? Start at `phase-3/production/`

**If you're browsing for reference:**
- `GLOSSARY.md` -- every AI/ML term defined
- `CHEATSHEET.md` -- copy-paste code patterns
- `RESOURCES.md` -- curated books, papers, videos, courses

---

## Progress Tracking

Run the setup script to create GitHub Issues and Milestones for every module:

```bash
chmod +x setup.sh && ./setup.sh
```

This creates 20 trackable issues across 4 milestones. Close them as you complete each module.

---

## The Curriculum at a Glance

| Week | Module | You Build | Hours |
|------|--------|-----------|-------|
| 1 | MNIST + Foundations | Neural network classifier from scratch | ~5h |
| 2 | Micrograd | Autograd engine + backpropagation | ~5h |
| 3 | Makemore | Character-level language model with embeddings | ~6h |
| 4 | NanoGPT | GPT transformer from scratch | ~6h |
| 5 | Model Comparison | Multi-provider LLM benchmarking tool | ~5h |
| 6 | RAG (scratch + framework) | Full RAG pipeline, then rebuild with LangChain | ~8h |
| 7 | AI Agent | Agent with tool use and multi-step reasoning | ~8h |
| 8 | Fine-tuning + Evaluation | LoRA fine-tuning + automated eval pipeline | ~8h |
| 9 | AWS Bedrock | Enterprise RAG on managed AWS services | ~8h |
| 10 | Production Deploy | Dockerized API with monitoring and CI/CD | ~8h |
| 11 | Eval + Cost Optimization | Production eval suite + cost reduction strategies | ~6h |
| 12 | Certifications | AWS GenAI Developer + Salesforce Agentforce | ~10h |
| 13-14 | Discovery + Demo | Find a real problem, build the solution | ~20h |
| 15 | Materials | Case study, demo video, presentation | ~6h |
| 16 | Launch | Outreach, pilot offers, first clients | ~5h |
| | | **Total** | **~114h** |

---

## Contributing

Found an error? Have a better explanation? PRs are welcome.

## License

MIT -- use this curriculum however you want.

---

<p align="center">
  <i>Built for engineers who learn by building.</i>
</p>
