#!/bin/bash
# =============================================================
# AI Curriculum - GitHub Project Setup
# =============================================================
# Prerequisites:
#   brew install gh
#   gh auth login
#
# Usage:
#   1. Create a new repo on GitHub (e.g. ai-curriculum)
#   2. Clone it locally
#   3. Run this script from inside the repo:
#      chmod +x setup.sh && ./setup.sh
# =============================================================

set -e

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Setting up project in: $REPO"

# --- Labels ---
echo "Creating labels..."
for entry in \
  "phase-1|0E8A16|Phase 1 - How LLMs work" \
  "phase-2|1D76DB|Phase 2 - Production AI engineering" \
  "phase-3|D93F0B|Phase 3 - Business stack and certs" \
  "phase-4|B60205|Phase 4 - Build your sales weapon" \
  "ongoing|5319E7|Ongoing habits" \
  "watch|C2E0C6|Video or lecture" \
  "build|0075CA|Hands-on build task" \
  "read|FEF2C0|Reading material" \
  "cert|F9D0C4|Certification" \
  "milestone-project|FF6600|Key deliverable"; do
  name=$(echo "$entry" | cut -d'|' -f1)
  color=$(echo "$entry" | cut -d'|' -f2)
  desc=$(echo "$entry" | cut -d'|' -f3)
  gh label create "$name" --color "$color" --description "$desc" --force 2>/dev/null || true
done

# --- Milestones ---
echo "Creating milestones..."
gh api repos/$REPO/milestones -f title="Phase 1: The Engine Room" -f description="Understand how LLMs work at the code level" -f due_on="$(date -v+4w +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d '+4 weeks' +%Y-%m-%dT00:00:00Z)" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 2: Production AI" -f description="Build production-grade RAG, agents, fine-tuning" -f due_on="$(date -v+8w +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d '+8 weeks' +%Y-%m-%dT00:00:00Z)" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 3: Business Stack" -f description="AWS, MLOps, Salesforce, certs" -f due_on="$(date -v+12w +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d '+12 weeks' +%Y-%m-%dT00:00:00Z)" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 4: Your Weapon" -f description="Build the demo, launch the business" -f due_on="$(date -v+16w +%Y-%m-%dT00:00:00Z 2>/dev/null || date -d '+16 weeks' +%Y-%m-%dT00:00:00Z)" 2>/dev/null || true

echo "Creating issues..."

# =============================================================
# PHASE 1: THE ENGINE ROOM (Weeks 1-4)
# =============================================================

gh issue create \
  --title "1.1 - Watch: Karpathy Intro to Large Language Models" \
  --label "phase-1,watch" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Watch Andrej Karpathy's 1-hour overview of LLMs.

**Link:** https://www.youtube.com/watch?v=zjkBMFhNj_g

## What to pay attention to
- How LLMs are trained (pretraining vs fine-tuning)
- What LLMs can and can't do
- Security implications
- The 'LLM OS' concept - this is what agents are becoming

## Done when
- [ ] Watched the full video
- [ ] Can explain pretraining vs fine-tuning in your own words
- [ ] Can explain what a token is and why tokenization matters

**Week 1 - ~1 hour**"

gh issue create \
  --title "1.2 - Watch: 3Blue1Brown neural networks series" \
  --label "phase-1,watch" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Watch 3Blue1Brown's neural network series for visual intuition.

**Links:**
- Ch 1: But what is a neural network?
- Ch 2: Gradient descent
- Ch 3: Backpropagation
- Ch 4: Backpropagation calculus
- Bonus: Attention in transformers, visually explained

Search '3blue1brown neural networks' on YouTube.

## Done when
- [ ] Watched all 4 + the attention video
- [ ] Can draw a simple neural network and explain forward pass
- [ ] Understand what gradient descent is doing intuitively (the ball rolling downhill)
- [ ] Understand why attention is the key innovation

**Week 1 - ~1.5 hours**"

gh issue create \
  --title "1.3 - Build: First neural net on MNIST" \
  --label "phase-1,build" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Install PyTorch and train a basic neural network on MNIST.

\`\`\`bash
pip install torch torchvision
\`\`\`

## Steps
1. Load MNIST dataset using torchvision
2. Build a simple 2-layer neural net (no CNN, just Linear layers)
3. Train it for a few epochs
4. Check accuracy on test set
5. Visualize some predictions (correct + incorrect)

## Why this matters
This is your hello world for deep learning. The same training loop concept (forward - loss - backward - update) is used in every model from MNIST to GPT-4.

## Done when
- [ ] PyTorch installed and working
- [ ] Model trains and achieves >95% accuracy
- [ ] You can modify the architecture (add layers, change sizes) and see the effect
- [ ] Code committed to this repo: phase-1/mnist/

**Week 1 - ~3 hours**"

gh issue create \
  --title "1.4 - Read: Karpathy Unreasonable Effectiveness of RNNs" \
  --label "phase-1,read" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Read Karpathy's classic blog post on character-level language models.

**Link:** https://karpathy.github.io/2015/05/21/rnn-effectiveness/

## Why
This is the precursor to everything. It shows how neural nets can learn to generate text character by character - Shakespeare, LaTeX, C code, music. Simple idea, mind-blowing results. Sets the stage for understanding why transformers were such a leap.

## Done when
- [ ] Read the full post
- [ ] Understand what an RNN does at a high level
- [ ] Understand why this was exciting and what the limitations were (context window, vanishing gradients)

**Week 1 - ~30 min**"

gh issue create \
  --title "1.5 - Build: micrograd (backprop from scratch)" \
  --label "phase-1,build,milestone-project" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Watch Karpathy's micrograd lecture and code along. Then rebuild from memory.

**Video:** The spelled-out intro to neural networks and backpropagation: building micrograd (~2.5hrs)

**Repo:** https://github.com/karpathy/micrograd

## Steps
1. Watch the video, code along in a Jupyter notebook
2. Close the video
3. Rebuild micrograd from scratch without looking at the code
4. If you get stuck, check the code, then close it and try again

## This is the most important exercise in the entire curriculum.
If you understand backpropagation at the code level, everything else is just scale.

## Done when
- [ ] Watched + coded along
- [ ] Rebuilt from memory (even partially is fine)
- [ ] Can explain in plain words: what is a gradient? why do we go backward?
- [ ] Code committed: phase-1/micrograd/

**Week 2 - ~5 hours**"

gh issue create \
  --title "1.6 - Build: makemore parts 1-3 (language modeling)" \
  --label "phase-1,build" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Follow Karpathy's makemore series - build character-level language models.

**Videos:**
- Part 1: Bigram model
- Part 2: MLP (following Bengio et al. 2003)
- Part 3: Activations, gradients, BatchNorm

**Repo:** https://github.com/karpathy/makemore

## What you learn
- Tokenization at the character level
- Embeddings - how discrete tokens become continuous vectors
- The MLP architecture - precursor to transformers
- Why BatchNorm exists and what it fixes
- How to read loss curves and debug training

## Done when
- [ ] Built bigram, MLP, and BatchNorm models
- [ ] Can explain what an embedding is (this will come up in every client conversation about RAG)
- [ ] Code committed: phase-1/makemore/

**Week 3 - ~6 hours**"

gh issue create \
  --title "1.7 - Read: Attention Is All You Need (first pass)" \
  --label "phase-1,read" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Read the original transformer paper. First pass - get the architecture shape, skip the math you don't understand yet.

**Link:** https://arxiv.org/abs/1706.03762

## Reading strategy
1. Read the abstract and introduction carefully
2. Look at Figure 1 (the architecture diagram) - this is the most important figure in modern AI
3. Read Section 3 (Model Architecture) - focus on understanding Q, K, V intuitively
4. Skim the rest - training details, results
5. Watch Yannic Kilcher's paper walkthrough on YouTube for a guided tour

## Done when
- [ ] Read once through
- [ ] Can identify the main components: encoder, decoder, self-attention, multi-head attention, feed-forward, positional encoding
- [ ] Have questions written down for the second pass after building nanoGPT

**Week 3 - ~2 hours**"

gh issue create \
  --title "1.8 - Build: nanoGPT (GPT from scratch)" \
  --label "phase-1,build,milestone-project" \
  --milestone "Phase 1: The Engine Room" \
  --body "## Task
Follow Karpathy's 'Let's build GPT from scratch' and build a working GPT.

**Video:** Let's build GPT: from scratch, in code, spelled out (~2hrs)
**Repo:** https://github.com/karpathy/nanoGPT

## Steps
1. Code along with the video
2. Train nanoGPT on the Shakespeare dataset (included)
3. Try training on a custom dataset - Japanese text, financial docs, whatever interests you
4. Experiment: change number of layers, heads, embedding dim. See what happens.

## Then re-read Attention Is All You Need - second pass
It will click now. You built the thing the paper describes.

## Done when
- [ ] nanoGPT built and trained
- [ ] Generated text output (doesn't need to be good - the point is it works)
- [ ] Re-read the transformer paper - concepts are solid
- [ ] Can explain self-attention in your own words to a non-technical person
- [ ] Code committed: phase-1/nanogpt/

**Week 4 - ~6 hours**

PHASE 1 COMPLETE - You understand the engine. Everything from here is applied."

# =============================================================
# PHASE 2: PRODUCTION AI ENGINEERING (Weeks 5-8)
# =============================================================

gh issue create \
  --title "2.1 - Build: Multi-model API comparison" \
  --label "phase-2,build" \
  --milestone "Phase 2: Production AI" \
  --body "## Task
Build a simple app that calls 3 different LLM APIs for the same task. Compare.

## Models to test
1. **OpenAI** (GPT-4o or GPT-4.1) - via API
2. **Anthropic** (Claude Sonnet) - via API
3. **Open-source** (Llama 3 or Mistral) - via Ollama locally

## Build
A Python script that:
- Takes a prompt as input
- Sends it to all 3 models
- Records: response text, latency, token count, cost
- Outputs a comparison table

Test with 10-20 diverse prompts (simple Q&A, summarization, code generation, Japanese text).

## Read
- Anthropic prompt engineering docs: https://docs.anthropic.com
- OpenAI production best practices: https://platform.openai.com/docs

## Done when
- [ ] Script works with all 3 providers
- [ ] Have a documented comparison of quality/speed/cost
- [ ] Understand API patterns: streaming, system prompts, temperature, max tokens
- [ ] Code committed: phase-2/model-comparison/

**Week 5 - ~5 hours**"

gh issue create \
  --title "2.2 - Build: RAG system from scratch (no framework)" \
  --label "phase-2,build,milestone-project" \
  --milestone "Phase 2: Production AI" \
  --body "## Task
Build a complete RAG system without LangChain or LlamaIndex. Raw code.

## Architecture
Documents - Chunking - Embeddings - Vector Store - Retrieval - LLM - Response

## Steps
1. **Document loading** - Start with 5-10 PDF or text documents. Use something real (Japanese business docs, financial reports, whatever you have).
2. **Chunking** - Write your own chunker. Experiment with:
   - Fixed size (500 tokens, 1000 tokens)
   - Overlap (100 token overlap vs none)
   - Semantic chunking (split on paragraphs/sections)
3. **Embeddings** - Use OpenAI text-embedding-3-small or open-source via sentence-transformers
4. **Vector store** - Use Qdrant (Docker) or pgvector (if you already have Postgres)
5. **Retrieval** - Implement similarity search. Try top-3, top-5, top-10. See what works.
6. **Generation** - Feed retrieved chunks + question to Claude or GPT. Compare results.
7. **Evaluation** - Manually test 20 questions. How often does it hallucinate? Miss context? Give wrong answers?

## Then rebuild with a framework
Use LangChain or LlamaIndex to build the same thing. Now you know what the framework hides.

## Done when
- [ ] Raw RAG system works end-to-end
- [ ] Tested with 20+ questions, documented accuracy
- [ ] Framework version works
- [ ] Can explain chunking strategy trade-offs to a client
- [ ] Code committed: phase-2/rag-scratch/ and phase-2/rag-framework/

**Week 6 - ~8 hours**

This is the most important build in the curriculum. RAG is 70% of enterprise AI projects."

gh issue create \
  --title "2.3 - Build: AI agent with tool use" \
  --label "phase-2,build,milestone-project" \
  --milestone "Phase 2: Production AI" \
  --body "## Task
Build an agent that can reason about which tools to use and execute multi-step tasks.

## Tools to give your agent
Pick 3-4:
- Web search (via Tavily or SerpAPI)
- Database query (SQL against a local SQLite DB)
- Calculator / code execution
- File reader (read and summarize docs)
- API call (weather, stock price, whatever)

## Implementation path
1. **Start with Claude tool use** - Use Anthropic's tool use API. Define tools as JSON schemas. Let Claude decide when to call them.
2. **Build your own agent loop:**
   while not done:
     response = llm(messages + tool_results)
     if response has tool_calls:
       execute tools, append results
     else:
       return final answer
3. **Add complexity:** Multi-step tasks where the agent needs to call tool A, use its output to decide whether to call tool B or C.

## Read
- Anthropic Building Effective Agents: https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems
- OpenAI Practical Guide to Building Agents

## Watch
- Sam Witteveen's agent tutorials on YouTube

## Done when
- [ ] Agent handles multi-step tasks with 3+ tools
- [ ] You've seen it fail and understand why (hallucinated tool calls, infinite loops, wrong tool selection)
- [ ] Added basic error handling and retry logic
- [ ] Code committed: phase-2/agent/

**Week 7 - ~8 hours**"

gh issue create \
  --title "2.4 - Build: Fine-tune a model + evaluation pipeline" \
  --label "phase-2,build" \
  --milestone "Phase 2: Production AI" \
  --body "## Task
Fine-tune a small open-source model and build an evaluation system.

## Part 1: Fine-tuning
1. Pick a small model: Llama 3 8B, Mistral 7B, or Phi-3
2. Create a dataset: 500-1000 instruction/response pairs for a specific task
3. Fine-tune using LoRA with Hugging Face PEFT:
   pip install transformers peft trl datasets bitsandbytes
4. Compare: base model vs fine-tuned model on your task

## Part 2: Evaluation pipeline
Build a system that:
1. Takes a test set of 50 question/expected-answer pairs
2. Runs them through your model (or RAG system, or agent)
3. Scores each answer:
   - Automated: BLEU, ROUGE, exact match where applicable
   - LLM-as-judge: Use Claude to grade answers on faithfulness (1-5) and relevance (1-5)
4. Outputs a report with pass rate, average scores, worst failures

## Why this matters
The number 1 question from enterprise clients: How do I know it works?
This evaluation pipeline IS the answer.

## Done when
- [ ] Fine-tuned model runs and generates better task-specific output than base
- [ ] Evaluation pipeline produces a clear report
- [ ] Can explain to a client: when to fine-tune vs when to use RAG vs when to just prompt better
- [ ] Code committed: phase-2/fine-tuning/ and phase-2/evaluation/

**Week 8 - ~8 hours**

PHASE 2 COMPLETE - You can build the things clients pay for."

# =============================================================
# PHASE 3: BUSINESS STACK (Weeks 9-12)
# =============================================================

gh issue create \
  --title "3.1 - Build: RAG system on AWS Bedrock" \
  --label "phase-3,build" \
  --milestone "Phase 3: Business Stack" \
  --body "## Task
Rebuild your RAG system on AWS using managed services.

## Steps
1. Set up AWS account (if you don't have one)
2. **Bedrock Knowledge Bases:**
   - Upload your documents to S3
   - Create a Knowledge Base with Bedrock
   - Test retrieval quality vs your from-scratch version
3. **Bedrock Agents:**
   - Create an agent with tool use via Bedrock
   - Connect it to your Knowledge Base
   - Add a custom action (Lambda function)
4. **SageMaker deployment:**
   - Deploy an open-source model to a SageMaker endpoint
   - Test latency and cost

## Compare
| Metric | From-scratch | AWS Bedrock |
|--------|-------------|-------------|
| Setup time | ? | ? |
| Retrieval quality | ? | ? |
| Cost per 1000 queries | ? | ? |
| Maintenance burden | ? | ? |

## Done when
- [ ] Bedrock Knowledge Base works with your docs
- [ ] Bedrock Agent handles multi-step queries
- [ ] SageMaker endpoint serves an open-source model
- [ ] Comparison documented
- [ ] Code committed: phase-3/aws-bedrock/

**Week 9 - ~8 hours**"

gh issue create \
  --title "3.2 - Build: Production-ready deployment" \
  --label "phase-3,build,milestone-project" \
  --milestone "Phase 3: Business Stack" \
  --body "## Task
Take your best project and make it production-grade.

## Production checklist
- [ ] **Containerized** - Dockerfile that builds and runs
- [ ] **API** - FastAPI or similar with proper error handling
- [ ] **Monitoring** - Logging every request/response, latency, token usage
- [ ] **Guardrails** - Input validation, content filtering, PII detection
- [ ] **Cost tracking** - Track API costs per request, daily/weekly rollups
- [ ] **Rate limiting** - Protect against runaway costs
- [ ] **CI/CD** - GitHub Actions pipeline: lint - test - build - deploy
- [ ] **Health check** - Endpoint that confirms the service is alive
- [ ] **Error handling** - Graceful failures, retry logic for API timeouts
- [ ] **Documentation** - README that someone else could deploy from

## Read
- Relevant chapters from LLM Engineer's Handbook (Iusztin and Labonne)

## Done when
- [ ] Service runs in Docker
- [ ] CI/CD pipeline passes
- [ ] Can show monitoring dashboard with real metrics
- [ ] Code committed: phase-3/production/

**Week 10 - ~8 hours**"

gh issue create \
  --title "3.3 - Build: Evaluation suite + cost optimization" \
  --label "phase-3,build" \
  --milestone "Phase 3: Business Stack" \
  --body "## Task
Build a comprehensive evaluation + cost optimization system.

## Evaluation suite
Expand your week 8 eval pipeline:
- [ ] **Faithfulness** - Is the answer grounded in provided sources?
- [ ] **Relevance** - Did it answer the actual question?
- [ ] **Completeness** - Did it miss important information?
- [ ] **Harmfulness** - Does it generate anything inappropriate?
- [ ] **Latency** - P50, P95, P99 response times
- [ ] **Cost** - Cost per query at current volume, projected cost at 10x

## Cost optimization experiments
Try each and measure impact on quality + cost:
- [ ] **Prompt caching** - Cache system prompts (Anthropic supports this natively)
- [ ] **Model routing** - Cheap model (Haiku) for easy queries, expensive (Sonnet/Opus) for hard ones
- [ ] **Batch processing** - Async batch for non-real-time tasks
- [ ] **Chunking optimization** - Smaller chunks = fewer tokens in context = cheaper
- [ ] **Response length control** - Constrain output tokens where appropriate

## Done when
- [ ] Eval suite runs automatically and produces a report
- [ ] At least 2 cost optimizations implemented and measured
- [ ] Can tell a client: This costs X yen per 1000 queries with Y% accuracy
- [ ] Code committed: phase-3/evaluation/

**Week 11 - ~6 hours**"

gh issue create \
  --title "3.4 - Cert: Salesforce Agentforce Specialist" \
  --label "phase-3,cert,build" \
  --milestone "Phase 3: Business Stack" \
  --body "## Task
Get certified and build a working Agentforce demo.

## Steps
1. **Get a Salesforce Developer Edition org:** https://developer.salesforce.com/signup
2. **Study via Trailhead:**
   - Agentforce Specialist exam guide
   - Complete the recommended Trailmixes
   - Focus on: Agent Builder, Prompt Builder, topics and actions
3. **Build a demo agent:**
   - Customer inquiry routing agent
   - Or: Lead qualification agent
   - Or: Internal knowledge base assistant
   - Must work in Japanese (test with Japanese inputs)
4. **Take the exam** (USD 200)

## Why this matters
Japanese enterprises use Salesforce. This cert + demo = immediate credibility and a concrete thing to sell.

## Done when
- [ ] Agentforce Specialist certification passed
- [ ] Working demo agent in your dev org
- [ ] 3-minute screen recording of the demo
- [ ] Can explain Agentforce architecture to a Salesforce admin

**Week 12 - ~10 hours**

PHASE 3 COMPLETE - You can deliver enterprise AI on real infrastructure."

# =============================================================
# PHASE 4: YOUR WEAPON (Weeks 13-16)
# =============================================================

gh issue create \
  --title "4.1 - Discovery: Document pain points from real conversations" \
  --label "phase-4,milestone-project" \
  --milestone "Phase 4: Your Weapon" \
  --body "## Task
By now you should have had 5-10 discovery conversations with real companies. Document the patterns.

## Template for each conversation
Company:
Industry:
Size:
Current tech stack:
Pain point described:
Their words (exact quotes):
What they have tried:
Estimated cost of the problem (in time or money):
Could AI solve this? How?
Would they pay for a solution? How much?

## Pattern analysis
After 5+ conversations:
- What pain points keep coming up?
- Which ones are AI-solvable with current technology?
- Which ones do your skills uniquely qualify you for?
- Which ones have the biggest budget?

**Pick ONE.** That is what you build.

## Done when
- [ ] 5+ conversations documented
- [ ] Patterns identified
- [ ] ONE workflow chosen as your target
- [ ] Written in: phase-4/discovery/notes.md

**Week 13 - ongoing (start during phase 2)**"

gh issue create \
  --title "4.2 - Build: The demo (your sales weapon)" \
  --label "phase-4,build,milestone-project" \
  --milestone "Phase 4: Your Weapon" \
  --body "## Task
Build the end-to-end solution for the workflow you chose in 4.1.

## Requirements
- [ ] Solves a real, validated pain point
- [ ] Uses RAG and/or agents (whatever the problem requires)
- [ ] Runs on AWS Bedrock or Salesforce Agentforce
- [ ] Has a clean UI (Streamlit, Next.js, or similar - doesn't need to be beautiful, needs to be demo-able)
- [ ] Handles Japanese language natively
- [ ] Has evaluation metrics you can show (accuracy, latency, cost)
- [ ] Includes at least basic monitoring and error handling

## This is not a prototype
This is the thing you walk into a presale meeting with and say let me show you. It should work reliably on live data.

## Done when
- [ ] End-to-end system works
- [ ] Tested with real (or realistic) data
- [ ] Evaluation shows measurable results
- [ ] Someone who isn't you can understand what it does by looking at the UI
- [ ] Code committed: phase-4/demo/

**Weeks 13-14 - ~20 hours**"

gh issue create \
  --title "4.3 - Package: Record demo, write case study, build landing page" \
  --label "phase-4,build" \
  --milestone "Phase 4: Your Weapon" \
  --body "## Task
Turn your demo into sales materials.

## Deliverables
- [ ] **5-min Loom video** walking through the demo. Script it:
  1. Here is the problem (30 sec)
  2. Here is what we built (60 sec)
  3. Here is how it works (120 sec)
  4. Here are the results (60 sec)
  5. Here is what it costs (30 sec)
- [ ] **One-page case study** (Japanese + English versions)
  - Problem - Solution - Results - Tech stack
  - Even if the case is your own internal project, write it as if it were a client engagement
- [ ] **LinkedIn post** in Japanese announcing your AI automation practice
- [ ] **30-min presale presentation** template (Keynote/Google Slides)
  - Reusable across prospects with minor customization

## Done when
- [ ] All 4 deliverables created
- [ ] Someone has reviewed the video and confirmed it is clear
- [ ] Materials committed: phase-4/materials/

**Week 15 - ~6 hours**"

gh issue create \
  --title "4.4 - Launch: First 10 outreach + pilot offer" \
  --label "phase-4,milestone-project" \
  --milestone "Phase 4: Your Weapon" \
  --body "## Task
Reach out to 10 companies. Offer pilot engagements.

## Outreach list
- [ ] Company 1:
- [ ] Company 2:
- [ ] Company 3:
- [ ] Company 4:
- [ ] Company 5:
- [ ] Company 6:
- [ ] Company 7:
- [ ] Company 8:
- [ ] Company 9:
- [ ] Company 10:

## Criteria
- Japanese company with 100-2000 employees
- Uses Salesforce or has significant manual workflows
- Someone in your network can intro you (warm > cold)

## The offer
We are looking for 2-3 pilot companies to work with on AI workflow automation. In exchange for being a pilot and providing a case study, we offer a steep discount on standard pricing. The engagement is fixed-scope, fixed-price, delivered in 6 weeks.

## Done when
- [ ] 10 outreach messages sent
- [ ] At least 3 meetings booked
- [ ] At least 1 pilot signed
- [ ] Tracking in: phase-4/pipeline.md

**Week 16**

PHASE 4 COMPLETE - You have a cert, a demo, a pipeline, and a business."

gh issue create \
  --title "4.5 - Cert: AWS GenAI Developer Professional" \
  --label "phase-4,cert" \
  --milestone "Phase 4: Your Weapon" \
  --body "## Task
Take the AWS Certified Generative AI Developer Professional exam.

You have been building on AWS since week 9. The knowledge is there. Now formalize it.

## Prep
- Stephane Maarek Udemy course (structured review)
- AWS Skill Builder practice exams
- Review your own Bedrock/SageMaker projects from phase 3

## Exam details
- 85 questions, 205 minutes
- USD 300
- Passing: 750/1000

## Done when
- [ ] Exam passed
- [ ] Cert added to LinkedIn
- [ ] Business card updated

**Week 16 - exam day**"

# =============================================================
# ONGOING
# =============================================================

gh issue create \
  --title "Ongoing: Daily reading habit (15 min)" \
  --label "ongoing,read" \
  --body "## Daily habit - never close this issue

### Every day, 15 minutes:
- Simon Willison's blog: https://simonwillison.net
- Hacker News front page - scan for AI/ML posts

### Track interesting finds here
Add comments to this issue with links and one-sentence notes on anything useful.

This is how you stay current without drowning in content."

gh issue create \
  --title "Ongoing: Weekly deep dive (2-3 hrs)" \
  --label "ongoing,read" \
  --body "## Weekly habit - never close this issue

### Every week, pick ONE:
- A new research paper relevant to your current project
- A technical blog post from Anthropic, OpenAI, or Meta engineering
- A module from Maxime Labonne's LLM Course: https://github.com/mlabonne/llm-course
- A chapter from one of the reference books

### Log your reads here
Add a comment each week with what you read and key takeaway."

gh issue create \
  --title "Ongoing: Monthly rebuild + review" \
  --label "ongoing,build" \
  --body "## Monthly habit - never close this issue

### Every month:
1. Rebuild one component of your stack with newer tools/models
2. Re-run your evaluation benchmarks - are results improving?
3. Ship one improvement to your demo
4. Review pipeline - where are you stuck? What is working?

### Log updates here
Add a comment each month with what you rebuilt and what you learned."

# =============================================================
# README + Directory Structure
# =============================================================

cat > README.md << 'READMEEOF'
# AI Curriculum - From Foundations to B2B Delivery

16-week curriculum for a senior developer entering B2B AI consulting.

## Structure

```
phase-1/          # The engine room - how LLMs work (weeks 1-4)
  mnist/
  micrograd/
  makemore/
  nanogpt/
phase-2/          # Production AI engineering (weeks 5-8)
  model-comparison/
  rag-scratch/
  rag-framework/
  agent/
  fine-tuning/
  evaluation/
phase-3/          # Business stack + certs (weeks 9-12)
  aws-bedrock/
  production/
  evaluation/
phase-4/          # Your weapon (weeks 13-16)
  discovery/
  demo/
  materials/
```

## Progress

Track progress via GitHub Issues and Milestones.

## Milestones

- Phase 1: The Engine Room - Understand transformers at the code level
- Phase 2: Production AI - Build RAG, agents, fine-tuning, evaluation
- Phase 3: Business Stack - AWS, MLOps, Salesforce, certs
- Phase 4: Your Weapon - Build the demo, launch the business
READMEEOF

mkdir -p phase-1/{mnist,micrograd,makemore,nanogpt}
mkdir -p phase-2/{model-comparison,rag-scratch,rag-framework,agent,fine-tuning,evaluation}
mkdir -p phase-3/{aws-bedrock,production,evaluation}
mkdir -p phase-4/{discovery,demo,materials}

find . -type d -empty -not -path './.git/*' -exec touch {}/.gitkeep \;

git add -A
git commit -m "Initial curriculum setup: README, directory structure"
git push

echo ""
echo "============================================="
echo "Done! Created:"
echo "  - Labels (10)"
echo "  - Milestones (4)"
echo "  - Issues (20)"
echo "  - Directory structure"
echo "  - README.md"
echo "============================================="
echo ""
echo "View your issues:"
echo "  gh issue list"
echo ""
echo "Open in browser:"
echo "  gh browse"
echo "============================================="