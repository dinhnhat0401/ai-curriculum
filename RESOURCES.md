# Resources

Curated books, papers, videos, courses, and tools referenced throughout the curriculum.

---

## Essential Videos

| Video | Creator | Duration | Relevant Module |
|-------|---------|----------|----------------|
| Intro to Large Language Models | Andrej Karpathy | 1h | Week 1 |
| But what is a neural network? (Ch 1-4) | 3Blue1Brown | 1.5h | Week 1 |
| Attention in transformers, visually explained | 3Blue1Brown | 25min | Week 4 |
| Building micrograd | Andrej Karpathy | 2.5h | Week 2 |
| Building makemore (Parts 1-3) | Andrej Karpathy | ~4h | Week 3 |
| Let's build GPT from scratch | Andrej Karpathy | 2h | Week 4 |
| Let's build the GPT tokenizer | Andrej Karpathy | 2h | Supplemental |

## Essential Papers

| Paper | Year | Key Takeaway | Difficulty |
|-------|------|-------------|-----------|
| Attention Is All You Need (Vaswani et al.) | 2017 | The transformer architecture | Medium |
| BERT (Devlin et al.) | 2019 | Bidirectional pre-training for NLP | Medium |
| Language Models are Few-Shot Learners (GPT-3) | 2020 | Scale enables in-context learning | Medium |
| LoRA (Hu et al.) | 2021 | Efficient fine-tuning via low-rank adapters | Medium |
| RAG (Lewis et al.) | 2020 | Retrieval-augmented generation for knowledge tasks | Medium |
| ReAct (Yao et al.) | 2022 | Reasoning + Acting in LLM agents | Easy |
| Chain-of-Thought Prompting (Wei et al.) | 2022 | "Let's think step by step" improves reasoning | Easy |
| Constitutional AI (Bai et al.) | 2022 | Self-improving AI safety | Hard |
| Scaling Laws for Neural LMs (Kaplan et al.) | 2020 | Predictable performance from scale | Hard |

## Essential Blog Posts

| Post | Author | Why Read It |
|------|--------|-------------|
| The Unreasonable Effectiveness of RNNs | Andrej Karpathy | The precursor to modern LLMs |
| The Illustrated Transformer | Jay Alammar | Best visual explanation of transformers |
| Attention? Attention! | Lilian Weng | Comprehensive attention mechanism survey |
| Building LLM applications for production | Chip Huyen | Practical production advice |
| Prompt Engineering Guide | DAIR.AI | Comprehensive prompting techniques |
| Simon Willison's Blog | Simon Willison | Stay current on AI developments (read daily) |
| Anthropic Research Blog | Anthropic | Understanding Claude's capabilities |

## Books

| Book | Author(s) | Best For |
|------|-----------|---------|
| Deep Learning | Goodfellow, Bengio, Courville | Theoretical reference (free online) |
| Hands-On Machine Learning | Aurelien Geron | Practical ML with scikit-learn + TF |
| Build a Large Language Model from Scratch | Sebastian Raschka | Step-by-step LLM construction |
| LLM Engineer's Handbook | Paul Iusztin, Maxime Labonne | Production LLM systems |
| Designing Machine Learning Systems | Chip Huyen | ML system design for production |

## Online Courses

| Course | Platform | Best For |
|--------|----------|---------|
| Practical Deep Learning | fast.ai | Practical top-down learning |
| CS231n: CNNs for Visual Recognition | Stanford | Deep learning fundamentals |
| CS224n: NLP with Deep Learning | Stanford | NLP and transformer models |
| LLM Course | Maxime Labonne (GitHub) | Comprehensive LLM engineering |
| Hugging Face NLP Course | Hugging Face | Transformers library + NLP tasks |
| Generative AI with LLMs | DeepLearning.AI + AWS | Production generative AI |

## Tools and Libraries

### Core ML
- **PyTorch** -- The deep learning framework (what we use throughout)
- **Hugging Face Transformers** -- Pre-trained models and tokenizers
- **Hugging Face PEFT** -- Parameter-efficient fine-tuning (LoRA)

### LLM APIs
- **Anthropic SDK** -- `pip install anthropic`
- **OpenAI SDK** -- `pip install openai`
- **Ollama** -- Run open-source models locally

### RAG
- **ChromaDB** -- Easy embedded vector store
- **Qdrant** -- Production vector store
- **pgvector** -- Vector search in PostgreSQL
- **LangChain** -- RAG + agent framework
- **LlamaIndex** -- Data framework for LLM apps

### Production
- **FastAPI** -- Modern Python web framework
- **Streamlit** -- Fast data/ML app prototyping
- **Docker** -- Containerization
- **GitHub Actions** -- CI/CD

### Cloud
- **AWS Bedrock** -- Managed foundation models
- **AWS SageMaker** -- Custom model deployment
- **AWS Lambda** -- Serverless functions (for Bedrock Agents)

## Certification Prep

### AWS Certified Generative AI Developer Professional
- Stephane Maarek's Udemy course
- AWS Skill Builder practice exams
- AWS documentation on Bedrock, SageMaker
- 85 questions, 205 minutes, $300, passing: 750/1000

### Salesforce Agentforce Specialist
- Salesforce Trailhead modules
- Developer Edition org for practice
- Focus on: Agent Builder, Prompt Builder, topics and actions
- $200

## Communities

- **Hugging Face Discord** -- Active ML/NLP community
- **MLOps Community** -- Production ML discussion
- **r/MachineLearning** -- Research discussion
- **r/LocalLLaMA** -- Open-source LLM community
- **Latent Space Podcast** -- AI engineering interviews
