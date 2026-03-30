# Glossary

Every AI/ML term you'll encounter in this curriculum, defined in plain language.

---

## A

**Activation Function** -- A nonlinear function applied after a linear layer (e.g., ReLU, tanh, sigmoid). Without it, stacking linear layers is equivalent to a single linear layer. *First encountered: MNIST*

**Adam** -- An optimizer that combines momentum and adaptive learning rates. The default choice for most deep learning training. *First encountered: MNIST*

**Agent** -- An LLM-based system that can use tools, make decisions, and take actions autonomously. Goes beyond Q&A to actually *do* things. *First encountered: Agent module*

**Attention** -- A mechanism that lets each token in a sequence look at other tokens to gather relevant information. The core innovation of transformers. *First encountered: NanoGPT*

**Autograd** -- Automatic differentiation. A system that tracks operations to compute gradients automatically. PyTorch's autograd is a scaled-up version of micrograd. *First encountered: Micrograd*

## B

**Backpropagation** -- The algorithm for computing gradients by applying the chain rule backward through a computational graph. How neural networks learn. *First encountered: Micrograd*

**Batch** -- A group of training examples processed together. Larger batches = more stable gradients but slower convergence per epoch. *First encountered: MNIST*

**Batch Normalization** -- Normalizes activations within a layer across a batch to stabilize training. *First encountered: Makemore*

**BLEU** -- Bilingual Evaluation Understudy. A metric measuring n-gram overlap between generated and reference text. Common for translation. *First encountered: Evaluation*

## C

**Causal Mask** -- A triangular mask in transformers that prevents tokens from attending to future positions. Makes the model autoregressive (generates left to right). *First encountered: NanoGPT*

**Chain of Thought** -- A prompting technique that asks the model to show its reasoning step by step. Improves performance on complex tasks. *First encountered: Model Comparison*

**Chunking** -- Splitting documents into smaller pieces for RAG. Critical for retrieval quality. *First encountered: RAG*

**Computational Graph** -- A directed graph of operations that records how outputs are computed from inputs. Used by autograd to compute gradients. *First encountered: Micrograd*

**Context Window** -- The maximum number of tokens a model can process at once. GPT-4: 128K tokens. Claude: 200K tokens. *First encountered: NanoGPT*

**Cosine Similarity** -- A measure of similarity between two vectors, computed as their dot product divided by the product of their magnitudes. Range: -1 to 1. *First encountered: RAG*

**Cross-Entropy Loss** -- The standard loss function for classification. Measures the difference between predicted probabilities and true labels. *First encountered: MNIST*

## D

**Decoder** -- The part of a transformer that generates output tokens. GPT is decoder-only. *First encountered: NanoGPT*

**Dropout** -- Regularization technique that randomly zeros out neurons during training. Prevents overfitting. *First encountered: NanoGPT*

## E

**Embedding** -- A learned dense vector representation of a discrete token (word, character, document). Similar items have similar embeddings. THE concept that connects Phase 1 to Phase 2. *First encountered: Makemore*

**Encoder** -- The part of a transformer that processes input tokens. BERT is encoder-only. *First encountered: NanoGPT*

**Epoch** -- One complete pass through the entire training dataset. *First encountered: MNIST*

**Evaluation** -- The process of measuring AI system quality with metrics and test data. Without evaluation, you're guessing. *First encountered: Evaluation*

## F

**Feed-Forward Network (FFN)** -- A simple neural network (linear -> activation -> linear) applied to each token independently after attention. In a transformer: attention = communication, FFN = computation. *First encountered: NanoGPT*

**Few-Shot** -- Providing a few examples in the prompt to guide the model's output format and behavior. *First encountered: Model Comparison*

**Fine-tuning** -- Training a pre-trained model on new data to adapt it for a specific task. *First encountered: Fine-tuning*

**Forward Pass** -- Running input data through the network to get predictions. *First encountered: MNIST*

**Function Calling** -- The ability of an LLM to generate structured tool calls (JSON) when it needs to use external tools. *First encountered: Agent*

## G

**Gradient** -- The derivative of the loss with respect to a parameter. Points in the direction of steepest increase; we go the opposite way to minimize loss. *First encountered: Micrograd*

**Gradient Descent** -- The optimization algorithm: update each weight by subtracting the learning rate times the gradient. *First encountered: Micrograd*

## H

**Hallucination** -- When an LLM generates plausible-sounding but factually incorrect information. RAG reduces hallucination by grounding responses in retrieved context. *First encountered: RAG*

**Hyperparameter** -- A configuration value set before training (learning rate, batch size, number of layers). Not learned from data. *First encountered: MNIST*

## I

**Inference** -- Using a trained model to make predictions on new data. The production use of a model. *First encountered: Production*

## K

**Key (K)** -- In attention, the vector representing "what information this token contains." Used with Query to compute attention scores. *First encountered: NanoGPT*

## L

**Layer Normalization** -- Normalizes activations per token (vs BatchNorm which normalizes per batch). Used in transformers. *First encountered: NanoGPT*

**Learning Rate** -- The step size for weight updates during gradient descent. Too high = diverge. Too low = slow. *First encountered: MNIST*

**LLM** -- Large Language Model. A neural network trained on massive text data that can generate, understand, and reason about text. GPT-4, Claude, Llama are LLMs.

**LLM-as-Judge** -- Using a strong LLM to evaluate the outputs of another AI system. Cheaper than human evaluation, often just as good. *First encountered: Evaluation*

**LoRA** -- Low-Rank Adaptation. A fine-tuning technique that freezes the base model and trains small adapter matrices. Drastically reduces compute needed. *First encountered: Fine-tuning*

**Loss Function** -- A function that measures how wrong the model's predictions are. Training minimizes this. *First encountered: MNIST*

## M

**MLP** -- Multi-Layer Perceptron. A neural network with one or more hidden layers. The simplest feedforward architecture. *First encountered: Micrograd*

**Multi-Head Attention** -- Running multiple attention heads in parallel, each learning different types of relationships, then concatenating results. *First encountered: NanoGPT*

## N

**Negative Log Likelihood (NLL)** -- A loss function for language models. The negative log of the probability the model assigns to the correct next token. *First encountered: Makemore*

## O

**Optimizer** -- The algorithm that updates model weights using gradients. Common: Adam, SGD, AdamW. *First encountered: MNIST*

**Overfitting** -- When a model memorizes training data instead of learning general patterns. Test performance degrades while training performance improves. *First encountered: MNIST*

## P

**Positional Encoding** -- Information added to token embeddings to tell the model where each token is in the sequence. Without it, transformers can't distinguish order. *First encountered: NanoGPT*

**Precision@K** -- Of the top K retrieved items, what fraction is relevant? A key RAG retrieval metric. *First encountered: Evaluation*

**Prompt Engineering** -- The practice of crafting effective prompts to get better outputs from LLMs. *First encountered: Model Comparison*

## Q

**QLoRA** -- Quantized LoRA. Combines LoRA with model quantization (4-bit or 8-bit) to fit fine-tuning on consumer GPUs. *First encountered: Fine-tuning*

**Query (Q)** -- In attention, the vector representing "what information this token is looking for." *First encountered: NanoGPT*

**Quantization** -- Reducing the numerical precision of model weights (32-bit -> 8-bit or 4-bit) to reduce memory and compute. *First encountered: Fine-tuning*

## R

**RAG** -- Retrieval-Augmented Generation. Retrieve relevant documents, inject them as context, then generate an answer. The most common enterprise AI pattern. *First encountered: RAG*

**Recall@K** -- Of all relevant items, what fraction was retrieved in the top K? *First encountered: Evaluation*

**ReAct** -- Reason + Act. An agent architecture where the LLM alternates between reasoning about what to do and taking actions. *First encountered: Agent*

**Residual Connection** -- A "skip connection" that adds the input to the output of a layer: `output = layer(x) + x`. Makes deep networks trainable. *First encountered: NanoGPT*

**RLHF** -- Reinforcement Learning from Human Feedback. The training technique that makes LLMs helpful and safe. Happens after pre-training.

**ROUGE** -- Recall-Oriented Understudy for Gisting Evaluation. A metric measuring overlap between generated and reference text. Common for summarization. *First encountered: Evaluation*

## S

**Self-Attention** -- Attention applied within a single sequence (each token attends to all others in the same sequence). *First encountered: NanoGPT*

**Softmax** -- A function that converts a vector of numbers into a probability distribution (positive values that sum to 1). *First encountered: Makemore*

**Streaming** -- Generating and sending tokens one at a time instead of waiting for the complete response. Critical for UX. *First encountered: Model Comparison*

**System Prompt** -- A message that sets the behavior and personality of an LLM. Not visible to the user but shapes all responses.

## T

**Temperature** -- A parameter controlling output randomness. 0 = deterministic (always pick the most likely token). 1 = sample from the full distribution. *First encountered: Model Comparison*

**Tensor** -- A multi-dimensional array. The fundamental data structure of deep learning. *First encountered: MNIST*

**Token** -- A piece of text (word, subword, or character) that serves as the basic unit of processing for an LLM. GPT tokenizers split text into subword pieces.

**Topological Sort** -- An ordering of nodes in a directed graph such that every node comes after its dependencies. Used in backpropagation to process nodes in the correct order. *First encountered: Micrograd*

**Transformer** -- The architecture behind all modern LLMs. Key components: attention, feed-forward networks, residual connections, layer normalization. *First encountered: NanoGPT*

## V

**Value (V)** -- In attention, the vector representing "what information this token provides." The weighted sum of Value vectors is the attention output. *First encountered: NanoGPT*

**Vector Store** -- A database optimized for storing and searching high-dimensional vectors using approximate nearest neighbor algorithms. *First encountered: RAG*

## Z

**Zero-Shot** -- Asking a model to perform a task with no examples, just instructions. *First encountered: Model Comparison*
