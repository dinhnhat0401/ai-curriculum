# AWS Bedrock: Enterprise AI on Managed Services

**Time: ~8 hours | Files: `bedrock_chat.py`, `bedrock_rag.py`**

---

## Why AWS Bedrock?

Enterprise clients don't want to manage ML infrastructure. They want:
- **Data residency**: data stays in their AWS account
- **IAM integration**: existing auth and access control
- **Compliance**: SOC 2, HIPAA, GDPR compatible
- **No servers to manage**: fully managed service

Bedrock gives them all of this with Claude, Titan, Llama, and Mistral models.

## What Bedrock Offers

### Foundation Models
Call Claude, Llama, Mistral, and Amazon Titan through a single API (boto3). Same models, AWS-hosted, data stays in your account.

### Knowledge Bases (Managed RAG)
Upload documents to S3 -> Bedrock automatically chunks, embeds, stores in OpenSearch Serverless, and handles retrieval. Your entire RAG pipeline, managed.

```
    Your From-Scratch RAG          Bedrock Knowledge Bases
    ─────────────────────          ──────────────────────────
    chunker.py               →    Automatic (configurable)
    embeddings.py            →    Titan Embeddings (automatic)
    vector_store.py          →    OpenSearch Serverless (managed)
    rag.py (retrieval)       →    RetrieveAndGenerate API
    rag.py (generation)      →    Claude on Bedrock (automatic)

    You manage: everything        You manage: documents + config
```

### Agents
Define tools as Lambda functions, connect to Knowledge Bases, let the LLM reason and call tools. Same agent pattern you built in Phase 2, but managed.

## Cost Considerations

| Service | Pricing Model |
|---------|--------------|
| Foundation Models | Per input/output token (same as direct API, sometimes slight markup) |
| Knowledge Bases | OpenSearch Serverless (~$0.24/hr per OCU) + embedding costs |
| Provisioned Throughput | Reserved capacity for guaranteed performance |

For a typical Knowledge Base: ~$200-500/month depending on data volume and query rate.

## Exercises

1. Call Claude on Bedrock using boto3. Compare latency with the direct Anthropic API.
2. Create a Knowledge Base with your sample documents from Phase 2.
3. Query the Knowledge Base and compare retrieval quality with your from-scratch RAG.
4. Set up CloudWatch alarms for Bedrock API errors and latency.
5. Calculate the cost of running your RAG system on Bedrock at 1000 queries/day.
6. Deploy a Bedrock Agent with a Lambda function tool.
