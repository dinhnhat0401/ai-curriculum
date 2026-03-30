"""
AWS Bedrock - Foundation Model Calls

Demonstrates calling Claude and other models on AWS Bedrock using boto3.

Prerequisites:
    1. AWS account with Bedrock access enabled
    2. AWS CLI configured: aws configure
    3. Model access granted in Bedrock console for your region
    4. pip install boto3

Usage:
    python bedrock_chat.py
"""

import json
import time

try:
    import boto3
except ImportError:
    print("pip install boto3")
    exit(1)


def call_bedrock_claude(prompt: str, region: str = "us-east-1") -> dict:
    """Call Claude on AWS Bedrock using the Converse API.

    The Converse API provides a unified interface across all Bedrock models.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    start = time.time()

    response = client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude Sonnet on Bedrock
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={
            "maxTokens": 512,
            "temperature": 0.7,
        },
    )

    latency = (time.time() - start) * 1000

    # Extract response
    output_text = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]

    return {
        "text": output_text,
        "latency_ms": latency,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
    }


def call_bedrock_streaming(prompt: str, region: str = "us-east-1"):
    """Call Claude on Bedrock with streaming response.

    Streaming is critical for UX -- users see tokens as they're generated.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    response = client.converse_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={"maxTokens": 512},
    )

    # Process the stream
    full_text = ""
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            chunk = event["contentBlockDelta"]["delta"]["text"]
            print(chunk, end="", flush=True)
            full_text += chunk

    print()  # newline after streaming
    return full_text


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AWS BEDROCK - CLAUDE")
    print("=" * 60)

    print("\nNote: This requires AWS credentials and Bedrock model access.")
    print("If you get an error, check:")
    print("  1. aws configure (credentials set?)")
    print("  2. Bedrock model access (enabled in console?)")
    print("  3. Region (us-east-1 has the most models)")

    try:
        # Non-streaming call
        print("\n--- Non-streaming ---")
        result = call_bedrock_claude("Explain RAG in one sentence.")
        print(f"Response: {result['text']}")
        print(f"Latency:  {result['latency_ms']:.0f}ms")
        print(f"Tokens:   {result['input_tokens']} in / {result['output_tokens']} out")

        # Streaming call
        print("\n--- Streaming ---")
        call_bedrock_streaming("Write a haiku about cloud computing.")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nThis is expected if AWS credentials or Bedrock access aren't configured.")
        print("Follow the setup steps in the README to get started.")
