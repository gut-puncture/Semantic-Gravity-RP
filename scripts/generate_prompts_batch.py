#!/usr/bin/env python3
"""
Generate diverse prompts using OpenAI Batch API.
This script creates a batch job to generate 120 prompts per category (600 total).
"""

import json
import os
from openai import OpenAI

# Configuration
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Bucket definitions
BUCKETS = {
    "A_Idioms": "Create sentences that are fixed idioms (e.g., 'A blessing in...'). Target high probability. The completion must be the standard idiomatic word.",
    "B_Facts": "Create historical or scientific facts (e.g., 'The capital of France is...'). Target medium-high probability. The completion must be the factual answer.",
    "C_CommonSense": "Create questions with 2-3 likely answers (e.g., 'Name a common pet...'). Target medium probability.",
    "D_Creative": "Create open-ended story prompts with a cliffhanger ending (e.g., 'The spaceship landed in...'). Target low probability.",
    "E_OOD": "Create highly counter-intuitive, surreal, or logic-bending scenarios where the context forces a specific word that contradicts general world knowledge. Example: 'In this alternate universe, the color of a ripe banana is purple. The monkey peeled the banana and ate the delicious... -> Purple'. The target word MUST be the one forced by the weird context, not the real-world truth."
}

PROMPTS_PER_BUCKET = 120
BATCH_SIZE = 5

def create_batch_requests():
    """Create batch request file for OpenAI Batch API."""
    requests = []
    request_id = 0
    
    for bucket_name, instruction in BUCKETS.items():
        num_batches = PROMPTS_PER_BUCKET // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            # Create a request for this batch
            request = {
                "custom_id": f"{bucket_name}_batch_{batch_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are a dataset generator. Output a JSON object with a key 'examples' containing a list of exactly {BATCH_SIZE} distinct objects. Each object should have 'prompt' (the input text ending in a way that predicts a specific word), 'target_word' (the expected completion), and 'forbidden_word' (same as target_word). Ensure examples are maximally diverse - vary sentence structures, topics, semantic domains (politics, science, everyday life, hypotheticals, etc.). Do not repeat patterns or topics from previous examples."
                        },
                        {
                            "role": "user",
                            "content": f"Instruction: {instruction}. Generate {BATCH_SIZE} examples that are semantically diverse and cover different topics."
                        }
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.9  # Higher temperature for diversity
                }
            }
            requests.append(request)
            request_id += 1
    
    # Write to JSONL file
    batch_input_file = "batch_input.jsonl"
    with open(batch_input_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created {len(requests)} batch requests in {batch_input_file}")
    return batch_input_file

def upload_and_create_batch(batch_input_file):
    """Upload batch file and create batch job."""
    # Upload file
    with open(batch_input_file, 'rb') as f:
        batch_file = client.files.create(
            file=f,
            purpose='batch'
        )
    
    print(f"Uploaded batch file: {batch_file.id}")
    
    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Semantic Gravity prompt generation - 600 diverse prompts"
        }
    )
    
    print(f"Created batch job: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    
    # Save batch ID for later retrieval
    with open("batch_id.txt", 'w') as f:
        f.write(batch_job.id)
    
    print(f"\nBatch ID saved to batch_id.txt")
    print(f"Run check_batch_status.py to monitor progress.")
    
    return batch_job.id

if __name__ == "__main__":
    print("=== Semantic Gravity Prompt Generation (Batch API) ===\n")
    
    # Create batch requests
    batch_file = create_batch_requests()
    
    # Upload and create batch
    batch_id = upload_and_create_batch(batch_file)
    
    print(f"\n✓ Batch job created successfully!")
    print(f"  Batch ID: {batch_id}")
    print(f"  Expected prompts: {len(BUCKETS) * PROMPTS_PER_BUCKET}")
