#!/usr/bin/env python3
"""
Check status of OpenAI Batch API job and download results when complete.
"""

import json
import os
import time
from openai import OpenAI

# Configuration
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def check_batch_status(batch_id):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print(f"Request counts:")
    print(f"  Total: {batch.request_counts.total}")
    print(f"  Completed: {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")
    
    if batch.status == "completed":
        print(f"\n✓ Batch completed!")
        return batch
    elif batch.status == "failed":
        print(f"\n✗ Batch failed!")
        if batch.errors:
            print(f"Errors: {batch.errors}")
        return None
    else:
        print(f"\n⏳ Batch still processing...")
        return None

def download_batch_results(batch):
    """Download batch results to local file."""
    output_file_id = batch.output_file_id
    
    if not output_file_id:
        print("No output file available yet.")
        return None
    
    # Download the file content
    file_response = client.files.content(output_file_id)
    
    # Save to local file
    output_filename = "batch_output.jsonl"
    with open(output_filename, 'wb') as f:
        f.write(file_response.content)
    
    print(f"\n✓ Downloaded batch results to {output_filename}")
    
    # Parse and save as structured JSON
    parse_batch_output(output_filename)
    
    return output_filename

def parse_batch_output(jsonl_file):
    """Parse JSONL output and organize by bucket."""
    all_prompts = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            bucket_name = custom_id.split('_batch_')[0]
            
            # Extract the response
            response_body = result['response']['body']
            content = response_body['choices'][0]['message']['content']
            
            # Parse JSON response
            try:
                data = json.loads(content)
                examples = data.get('examples', data.get('items', []))
                
                # Add bucket to each example
                for example in examples:
                    example['bucket'] = bucket_name
                    all_prompts.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse response for {custom_id}: {e}")
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(all_prompts)
    df.to_csv("prompts_raw.csv", index=False)
    
    print(f"\n✓ Parsed {len(all_prompts)} prompts")
    print(f"  Saved to prompts_raw.csv")
    
    # Show distribution
    print("\nDistribution by bucket:")
    for bucket, count in df['bucket'].value_counts().items():
        print(f"  {bucket}: {count}")
    
    return all_prompts

def monitor_batch(batch_id, check_interval=60):
    """Monitor batch job until completion."""
    print(f"Monitoring batch {batch_id}...")
    print(f"Checking every {check_interval} seconds.\n")
    
    while True:
        batch = check_batch_status(batch_id)
        
        if batch and batch.status == "completed":
            download_batch_results(batch)
            print("\n✓ All done! Run deduplicate_prompts.py next.")
            break
        elif batch is None:
            print("\n✗ Batch failed or encountered an error.")
            break
        else:
            print(f"\nNext check in {check_interval} seconds...")
            time.sleep(check_interval)

if __name__ == "__main__":
    print("=== Check Batch Status ===\n")
    
    # Read batch ID from file
    if not os.path.exists("batch_id.txt"):
        print("Error: batch_id.txt not found. Run generate_prompts_batch.py first.")
        exit(1)
    
    with open("batch_id.txt", 'r') as f:
        batch_id = f.read().strip()
    
    # Check if user wants to monitor continuously
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_batch(batch_id)
    else:
        batch = check_batch_status(batch_id)
        if batch and batch.status == "completed":
            download_batch_results(batch)
            print("\n✓ All done! Run deduplicate_prompts.py next.")
        else:
            print("\nTip: Run with --monitor flag to auto-check until complete:")
            print(f"  python check_batch_status.py --monitor")
