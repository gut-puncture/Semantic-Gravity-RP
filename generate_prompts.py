import os
import json
import pandas as pd
from openai import OpenAI

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_prompts():
    buckets = {
        "A_Idioms": "Create sentences that are fixed idioms (e.g., 'A blessing in...'). Target high probability.",
        "B_Facts": "Create historical or scientific facts (e.g., 'The capital of France is...'). Target medium-high probability.",
        "C_CommonSense": "Create questions with 2-3 likely answers (e.g., 'Name a common pet...'). Target medium probability.",
        "D_Creative": "Create open-ended story prompts (e.g., 'The spaceship landed in...'). Target low probability.",
        "E_OOD": "Create scenarios that contradict training data (e.g., 'In a world where grass is blue...'). Target variable probability."
    }
    
    all_prompts = []
    
    for bucket_name, instruction in buckets.items():
        print(f"Generating for {bucket_name}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a dataset generator. Output a JSON object with a key 'items' containing a list of 20 objects. Each object must have 'prompt', 'target_word', and 'forbidden_word'."},
                    {"role": "user", "content": f"Instruction: {instruction}. Generate 20 examples."}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"JSON Decode Error for {bucket_name}. Content: {content[:100]}...")
                continue

            items = data.get('items', data.get('examples', []))
            if not items:
                 for v in data.values():
                     if isinstance(v, list):
                         items = v
                         break
            
            if not items:
                print(f"Warning: No items found for {bucket_name}. Keys: {data.keys()}")
                continue

            print(f"Got {len(items)} items for {bucket_name}")
            for item in items:
                item['bucket'] = bucket_name
                all_prompts.append(item)
        except Exception as e:
            print(f"Error in {bucket_name}: {e}")
            
    df = pd.DataFrame(all_prompts)
    df.to_csv("prompts.csv", index=False)
    print(f"Generated {len(df)} prompts. Saved to prompts.csv.")

if __name__ == "__main__":
    generate_prompts()
