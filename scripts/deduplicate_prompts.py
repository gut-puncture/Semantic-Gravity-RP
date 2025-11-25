#!/usr/bin/env python3
"""
Deduplicate prompts using semantic embeddings and cosine similarity.
Reduces 600 raw prompts to 500 highly diverse prompts (100 per bucket).
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold
TARGET_PER_BUCKET = 100

def load_raw_prompts():
    """Load raw prompts from CSV."""
    df = pd.read_csv("prompts_raw.csv")
    print(f"Loaded {len(df)} raw prompts")
    return df

def compute_embeddings(prompts):
    """Compute semantic embeddings for prompts."""
    print("Computing embeddings using SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompts, show_progress_bar=True)
    return embeddings

def deduplicate_bucket(df_bucket, embeddings):
    """
    Deduplicate prompts within a bucket using cosine similarity.
    Keep diverse prompts by iteratively selecting the one farthest from already selected.
    """
    n = len(df_bucket)
    indices = list(range(n))
    selected_indices = []
    
    # Start with a random prompt
    first_idx = np.random.choice(indices)
    selected_indices.append(first_idx)
    remaining_indices = [i for i in indices if i != first_idx]
    
    # Greedily select prompts that are most dissimilar to already selected
    while len(selected_indices) < TARGET_PER_BUCKET and remaining_indices:
        # Compute minimum similarity to any selected prompt
        min_similarities = []
        for idx in remaining_indices:
            similarities = cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings[selected_indices]
            )[0]
            min_sim = similarities.max()  # Max similarity to closest selected prompt
            min_similarities.append((idx, min_sim))
        
        # Select the one with lowest max similarity (i.e., most different)
        min_similarities.sort(key=lambda x: x[1])
        next_idx = min_similarities[0][0]
        
        # Check if it's above threshold with any selected prompt
        if min_similarities[0][1] < SIMILARITY_THRESHOLD:
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        else:
            # All remaining are too similar, stop
            break
    
    return selected_indices

def deduplicate_prompts(df, embeddings):
    """Deduplicate prompts for each bucket."""
    deduplicated_dfs = []
    
    for bucket_name in df['bucket'].unique():
        print(f"\nProcessing bucket: {bucket_name}")
        df_bucket = df[df['bucket'] == bucket_name].reset_index(drop=True)
        bucket_embeddings = embeddings[df['bucket'] == bucket_name]
        
        print(f"  Raw prompts: {len(df_bucket)}")
        
        # Deduplicate
        selected_indices = deduplicate_bucket(df_bucket, bucket_embeddings)
        df_dedup = df_bucket.iloc[selected_indices].reset_index(drop=True)
        
        print(f"  After deduplication: {len(df_dedup)}")
        
        # Compute diversity metric (average pairwise distance)
        selected_embeddings = bucket_embeddings[selected_indices]
        pairwise_sim = cosine_similarity(selected_embeddings)
        avg_similarity = (pairwise_sim.sum() - len(df_dedup)) / (len(df_dedup) * (len(df_dedup) - 1))
        avg_distance = 1 - avg_similarity
        print(f"  Average pairwise distance: {avg_distance:.3f}")
        
        deduplicated_dfs.append(df_dedup)
    
    return pd.concat(deduplicated_dfs, ignore_index=True)

if __name__ == "__main__":
    print("=== Semantic Deduplication ===\n")
    
    # Load raw prompts
    df = load_raw_prompts()
    
    # Compute embeddings
    prompts_text = df['prompt'].tolist()
    embeddings = compute_embeddings(prompts_text)
    
    # Deduplicate
    df_final = deduplicate_prompts(df, embeddings)
    
    # Save final prompts
    df_final.to_csv("prompts.csv", index=False)
    
    print(f"\n✓ Deduplication complete!")
    print(f"  Final dataset: {len(df_final)} prompts")
    print(f"  Saved to prompts.csv")
    
    print("\nFinal distribution:")
    for bucket, count in df_final['bucket'].value_counts().items():
        print(f"  {bucket}: {count}")
    
    print("\n✓ Ready to use in main.ipynb!")
