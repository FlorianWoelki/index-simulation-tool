from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import re
import torch
import numpy as np
import msgpack

model_id = 'naver/splade-cocondenser-ensembledistil'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

dataset_name = 'SandipPalit/Movie_Dataset'
dataset = load_dataset(dataset_name)
training_dataset = dataset['train']

def generate_related_query(text, num_words=5):
    words = text.split()
    return " ".join(np.random.choice(words, min(num_words, len(words)), replace=False))

def create_splade(text):
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    vec = torch.max(
        torch.log(
            1 + torch.relu(output.logits)
        ) * tokens.attention_mask.unsqueeze(-1),
    dim=1)[0].squeeze()
    return vec

def clean_text(text):
    return re.sub(r'[\r\n\t]+', ' ', text)

texts = []
for example in tqdm(training_dataset, desc="Processing"):
    text = clean_text(example['Overview'])
    texts.append(text)
    # if len(texts) == 25000: # Temporary due to resource constraints.
    #     break

def sparse_vector_to_dict(vector):
    indices = []
    values = []
    for idx, val in enumerate(vector):
        if val != 0:
            indices.append(idx)
            values.append(float(val))
    return {"indices": indices, "values": values}

def compute_similarity(vec1, vec2):
    set1 = set(vec1['indices'])
    set2 = set(vec2['indices'])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# Generate sparse vectors.
sparse_vectors = []
for text in tqdm(texts, desc="Generating sparse vectors"):
    sparse_vectors.append(sparse_vector_to_dict(create_splade(text)))

num_queries = round(len(texts) / 10) # Like in `generator_sparse.rs` file
queries = []
query_vectors = []
groundtruth = []
k = 10

print("\n=== Generated Queries and Groundtruth ===")
for i in tqdm(range(num_queries), desc="Generating queries"):
    random_doc = np.random.choice(texts)
    query = generate_related_query(random_doc)
    queries.append(query)
    query_vec = sparse_vector_to_dict(create_splade(query))
    query_vectors.append(query_vec)

    # Compute similarities with all vectors
    similarities = []
    for idx, vec in enumerate(sparse_vectors):
        sim = compute_similarity(query_vec, vec)
        similarities.append((idx, sim))

    # Sort by similarity and get top k indices
    top_k_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:k]]
    groundtruth.append(top_k_indices)

with open('data-50k.msgpack', 'wb') as f:
    msgpack.dump(sparse_vectors, f)

with open('queries-50k.msgpack', 'wb') as f:
    msgpack.dump(query_vectors, f)

with open('groundtruth-50k.msgpack', 'wb') as f:
    msgpack.dump(groundtruth, f)

print("\n=== Summary ===")
print(f"Generated {len(texts)} document vectors, {num_queries} query vectors.")
print("Datasets saved to 'data.msgpack', 'queries.msgpack', and 'groundtruth.msgpack'")
