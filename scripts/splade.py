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
    if len(texts) == 11: # temporary
        break

def sparse_vector_to_dict(vector):
    indices = []
    values = []
    for idx, val in enumerate(vector):
        if val != 0:
            indices.append(idx)
            values.append(float(val))
    return {"indices": indices, "values": values}

def compute_similarity(vec1, vec2):
    # Convert sparse vectors to dense format for similarity computation
    dense1 = torch.zeros(model.config.vocab_size)
    dense2 = torch.zeros(model.config.vocab_size)

    for idx, val in zip(vec1["indices"], vec1["values"]):
        dense1[idx] = val
    for idx, val in zip(vec2["indices"], vec2["values"]):
        dense2[idx] = val

    return float(torch.cosine_similarity(dense1.unsqueeze(0), dense2.unsqueeze(0)))

# Generate sparse vectors.
sparse_vectors = []
for text in tqdm(texts, desc="Generating sparse vectors"):
    sparse_vectors.append(sparse_vector_to_dict(create_splade(text)))

num_queries = 2
queries = []
query_vectors = []
groundtruth = []
k = 10

print("\n=== Generated Queries and Groundtruth ===")
for i in range(num_queries):
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

    print(f"\nQuery {i+1}:")
    print(f"Query Text: {query}")
    print(f"Query Vector: {query_vec}")
    print(f"\nTop {k} Groundtruth Documents:")
    for rank, idx in enumerate(top_k_indices):
        print(f"\nRank {rank+1}:")
        print(f"Text: {texts[idx]}")
        print(f"Vector: {sparse_vectors[idx]}")
        print(f"Similarity Score: {similarities[idx][1]:.4f}")

with open('data.msgpack', 'wb') as f:
    msgpack.dump(sparse_vectors, f)

with open('queries.msgpack', 'wb') as f:
    msgpack.dump(query_vectors, f)

with open('groundtruth.msgpack', 'wb') as f:
    msgpack.dump(groundtruth, f)

print("\n=== Summary ===")
print(f"Generated {len(texts)} document vectors, {num_queries} query vectors.")
print("Datasets saved to 'data.msgpack', 'queries.msgpack', and 'groundtruth.msgpack'")
