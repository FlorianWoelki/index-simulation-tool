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

# Generate sparse vectors.
sparse_vectors = []
for text in tqdm(texts, desc="Generating sparse vectors"):
    sparse_vectors.append(sparse_vector_to_dict(create_splade(text)))

num_queries = 2
queries = []
query_vectors = []
groundtruth = []
groundtruth_vectors = []
for _ in range(num_queries):
    random_doc = np.random.choice(texts)

    query = generate_related_query(random_doc)
    queries.append(query)
    query_vectors.append(sparse_vector_to_dict(create_splade(query)))

    groundtruth.append(random_doc)
    groundtruth_vectors.append(sparse_vector_to_dict(create_splade(random_doc)))

with open('data.msgpack', 'wb') as f:
    msgpack.dump(sparse_vectors, f)

with open('queries.msgpack', 'wb') as f:
    msgpack.dump(query_vectors, f)

with open('groundtruth.msgpack', 'wb') as f:
    msgpack.dump(groundtruth_vectors, f)

print(f"Generated {len(texts)} document vectors, {num_queries} query vectors.")
print(f"Sample query: {queries[0]}")
print(f"Sample groundtruth: {groundtruth[0]}")
print("Datasets saved to 'data.msgpack', 'queries.msgpack', and 'groundtruth.msgpack'")
