from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import re
import torch
import numpy as np

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

# Generate sparse vectors.
sparse_vectors = []
for text in tqdm(texts, desc="Generating sparse vectors"):
    sparse_vectors.append(create_splade(text))

num_queries = 2
queries = []
query_vectors = []
groundtruth = []
for _ in range(num_queries):
    random_doc = np.random.choice(texts)

    query = generate_related_query(random_doc)
    query_vector = create_splade(query)
    queries.append(query)
    query_vectors.append(query_vector)

    groundtruth.append(random_doc)

print(f"Generated {len(texts)} document vectors, {num_queries} query vectors.")
print(f"Sample query: {queries[0]}")
print(f"Sample groundtruth: {groundtruth[0]}")
print("Dataset saved to 'splade_sparse_vectors_dataset.csv'")
