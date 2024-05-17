from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import re
import torch
import numpy as np

model_id = 'naver/splade-cocondenser-ensembledistil'
dataset_name = 'SandipPalit/Movie_Dataset'

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

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

dataset = load_dataset(dataset_name)
training_dataset = dataset['train']

texts = []
vectors = []

i = 0 # temporary
for example in tqdm(training_dataset, desc="Processing"):
    text = clean_text(example['Overview'])
    texts.append(text)
    vector = create_splade(text)
    vectors.append(vector.tolist())
    i += 1
    if i == 11: # temporary
        break

df = pd.DataFrame({
    'text': texts,
    'splade_vector': vectors
})
query = "Space exploration"
df.at[0, 'text'] = query
df.at[0, 'splade_vector'] = create_splade(query).tolist()

d = {}
for num in np.array(df['splade_vector'].iloc[4]):
    if num in d:
        d[num] += 1
    else:
        d[num] = 1

print(d)
