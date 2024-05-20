from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

dataset_name = 'SandipPalit/Movie_Dataset'

def clean_text(text):
    return re.sub(r'[\r\n\t]+', ' ', text)

vectorizer = TfidfVectorizer()

dataset = load_dataset(dataset_name)
training_dataset = dataset['train']

texts = []
vectors = []

for i, example in enumerate(tqdm(training_dataset, desc="Processing")):
    text = clean_text(example['Overview'])
    texts.append(text)
    if i == 11: # temporary
        break

tfidf_matrix = vectorizer.fit_transform(texts).toarray()

df = pd.DataFrame({
    'text': texts,
    'tfidf_vector': list(tfidf_matrix)
})
query = "Space exploration"
df.at[0, 'text'] = query
df.at[0, 'tfidf_vector'] = vectorizer.transform([query]).toarray()[0]

print(df['tfidf_vector'].iloc[1])
d = {}
for num in np.array(df['tfidf_vector'].iloc[4]):
    if num in d:
        d[num] += 1
    else:
        d[num] = 1

print(d)
