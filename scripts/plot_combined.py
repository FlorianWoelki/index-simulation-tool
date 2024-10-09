import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algo = 'linscan'
distance_metrics = ['Cosine']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

for metric in distance_metrics:
    df = pd.read_csv(f"../index/{metric}/{algo}/{algo}.csv")

    recall = df['recall'].values
    qps = df['queries_per_second'].values
    dataset_size = df['dataset_size'].values

    ax1.plot(dataset_size, recall, label=f'{algo.upper()} ({metric})', marker='o')
    ax2.plot(dataset_size, qps, label=f'{algo.upper()} ({metric})', marker='s')

ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('Recall')
ax1.set_title('Recall vs Dataset Size')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_xlabel('Dataset Size')
ax2.set_ylabel('Queries per Second')
ax2.set_title('QPS vs Dataset Size')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=4.0)
plt.show()
