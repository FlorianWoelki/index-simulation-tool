import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algorithms = ['annoy']
data = {}
for algo in algorithms:
    data[algo] = pd.read_csv(f"../index/2024-09-25_10-24-41/{algo}_serial_500.csv")

fig, ax = plt.subplots(figsize=(12, 7))
current_index = 0

def plot_execution_time(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['execution_time'], marker='o', label=algo.upper())
    ax.set_title('Execution Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Execution Time (s)')
    ax.legend()

def plot_recall(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['recall'], marker='o', label=algo.upper())
    ax.set_title('Recall vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Recall')
    ax.legend()

def plot_queries_per_second(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['queries_per_second'], marker='o', label=algo.upper())
    ax.set_title('Queries per Second vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Queries per Second')
    ax.legend()

def plot_index_disk_space(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['index_disk_space'], marker='o', label=algo.upper())
    ax.set_title('Index Disk Space vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Index Disk Space (MB)')
    ax.legend()

def plot_performance_metrics(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['add_vector_performance'], marker='o', linestyle='--', label=f'{algo.upper()} Add Vector')
        ax.plot(df['dataset_size'], df['remove_vector_performance'], marker='s', linestyle='--', label=f'{algo.upper()} Remove Vector')
        ax.plot(df['dataset_size'], df['build_time'], marker='^', linestyle='--', label=f'{algo.upper()} Build Time')
        ax.plot(df['dataset_size'], df['search_time'], marker='D', linestyle='--', label=f'{algo.upper()} Search Time')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (s)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_index_operations(ax):
    for algo, df in data.items():
        ax.plot(df['dataset_size'], df['index_saving_time'], marker='o', linestyle='--', label=f'{algo.upper()} Saving Time')
        ax.plot(df['dataset_size'], df['index_loading_time'], marker='s', linestyle='--', label=f'{algo.upper()} Loading Time')
    ax.set_title('Index Operations Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (s)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_scalability_factor(ax):
    for algo, df in data.items():
        valid_data = df[np.isfinite(df['scalability_factor'])]
        ax.plot(valid_data['dataset_size'], valid_data['scalability_factor'], marker='o', label=algo.upper())
    ax.set_title('Scalability Factor vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Scalability Factor')
    ax.legend()

plot_functions = [
    plot_execution_time,
    plot_recall,
    plot_queries_per_second,
    plot_index_disk_space,
    plot_performance_metrics,
    plot_index_operations,
    plot_scalability_factor
]

def update_plot(index):
    ax.clear()
    plot_functions[index](ax)
    ax.grid(True)
    fig.canvas.draw()

def on_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(plot_functions)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(plot_functions)
    update_plot(current_index)

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot(current_index)
plt.tight_layout()
plt.show()
