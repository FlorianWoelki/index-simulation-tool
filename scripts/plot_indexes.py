import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algo = 'linscan'
df = pd.read_csv(f"../index/Cosine/2024-10-07_16-13-15/{algo}.csv")
df_build = pd.read_csv(f"../index/Cosine/2024-10-07_16-13-15/{algo}_build.csv")

fig, ax = plt.subplots(figsize=(7, 7))
current_index = 0

def plot_execution_time(ax):
    ax.plot(df['dataset_size'], df['execution_time'], marker='o', label=algo.upper())
    ax.set_title('Execution Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Execution Time (in s)')
    ax.legend()

def plot_recall(ax):
    ax.plot(df['dataset_size'], df['recall'], marker='o', label=algo.upper())
    ax.set_title('Recall vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Recall')
    ax.legend()

def plot_queries_per_second(ax):
    ax.plot(df['dataset_size'], df['queries_per_second'], marker='o', label=algo.upper())
    ax.set_title('Queries per Second vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Queries per Second')
    ax.legend()

def plot_index_disk_space(ax):
    ax.plot(df['dataset_size'], df['index_disk_space'], marker='o', label=algo.upper())
    ax.set_title('Index Disk Space vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Index Disk Space (in mb)')
    ax.legend()

def plot_performance_metrics(ax):
    ax.plot(df['dataset_size'], df['add_vector_performance'], marker='o', linestyle='--', label=f'{algo.upper()} Add Vector')
    ax.plot(df['dataset_size'], df['remove_vector_performance'], marker='s', linestyle='--', label=f'{algo.upper()} Remove Vector')
    ax.plot(df['dataset_size'], df['build_time'], marker='^', linestyle='--', label=f'{algo.upper()} Build Time')
    ax.plot(df['dataset_size'], df['search_time'], marker='D', linestyle='--', label=f'{algo.upper()} Search Time')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (in s)')
    ax.legend()

def plot_index_operations(ax):
    ax.plot(df['dataset_size'], df['index_saving_time'], marker='o', linestyle='--', label=f'{algo.upper()} Saving Time')
    ax.plot(df['dataset_size'], df['index_loading_time'], marker='s', linestyle='--', label=f'{algo.upper()} Loading Time')
    ax.set_title('Index Operations Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (in s)')
    ax.legend()

def plot_scalability_factor(ax):
    valid_data = df[np.isfinite(df['scalability_factor'])]
    ax.plot(valid_data['dataset_size'], valid_data['scalability_factor'], marker='o', label=algo.upper())
    ax.set_title('Scalability Factor vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Scalability Factor')
    ax.legend()

def plot_consumed_memory(ax):
    ax.plot(df_build['dataset_size'], df_build['consumed_memory'], marker='o', label=algo.upper())
    ax.set_title('Consumed Memory vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Consumed Memory (in mb)')
    ax.legend()

def plot_consumed_cpu(ax):
    ax.plot(df_build['dataset_size'], df_build['consumed_cpu'], marker='o', label=algo.upper())
    ax.set_title('Consumed CPU vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Consumed CPU (in %)')
    ax.legend()

plot_functions = [
    plot_execution_time,
    plot_recall,
    plot_queries_per_second,
    plot_index_disk_space,
    plot_performance_metrics,
    plot_index_operations,
    plot_scalability_factor,
    plot_consumed_memory,
    plot_consumed_cpu,
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
plt.tight_layout(pad=4.0)
plt.show()
