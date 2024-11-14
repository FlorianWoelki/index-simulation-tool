import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algo = 'lsh-simhash'
distance_metrics = ['Angular', 'Cosine', 'Dot', 'Euclidean', 'Jaccard']

dfs = {}
dfs_build = {}

for metric in distance_metrics:
    dfs[metric] = pd.read_csv(f"../index/artificial/{metric}/{algo}/{algo}.csv")
    dfs_build[metric] = pd.read_csv(f"../index/artificial/{metric}/{algo}/{algo}_build.csv")

fig, ax = plt.subplots(figsize=(12, 7))
current_index = 0

def plot_recall_and_qps(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        axs[0].plot(dfs[metric]['dataset_size'], dfs[metric]['recall'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['queries_per_second'], marker='o', label=f'{metric}')

    axs[0].set_title('Recall vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Recall')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Queries per Second vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Queries per Second')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

def plot_add_and_remove_vector(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        axs[0].plot(dfs[metric]['dataset_size'], dfs[metric]['add_vector_performance'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['remove_vector_performance'], marker='o', label=f'{metric}')

    axs[0].set_title('Add Vector Performance vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Time (s)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Remove Vector Performance vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

def plot_build_and_search_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        axs[0].plot(dfs[metric]['dataset_size'], dfs[metric]['build_time'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['search_time'], marker='o', label=f'{metric}')

    axs[0].set_title('Build Time vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Time (s)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Search Time vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

def plot_saving_and_loading_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        axs[0].plot(dfs[metric]['dataset_size'], dfs[metric]['index_saving_time'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['index_loading_time'], marker='o', label=f'{metric}')

    axs[0].set_title('Index Saving Time vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Time (s)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Index Loading Time vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

def plot_cpu_memory_disk(fig):
    fig.clear()
    axs = fig.subplots(1, 3)

    for metric in distance_metrics:
        axs[0].plot(dfs_build[metric]['dataset_size'], dfs_build[metric]['consumed_cpu'], marker='o', label=f'{metric}')
        axs[1].plot(dfs_build[metric]['dataset_size'], dfs_build[metric]['consumed_memory'], marker='o', label=f'{metric}')
        axs[2].plot(dfs[metric]['dataset_size'], dfs[metric]['index_disk_space'], marker='o', label=f'{metric}')

    axs[0].set_title('Consumed CPU vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Consumed CPU (%)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Consumed Memory vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Consumed Memory (MB)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title('Index Disk Space vs Dataset Size')
    axs[2].set_xlabel('Dataset Size')
    axs[2].set_ylabel('Index Disk Space (MB)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

def plot_scalability_and_execution_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        valid_data = dfs[metric][np.isfinite(dfs[metric]['scalability_factor'])]
        axs[0].plot(valid_data['dataset_size'], valid_data['scalability_factor'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['execution_time'], marker='o', label=f'{metric}')

    axs[0].set_title('Scalability Factor vs Dataset Size')
    axs[0].set_xlabel('Dataset Size')
    axs[0].set_ylabel('Scalability Factor')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Execution Time vs Dataset Size')
    axs[1].set_xlabel('Dataset Size')
    axs[1].set_ylabel('Execution Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

plot_functions = [
    plot_recall_and_qps,
    plot_add_and_remove_vector,
    plot_build_and_search_time,
    plot_saving_and_loading_time,
    plot_cpu_memory_disk,
    plot_scalability_and_execution_time,
]

def update_plot(index):
    plot_functions[index](fig)
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
