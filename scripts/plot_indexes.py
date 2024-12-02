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

def print_metric_differences(dataset_sizes, metric_values, metric_name):
    print(f"\n{metric_name} differences per dataset size:")
    for size in dataset_sizes:
        values_at_size = {}
        for metric in distance_metrics:
            idx = dfs[metric]['dataset_size'] == size
            if idx.any():
                values_at_size[metric] = metric_values[metric][idx].iloc[0]

        if values_at_size:
            max_val = max(values_at_size.values())
            min_val = min(values_at_size.values())
            diff = max_val - min_val
            max_metric = max(values_at_size.items(), key=lambda x: x[1])[0]
            min_metric = min(values_at_size.items(), key=lambda x: x[1])[0]
            print(f"Dataset size {size}: diff = {diff:.4f}, max metric = {max_metric} ({max_val:.2f}), min metric = {min_metric} ({min_val:.2f})")

def plot_recall_and_qps(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    for metric in distance_metrics:
        axs[0].plot(dfs[metric]['dataset_size'], dfs[metric]['recall'], marker='o', label=f'{metric}')
        axs[1].plot(dfs[metric]['dataset_size'], dfs[metric]['queries_per_second'], marker='o', label=f'{metric}')

    recall_values = {metric: dfs[metric]['recall'] for metric in distance_metrics}
    qps_values = {metric: dfs[metric]['queries_per_second'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, recall_values, "Recall")
    print_metric_differences(dataset_sizes, qps_values, "Queries per Second")

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

    add_values = {metric: dfs[metric]['add_vector_performance'] for metric in distance_metrics}
    remove_values = {metric: dfs[metric]['remove_vector_performance'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, add_values, "Add Vector Performance")
    print_metric_differences(dataset_sizes, remove_values, "Remove Vector Performance")

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

    build_values = {metric: dfs[metric]['build_time'] for metric in distance_metrics}
    search_values = {metric: dfs[metric]['search_time'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, build_values, "Build Time")
    print_metric_differences(dataset_sizes, search_values, "Search Time")

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

    saving_values = {metric: dfs[metric]['index_saving_time'] for metric in distance_metrics}
    loading_values = {metric: dfs[metric]['index_loading_time'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, saving_values, "Index Saving Time")
    print_metric_differences(dataset_sizes, loading_values, "Index Loading Time")

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

    cpu_values = {metric: dfs_build[metric]['consumed_cpu'] for metric in distance_metrics}
    memory_values = {metric: dfs_build[metric]['consumed_memory'] for metric in distance_metrics}
    disk_values = {metric: dfs[metric]['index_disk_space'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, cpu_values, "Consumed CPU")
    print_metric_differences(dataset_sizes, memory_values, "Consumed Memory")
    print_metric_differences(dataset_sizes, disk_values, "Index Disk Space")

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

    scalability_values = {metric: dfs[metric]['scalability_factor'] for metric in distance_metrics}
    execution_values = {metric: dfs[metric]['execution_time'] for metric in distance_metrics}
    dataset_sizes = dfs[distance_metrics[0]]['dataset_size'].unique()
    print_metric_differences(dataset_sizes, scalability_values, "Scalability Factor")
    print_metric_differences(dataset_sizes, execution_values, "Execution Time")

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
