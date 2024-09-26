import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("../index/2024-09-25_10-24-41/annoy_serial_500.csv")

fig, ax = plt.subplots(figsize=(10, 6))

current_index = 0

def plot_execution_time(ax):
    ax.plot(data['dataset_size'], data['execution_time'], marker='o')
    ax.set_title('Execution Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Execution Time (s)')

def plot_recall(ax):
    ax.plot(data['dataset_size'], data['recall'], marker='o')
    ax.set_title('Recall vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Recall')

def plot_queries_per_second(ax):
    ax.plot(data['dataset_size'], data['queries_per_second'], marker='o')
    ax.set_title('Queries per Second vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Queries per Second')

def plot_index_disk_space(ax):
    ax.plot(data['dataset_size'], data['index_disk_space'], marker='o')
    ax.set_title('Index Disk Space vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Index Disk Space (MB)')

def plot_performance_metrics(ax):
    ax.plot(data['dataset_size'], data['add_vector_performance'], marker='o', label='Add Vector')
    ax.plot(data['dataset_size'], data['remove_vector_performance'], marker='s', label='Remove Vector')
    ax.plot(data['dataset_size'], data['build_time'], marker='^', label='Build Time')
    ax.plot(data['dataset_size'], data['search_time'], marker='D', label='Search Time')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (s)')
    ax.legend()

def plot_index_operations(ax):
    ax.plot(data['dataset_size'], data['index_saving_time'], marker='o', label='Saving Time')
    ax.plot(data['dataset_size'], data['index_loading_time'], marker='s', label='Loading Time')
    ax.set_title('Index Operations Time vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Time (s)')
    ax.legend()

def plot_scalability_factor(ax):
    # Remove any NaN or infinite values
    valid_data = data[np.isfinite(data['scalability_factor'])]
    ax.plot(valid_data['dataset_size'], valid_data['scalability_factor'], marker='o')
    ax.set_title('Scalability Factor vs Dataset Size')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Scalability Factor')

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

update_plot(current_index + 1)

plt.figtext(0.5, 0.01, 'Use left/right arrow keys to navigate between plots',
            ha='center', fontsize=10)

plt.tight_layout()
plt.show()
