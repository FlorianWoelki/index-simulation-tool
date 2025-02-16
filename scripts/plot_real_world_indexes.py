import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

distance_metric = "Jaccard"
algos = ["linscan", "annoy", "hnsw", "ivfpq", "lsh-minhash", "lsh-simhash"]

dfs = {}
dfs_build = {}

for algo in algos:
    dfs[algo] = pd.read_csv(f"../index/10k/{distance_metric}/{algo}/{algo}.csv")
    dfs_build[algo] = pd.read_csv(f"../index/10k/{distance_metric}/{algo}/{algo}_build.csv")

fig, ax = plt.subplots(figsize=(12, 7))
current_index = 0


def sort_data(values, labels):
    # Sort both values and labels based on values in ascending order
    sorted_pairs = sorted(zip(values, labels))
    sorted_values, sorted_labels = zip(*sorted_pairs)
    return list(sorted_values), list(sorted_labels)


def plot_recall_precision_f1_score(fig):
    fig.clear()
    axs = fig.subplots(1, 3)

    recalls, recalls_labels = sort_data([dfs[algo]['recall'].iloc[-1] for algo in algos], algos)
    precisions, precisions_labels = sort_data([dfs[algo]['precision'].iloc[-1] for algo in algos], algos)
    f1_scores, f1_scores_labels = sort_data([dfs[algo]['f1_score'].iloc[-1] for algo in algos], algos)

    # Plot recall
    bars1 = axs[0].bar(recalls_labels, recalls, color="gray")
    axs[0].set_title('Recall by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Recall')
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot precision
    bars2 = axs[1].bar(precisions_labels, precisions, color="gray")
    axs[1].set_title('Precision by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Precision')
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot precision
    bars3 = axs[2].bar(f1_scores_labels, f1_scores, color="gray")
    axs[2].set_title('F1 Score by Algorithm')
    axs[2].set_xlabel('Algorithm')
    axs[2].set_ylabel('F1 Score')
    # Add value labels on top of bars
    for bar in bars3:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()


def plot_recall_and_qps(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size for each algorithm
    recalls, recalls_labels = sort_data([dfs[algo]['recall'].iloc[-1] for algo in algos], algos)
    qps, qps_labels = sort_data([dfs[algo]['queries_per_second'].iloc[-1] for algo in algos], algos)

    # Plot recall
    bars1 = axs[0].bar(recalls_labels, recalls, color="gray")
    axs[0].set_title('Recall by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Recall')
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot QPS
    bars2 = axs[1].bar(qps_labels, qps, color="gray")
    axs[1].set_title('Queries per Second by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Queries per Second')
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_add_and_remove_vector(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size
    add_perf, add_perf_labels = sort_data([dfs[algo]['add_vector_performance'].iloc[-1] for algo in algos], algos)
    remove_perf, remove_perf_labels = sort_data([dfs[algo]['remove_vector_performance'].iloc[-1] for algo in algos], algos)

    # Plot add vector performance
    bars1 = axs[0].bar(add_perf_labels, add_perf, color="gray")
    axs[0].set_title('Add Vector Performance by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot remove vector performance
    bars2 = axs[1].bar(remove_perf_labels, remove_perf, color="gray")
    axs[1].set_title('Remove Vector Performance by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (s)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_build_and_search_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size
    build_times, build_times_labels = sort_data([dfs[algo]['build_time'].iloc[-1] for algo in algos], algos)
    search_times, search_times_labels = sort_data([dfs[algo]['search_time'].iloc[-1] for algo in algos], algos)

    # Plot build time
    bars1 = axs[0].bar(build_times_labels, build_times, color="gray")
    axs[0].set_title('Build Time by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot search time
    bars2 = axs[1].bar(search_times_labels, search_times, color="gray")
    axs[1].set_title('Search Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (s)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_saving_and_loading_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size
    save_times, save_times_labels = sort_data([dfs[algo]['index_saving_time'].iloc[-1] for algo in algos], algos)
    load_times, load_times_labels = sort_data([dfs[algo]['index_loading_time'].iloc[-1] for algo in algos], algos)

    # Plot saving time
    bars1 = axs[0].bar(save_times_labels, save_times, color="gray")
    axs[0].set_title('Index Saving Time by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot loading time
    bars2 = axs[1].bar(load_times_labels, load_times, color="gray")
    axs[1].set_title('Index Loading Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (s)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_cpu_memory_disk(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size
    cpu_usage, cpu_usage_labels = sort_data([dfs_build[algo]['consumed_cpu'].iloc[-1] for algo in algos], algos)
    memory_usage, memory_usage_labels = sort_data([dfs_build[algo]['consumed_memory'].iloc[-1] for algo in algos], algos)

    # Plot CPU usage
    bars1 = axs[0].bar(cpu_usage_labels, cpu_usage, color="gray")
    axs[0].set_title('CPU Usage by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('CPU Usage (%)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot memory usage
    bars2 = axs[1].bar(memory_usage_labels, memory_usage, color="gray")
    axs[1].set_title('Memory Usage by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Memory (MB)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_disk_size_and_execution_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    disk_space, disk_space_labels = sort_data([dfs[algo]['index_disk_space'].iloc[-1] for algo in algos], algos)
    execution_times, execution_times_labels = sort_data([dfs[algo]['execution_time'].iloc[-1] for algo in algos], algos)

    # Plot disk space
    bars1 = axs[0].bar(disk_space_labels, disk_space, color="gray")
    axs[0].set_title('Index Size by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Disk Space (MB)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot execution time
    bars2 = axs[1].bar(execution_times_labels, execution_times, color="gray")
    axs[1].set_title('Execution Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (s)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

plot_functions = [
    plot_recall_and_qps,
    plot_add_and_remove_vector,
    plot_build_and_search_time,
    plot_saving_and_loading_time,
    plot_cpu_memory_disk,
    plot_disk_size_and_execution_time,
    plot_recall_precision_f1_score,
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
