import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

distance_metric = "Jaccard"
algos = ["linscan"]

dfs = {}
dfs_build = {}

for algo in algos:
    dfs[algo] = pd.read_csv(f"../index/real/{distance_metric}/{algo}/{algo}.csv")
    dfs_build[algo] = pd.read_csv(f"../index/real/{distance_metric}/{algo}/{algo}_build.csv")

fig, ax = plt.subplots(figsize=(12, 7))
current_index = 0

def plot_recall_and_qps(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size for each algorithm
    recalls = [dfs[algo]['recall'].iloc[-1] for algo in algos]
    qps = [dfs[algo]['queries_per_second'].iloc[-1] for algo in algos]

    # Plot recall
    bars1 = axs[0].bar(algos, recalls)
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
    bars2 = axs[1].bar(algos, qps)
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
    add_perf = [dfs[algo]['add_vector_performance'].iloc[-1] for algo in algos]
    remove_perf = [dfs[algo]['remove_vector_performance'].iloc[-1] for algo in algos]

    # Plot add vector performance
    bars1 = axs[0].bar(algos, add_perf)
    axs[0].set_title('Add Vector Performance by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (in s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot remove vector performance
    bars2 = axs[1].bar(algos, remove_perf)
    axs[1].set_title('Remove Vector Performance by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (in s)')
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
    build_times = [dfs[algo]['build_time'].iloc[-1] for algo in algos]
    search_times = [dfs[algo]['search_time'].iloc[-1] for algo in algos]

    # Plot build time
    bars1 = axs[0].bar(algos, build_times)
    axs[0].set_title('Build Time by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (in s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot search time
    bars2 = axs[1].bar(algos, search_times)
    axs[1].set_title('Search Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (in s)')
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
    save_times = [dfs[algo]['index_saving_time'].iloc[-1] for algo in algos]
    load_times = [dfs[algo]['index_loading_time'].iloc[-1] for algo in algos]

    # Plot saving time
    bars1 = axs[0].bar(algos, save_times)
    axs[0].set_title('Index Saving Time by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Time (in s)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot loading time
    bars2 = axs[1].bar(algos, load_times)
    axs[1].set_title('Index Loading Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (in s)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_cpu_memory_disk(fig):
    fig.clear()
    axs = fig.subplots(1, 3)

    # Get values for the latest dataset size
    cpu_usage = [dfs_build[algo]['consumed_cpu'].iloc[-1] for algo in algos]
    memory_usage = [dfs_build[algo]['consumed_memory'].iloc[-1] for algo in algos]
    disk_space = [dfs[algo]['index_disk_space'].iloc[-1] for algo in algos]

    # Plot CPU usage
    bars1 = axs[0].bar(algos, cpu_usage)
    axs[0].set_title('CPU Usage by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('CPU Usage (%)')
    for bar in bars1:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot memory usage
    bars2 = axs[1].bar(algos, memory_usage)
    axs[1].set_title('Memory Usage by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Memory (MB)')
    for bar in bars2:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot disk space
    bars3 = axs[2].bar(algos, disk_space)
    axs[2].set_title('Index Size by Algorithm')
    axs[2].set_xlabel('Algorithm')
    axs[2].set_ylabel('Disk Space (MB)')
    for bar in bars3:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

def plot_scalability_and_execution_time(fig):
    fig.clear()
    axs = fig.subplots(1, 2)

    # Get values for the latest dataset size where scalability factor exists
    scalability = []
    for algo in algos:
        valid_data = dfs[algo][np.isfinite(dfs[algo]['scalability_factor'])]
        if len(valid_data) > 0:
            scalability.append(valid_data['scalability_factor'].iloc[-1])
        else:
            scalability.append(0)  # or np.nan

    execution_times = [dfs[algo]['execution_time'].iloc[-1] for algo in algos]

    # Plot scalability factor
    bars1 = axs[0].bar(algos, scalability)
    axs[0].set_title('Scalability Factor by Algorithm')
    axs[0].set_xlabel('Algorithm')
    axs[0].set_ylabel('Scalability Factor')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:  # Only show label if value exists
            axs[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

    # Plot execution time
    bars2 = axs[1].bar(algos, execution_times)
    axs[1].set_title('Execution Time by Algorithm')
    axs[1].set_xlabel('Algorithm')
    axs[1].set_ylabel('Time (in s)')
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
