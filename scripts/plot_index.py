import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../index/2024-09-25_10-24-41/annoy_serial_500.csv")

plt.figure(figsize=(10, 6))

# -- Execution Time vs. Dataset Size
# plt.plot(data['dataset_size'], data['execution_time'], marker='o')
# plt.title('Execution Time vs Dataset Size')
# plt.xlabel('Dataset Size')
# plt.ylabel('Execution Time (s)')
# plt.grid(True)
# plt.show()

# -- Recall vs. Dataset Size
# plt.plot(data['dataset_size'], data['recall'], marker='o')
# plt.title('Recall vs Dataset Size')
# plt.xlabel('Dataset Size')
# plt.ylabel('Recall')
# plt.grid(True)
# plt.show()

# -- QPS vs. Dataset Size
# plt.plot(data['dataset_size'], data['queries_per_second'], marker='o')
# plt.title('Queries per Second vs Dataset Size')
# plt.xlabel('Dataset Size')
# plt.ylabel('Queries per Second')
# plt.grid(True)
# plt.show()

# -- Index Disk Space vs. Dataset Size
# plt.plot(data['dataset_size'], data['index_disk_space'], marker='o')
# plt.title('Index Disk Space vs Dataset Size')
# plt.xlabel('Dataset Size')
# plt.ylabel('Index Disk Space (MB)')
# plt.grid(True)
# plt.show()

# -- Performance Metrics Comparison
# plt.figure(figsize=(12, 6))
# plt.plot(data['dataset_size'], data['add_vector_performance'], marker='o', label='Add Vector')
# plt.plot(data['dataset_size'], data['remove_vector_performance'], marker='s', label='Remove Vector')
# plt.plot(data['dataset_size'], data['build_time'], marker='^', label='Build Time')
# plt.plot(data['dataset_size'], data['search_time'], marker='D', label='Search Time')
# plt.title('Performance Metrics Comparison')
# plt.xlabel('Dataset Size')
# plt.ylabel('Time (s)')
# plt.legend()
# plt.grid(True)

# -- Index Operations Time
# plt.plot(data['dataset_size'], data['index_saving_time'], marker='o', label='Saving Time')
# plt.plot(data['dataset_size'], data['index_loading_time'], marker='s', label='Loading Time')
# plt.title('Index Operations Time vs Dataset Size')
# plt.xlabel('Dataset Size')
# plt.ylabel('Time (s)')
# plt.legend()
# plt.grid(True)
# plt.show()
