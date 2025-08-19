import torch
from torch.utils.data import Dataset
import os
import copy

def load_all_instances(folder_path):
    datasets = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            num_jobs, num_machines, processing_times = read_flow_shop_data(full_path)
            labels = generate_labels_with_local_search(processing_times)  # ZAMIANA FUNKCJI
            dataset = FlowShopDataset(processing_times, labels)
            datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def read_flow_shop_data(filename):
    with open(filename, "r") as f:
        num_jobs, num_machines = map(int, f.readline().split())
        processing_times = [list(map(int, line.strip().split())) for line in f]
    return num_jobs, num_machines, processing_times

def generate_labels_with_neh(processing_times):
    num_jobs = len(processing_times)
    num_machines = len(processing_times[0])

    job_sums = [(i, sum(processing_times[i])) for i in range(num_jobs)]
    job_sums.sort(key=lambda x: -x[1])
    sequence = []

    for job_id, _ in job_sums:
        best_sequence = None
        best_makespan = float('inf')
        for i in range(len(sequence) + 1):
            test_seq = copy.deepcopy(sequence)
            test_seq.insert(i, job_id)
            makespan, _ = calculate_makespan_from_order(test_seq, processing_times)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = test_seq
        sequence = best_sequence

    labels = [0] * num_jobs
    for position, job_id in enumerate(sequence):
        if job_id < 0 or job_id >= num_jobs:
            raise ValueError(f"Błąd etykiety: job_id={job_id} przy num_jobs={num_jobs}")
        labels[job_id] = position

    return labels

def generate_labels_with_local_search(processing_times):
    initial_order = generate_order_with_neh(processing_times)
    improved_order = improve_order_local_search(initial_order, processing_times)

    num_jobs = len(processing_times)
    labels = [0] * num_jobs
    for position, job_id in enumerate(improved_order):
        labels[job_id] = position

    return labels

def generate_order_with_neh(processing_times):
    job_sums = [(i, sum(processing_times[i])) for i in range(len(processing_times))]
    job_sums.sort(key=lambda x: -x[1])
    sequence = []

    for job_id, _ in job_sums:
        best_sequence = None
        best_makespan = float('inf')
        for i in range(len(sequence) + 1):
            test_seq = copy.deepcopy(sequence)
            test_seq.insert(i, job_id)
            makespan, _ = calculate_makespan_from_order(test_seq, processing_times)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = test_seq
        sequence = best_sequence

    return sequence

def improve_order_local_search(order, processing_times, iterations=10):
    best_order = copy.deepcopy(order)
    best_makespan, _ = calculate_makespan_from_order(best_order, processing_times)

    for _ in range(iterations):
        improved = False

        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan, _ = calculate_makespan_from_order(new_order, processing_times)
                if new_makespan < best_makespan:
                    best_order = new_order
                    best_makespan = new_makespan
                    improved = True

        for i in range(len(best_order)):
            for j in range(len(best_order)):
                if i == j:
                    continue
                new_order = best_order[:]
                job = new_order.pop(i)
                new_order.insert(j, job)
                new_makespan, _ = calculate_makespan_from_order(new_order, processing_times)
                if new_makespan < best_makespan:
                    best_order = new_order
                    best_makespan = new_makespan
                    improved = True

        if not improved:
            break

    return best_order

def calculate_makespan_from_order(order, processing_times):
    num_jobs = len(order)
    num_machines = len(processing_times[0])
    completion = [[0] * num_machines for _ in range(num_jobs)]

    for i, job_idx in enumerate(order):
        for m in range(num_machines):
            time_val = processing_times[job_idx][m]
            if i == 0 and m == 0:
                completion[i][m] = time_val
            elif i == 0:
                completion[i][m] = completion[i][m-1] + time_val
            elif m == 0:
                completion[i][m] = completion[i-1][m] + time_val
            else:
                completion[i][m] = max(completion[i-1][m], completion[i][m-1]) + time_val

    makespan = completion[-1][-1]
    return makespan, completion

class FlowShopDataset(Dataset):
    def __init__(self, processing_times, labels):
        self.processing_times = torch.tensor(processing_times, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        num_jobs = self.processing_times.shape[0]
        for label in self.labels:
            if label.item() < 0 or label.item() >= num_jobs:
                raise ValueError(f"Nieprawidłowa etykieta: {label.item()} dla {num_jobs} zadań!")

    def __len__(self):
        return len(self.processing_times)

    def __getitem__(self, idx):
        return self.processing_times[idx], self.labels[idx]