import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import copy
import time

from lstm_data_prep import read_flow_shop_data, FlowShopDataset
from lstm_model import LSTMSequencer

def calculate_makespan_from_order(order, processing_times):
    num_jobs = len(order)
    num_machines = len(processing_times[0])
    completion = [[0] * num_machines for _ in range(num_jobs)]
    for i, job_index in enumerate(order):
        for m in range(num_machines):
            time_val = processing_times[job_index][m]
            if i == 0 and m == 0:
                completion[i][m] = time_val
            elif i == 0:
                completion[i][m] = completion[i][m-1] + time_val
            elif m == 0:
                completion[i][m] = completion[i-1][m] + time_val
            else:
                completion[i][m] = max(completion[i-1][m], completion[i][m-1]) + time_val
    return completion[-1][-1], completion

def calculate_utilization(start_times, end_times):
    num_machines = len(start_times[0])
    num_jobs = len(start_times)
    machine_work = [0] * num_machines
    for m in range(num_machines):
        for j in range(num_jobs):
            machine_work[m] += end_times[j][m] - start_times[j][m]
    makespan = end_times[-1][-1]
    utilizations = [(work / makespan) * 100 for work in machine_work]
    return utilizations

def plot_gantt(order, start_times, end_times, processing_times):
    num_jobs = len(order)
    num_machines = len(start_times[0])

    fig, ax = plt.subplots(figsize=(14, 6))

    # Paleta kolorow
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    def get_contrast_color(color):
        r, g, b = mcolors.to_rgb(color)
        brightness = r * 0.299 + g * 0.587 + b * 0.114
        return 'black' if brightness > 0.6 else 'white'

    for i in range(num_jobs):
        job_id = order[i]
        color = colors[job_id % len(colors)]
        for m in range(num_machines):
            start = start_times[i][m]
            duration = processing_times[job_id][m]
            ax.broken_barh([(start, duration)], (m * 10, 9),
                           facecolors=color, edgecolors='black', linewidth=0.5)
            text_color = get_contrast_color(color)
            ax.text(start + duration / 2, m * 10 + 4.5, f"Z{job_id + 1}",
                    ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')

    ax.set_xlabel("Czas", fontsize=12)
    ax.set_ylabel("Maszyny", fontsize=12)
    ax.set_yticks([i * 10 + 4.5 for i in range(num_machines)])
    ax.set_yticklabels([f"M{i+1}" for i in range(num_machines)])
    ax.set_title("Wykres Gantta – predykcja LSTM", fontsize=14, fontweight='bold')

    ax.grid(True, axis='x', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.set_ylim(0, num_machines * 10)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

def calculate_schedule(order, processing_times):
    num_jobs = len(order)
    num_machines = len(processing_times[0])
    start_times = [[0] * num_machines for _ in range(num_jobs)]
    end_times = [[0] * num_machines for _ in range(num_jobs)]
    for i, job_index in enumerate(order):
        for m in range(num_machines):
            start = 0
            if i > 0:
                start = end_times[i - 1][m]
            if m > 0:
                start = max(start, end_times[i][m - 1])
            start_times[i][m] = start
            end_times[i][m] = start + processing_times[job_index][m]
    return start_times, end_times

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

# --- GŁÓWNY KOD ---
if __name__ == "__main__":
    start_time = time.time()

    filename = "flowshop_100jobs_20machines.txt"
    if not os.path.exists(filename):
        print(f"Plik {filename} nie został znaleziony!")
        sys.exit(1)

    num_jobs, num_machines, processing_times = read_flow_shop_data(filename)
    dataset = FlowShopDataset(processing_times, list(range(num_jobs)))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMSequencer(input_size=1, hidden_size=128, output_size=num_jobs, num_layers=2).to(device)
    model.load_state_dict(torch.load("lstm_model.pth", map_location=device, weights_only=True))
    model.eval()

    raw_predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.unsqueeze(-1).to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            raw_predictions.append(predicted.item())

    unique_preds = []
    for p in raw_predictions:
        if p not in unique_preds:
            unique_preds.append(p)
    for i in range(num_jobs):
        if i not in unique_preds:
            unique_preds.append(i)
    predicted_order = unique_preds[:num_jobs]

    # print("Predykowana kolejność przed ulepszeniem:", [i + 1 for i in predicted_order])
    predicted_order = improve_order_local_search(predicted_order, processing_times)
    print("Predykowana kolejność:", [i + 1 for i in predicted_order])

    makespan, completion_times = calculate_makespan_from_order(predicted_order, processing_times)
    start_times, end_times = calculate_schedule(predicted_order, processing_times)
    utilization = calculate_utilization(start_times, end_times)

    print("Wykorzystanie maszyn (%):")
    for i, u in enumerate(utilization):
        print(f"Maszyna M{i+1}: {u:.2f}%")
    
    print("Makespan:", makespan)

    end_time = time.time()
    print(f"Czas działania algorytmu (s): {end_time - start_time:.6f}")


    # plot_gantt(predicted_order, start_times, end_times, processing_times)


