import matplotlib.pyplot as plt
import time
import os
import sys
import copy

def read_flow_shop_data(filename):
    with open(filename, "r") as f:
        num_jobs, num_machines = map(int, f.readline().split())
        processing_times = [list(map(int, line.strip().split())) for line in f]
    return num_jobs, num_machines, processing_times

def calculate_makespan_from_order(order, processing_times):
    num_jobs = len(order)
    num_machines = len(processing_times[0])
    completion = [[0] * num_machines for _ in range(num_jobs)]

    for i, job_index in enumerate(order):
        for m in range(num_machines):
            time_value = processing_times[job_index][m]
            if i == 0 and m == 0:
                completion[i][m] = time_value
            elif i == 0:
                completion[i][m] = completion[i][m-1] + time_value
            elif m == 0:
                completion[i][m] = completion[i-1][m] + time_value
            else:
                completion[i][m] = max(completion[i-1][m], completion[i][m-1]) + time_value
    return completion[-1][-1], completion

def neh_order(processing_times):
    job_scores = [(sum(times), idx) for idx, times in enumerate(processing_times)]
    job_scores.sort(reverse=True)  # sortujemy malejąco
    ordered_jobs = []

    for _, job in job_scores:
        best_makespan = float('inf')
        best_sequence = []
        for i in range(len(ordered_jobs)+1):
            temp_sequence = ordered_jobs[:i] + [job] + ordered_jobs[i:]
            makespan, _ = calculate_makespan_from_order(temp_sequence, processing_times)
            if makespan < best_makespan:
                best_makespan = makespan
                best_sequence = temp_sequence
        ordered_jobs = best_sequence
    return ordered_jobs

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

def calculate_utilization(start_times, end_times, processing_times):
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
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(num_jobs):
        job_id = order[i]
        for m in range(num_machines):
            start = start_times[i][m]
            duration = processing_times[job_id][m]
            ax.broken_barh([(start, duration)], (m * 10, 9),
                           facecolors=f"C{job_id % 10}")
            ax.text(start + duration / 2, m * 10 + 4.5, f"Z{job_id + 1}",
                    ha='center', va='center', color='white', fontsize=8)

    ax.set_xlabel("Czas")
    ax.set_ylabel("Maszyny")
    ax.set_yticks([i * 10 + 4.5 for i in range(num_machines)])
    ax.set_yticklabels([f"M{i+1}" for i in range(num_machines)])
    ax.set_title("Wykres Gantta – algorytm NEH")
    plt.tight_layout()
    plt.show()

# === GŁÓWNA CZĘŚĆ ===
if __name__ == "__main__":
    filename = "flowshop_100jobs_20machines.txt"
    if not os.path.exists(filename):
        print(f"Plik {filename} nie został znaleziony!")
        sys.exit(1)

    num_jobs, num_machines, processing_times = read_flow_shop_data(filename)

    # Pomiar czasu działania algorytmu
    start_algo = time.perf_counter()

    order = neh_order(processing_times)
    makespan, _ = calculate_makespan_from_order(order, processing_times)
    start_times, end_times = calculate_schedule(order, processing_times)

    end_algo = time.perf_counter()
    algo_time = end_algo - start_algo

    # Wyniki
    print("Zadania w kolejności NEH:", [i + 1 for i in order])
    print("Makespan:", makespan)
    print(f"Czas działania algorytmu (s): {algo_time:.6f}")

    utilizations = calculate_utilization(start_times, end_times, processing_times)
    print("\nWykorzystanie maszyn (%):")
    for i, u in enumerate(utilizations):
        print(f"Maszyna M{i+1}: {u:.2f}%")

    # plot_gantt(order, start_times, end_times, processing_times)
