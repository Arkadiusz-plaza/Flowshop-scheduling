import matplotlib.pyplot as plt
import random
import time
import os
import sys

# === Funkcje wspólne ===
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

# === Algorytm GA ===
def initial_population(size, num_jobs):
    population = []
    for _ in range(size):
        individual = list(range(num_jobs))
        random.shuffle(individual)
        population.append(individual)
    return population

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end+1] = parent1[start:end+1]
    fill_pos = 0
    for gene in parent2:
        if gene not in child:
            while child[fill_pos] is not None:
                fill_pos += 1
            child[fill_pos] = gene
    return child

def mutate(individual, mutation_rate=0.1):
    for _ in range(int(len(individual) * mutation_rate)):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
    return individual

def genetic_algorithm(processing_times, population_size=300, generations=3000, mutation_rate=0.03):
    num_jobs = len(processing_times)
    population = initial_population(population_size, num_jobs)
    best_individual = None
    best_makespan = float('inf')

    for _ in range(generations):
        # Ewaluacja
        scored_population = []
        for individual in population:
            makespan, _ = calculate_makespan_from_order(individual, processing_times)
            scored_population.append((makespan, individual))
            if makespan < best_makespan:
                best_makespan = makespan
                best_individual = individual[:]

        scored_population.sort()
        population = [ind for _, ind in scored_population[:population_size//2]]

        # Krzyżowanie i mutacja
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population

    return best_individual

# === Funkcje pomocnicze ===
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
    ax.set_title("Wykres Gantta – algorytm GA")
    plt.tight_layout()
    plt.show()

# === Główna część ===
if __name__ == "__main__":
    filename = "flowshop_100jobs_20machines.txt"
    if not os.path.exists(filename):
        print(f"Plik {filename} nie został znaleziony!")
        sys.exit(1)

    num_jobs, num_machines, processing_times = read_flow_shop_data(filename)

    # Pomiar czasu działania
    start_algo = time.perf_counter()

    order = genetic_algorithm(processing_times, population_size=300, generations=2000, mutation_rate=0.05)
    makespan, _ = calculate_makespan_from_order(order, processing_times)
    start_times, end_times = calculate_schedule(order, processing_times)

    end_algo = time.perf_counter()
    algo_time = end_algo - start_algo

    # Wyniki
    print("Zadania w kolejności GA:", [i + 1 for i in order])
    print("Makespan:", makespan)
    print(f"Czas działania algorytmu (s): {algo_time:.6f}")

    utilizations = calculate_utilization(start_times, end_times, processing_times)
    print("\nWykorzystanie maszyn (%):")
    for i, u in enumerate(utilizations):
        print(f"Maszyna M{i+1}: {u:.2f}%")

    # plot_gantt(order, start_times, end_times, processing_times)
