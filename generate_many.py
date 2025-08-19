import os
import random

def generate_instance(num_jobs, num_machines, min_time=1, max_time=99):
    return [[random.randint(min_time, max_time) for _ in range(num_machines)] for _ in range(num_jobs)]

def save_instance_to_file(data, path):
    with open(path, "w") as f:
        f.write(f"{len(data)} {len(data[0])}\n")
        for row in data:
            f.write(" ".join(map(str, row)) + "\n")

def generate_multiple_instances(num_files=100, num_jobs=20, num_machines=5, output_dir="data/flowshop_instances"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, num_files + 1):
        instance = generate_instance(num_jobs, num_machines)
        filename = os.path.join(output_dir, f"flowshop_{num_jobs}x{num_machines}_{i:03}.txt")
        save_instance_to_file(instance, filename)
    print(f"Wygenerowano {num_files} plików w folderze '{output_dir}'.")

if __name__ == "__main__":
    generate_multiple_instances()
