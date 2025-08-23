# Flowshop Scheduling — Heuristics, Metaheuristics & Neural Network

Solving the permutation flow shop scheduling problem (minimizing makespan) using:
- classic heuristics (NEH),
- metaheuristics (Simulated Annealing, Genetic Algorithm),
- a learning-based approach (LSTM neural network).

Author: Arkadiusz Płaza
GitHub: https://github.com/Arkadiusz-plaza

## What’s inside
- `neh.py` — NEH heuristic baseline.
- `simulated_annealing.py` — simulated annealing for sequence optimization.
- `genetic_algorithm.py` — GA (selection/crossover/mutation) for job ordering.
- `shortest_path.py` — helper utilities (e.g., schedule evaluation).
- `lstm_data_prep.py`, `lstm_model.py`, `lstm_train.py`, `lstm_predict.py` — LSTM pipeline.
- `generate_many.py` — dataset/instance generator.
- `images/` — figures used in the reports.

## Tech stack
- Python 3.11+
- numpy, pandas, matplotlib
- torch (PyTorch)
- scikit-learn

## Instances follow the standard flow shop definition:

n m
p11 p12 ... p1m
p21 p22 ... p2m
...
pn1 pn2 ... pnm

n = number of jobs
m = number of machines
pij = processing time of job i on machine j

## Results
- Main metric: makespan (lower = better).
- Comparison of heuristics, metaheuristics and LSTM available in `images/`.
