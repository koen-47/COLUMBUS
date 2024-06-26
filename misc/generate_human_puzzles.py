import json
import random
import matplotlib.pyplot as plt

random.seed(42)


with open("../benchmark.json", "r") as file:
    benchmark = json.load(file)
    random_sample = random.sample(benchmark, int(len(benchmark) / 10))
    with open("../results/analysis/results_v2/human/benchmark_human.json", "w") as file_:
        json.dump(random_sample, file_, indent=3)
