"""
Script to run our backward chaining approach through belief graphs (see the graphs directory).
"""

import json
import random

from tqdm import tqdm

from graphs.BeliefGraphGenerator import BeliefGraphGenerator
from graphs.BeliefGraphReasoner import BeliefGraphReasoner
from puzzles.Benchmark import Benchmark

# Set the seed
seed = 43
random.seed(seed)

# Load the benchmark and sample 50 random puzzles
n_puzzles = 50
benchmark = Benchmark()
puzzles = random.sample(benchmark.get_puzzles(), n_puzzles)

# Define hyperparameters
model = "gpt-4o"
max_depth = 1
n_examples = 0
verbose = False
hyperparameters = {
    "k": 9,
    "k_entailer": 36,
    "k_xor": 30,
    "k_mc": 9,
    "t_entailer": 1.02,
    "t_xor": 1.1,
    "t_mc": 0.98,
    "m_xor": 0.3,
    "c_xor": 1.,
    "c_mc": 1.
}

# Loop over puzzles, generate a belief graph for it, and optimize the graph to solve the given puzzle
for puzzle in tqdm(puzzles, desc="Running belief graphs"):
    # Get path to image and options
    image = puzzle["image"]
    options = list(puzzle["options"].values())

    # Generate a belief graph
    generator = BeliefGraphGenerator(image, n_examples, options, hyperparameters, max_depth=max_depth, model=model)
    graph = generator.generate_graph()

    # Optimize belief graph by fixing logical conflicts
    reasoner = BeliefGraphReasoner(hyperparameters)
    graph, _ = reasoner.fix_graph(graph, verbose=verbose)

    print(graph)
    graph.visualize(show=True)

    # Compute answer
    answer_csp = graph.get_answer()
    puzzle["output"] = answer_csp

    print(f"\nOptions:", options)
    print("Correct:", puzzle["correct"])
    print("Answer:", answer_csp)


# Save results to json file
with open(f"../results/analysis/results/belief_graphs_{model}.json", "w") as file:
    metadata = {
        "experiment": "Belief Graphs",
        "model": model,
        "seed": seed,
        "max_depth": max_depth,
        "n_puzzles": n_puzzles,
        "hyperparameters": hyperparameters
    }

    json.dump({
        "metadata": metadata,
        "results": puzzles
    }, file, indent=3)
