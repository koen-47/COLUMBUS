import random

from tqdm import tqdm

from graphs.BeliefGraphGenerator import BeliefGraphGenerator
from graphs.BeliefGraphReasoner import BeliefGraphReasoner
from puzzles.Benchmark import Benchmark

random.seed(42)

n_puzzles = 50
benchmark = Benchmark()
puzzles = random.sample(benchmark.get_puzzles(), n_puzzles)


for puzzle in tqdm(puzzles[:1], desc="Running belief graphs"):
    image = puzzle["image"]
    options = list(puzzle["options"].values())
    generator = BeliefGraphGenerator(image, options, max_depth=1)
    graph = generator.generate_graph()

    reasoner = BeliefGraphReasoner()
    graph, _ = reasoner.fix_graph(graph)

    orig_hypotheses = graph.get_original_hypotheses()
    answer = graph.get_answer()
    puzzle["output"] = answer

