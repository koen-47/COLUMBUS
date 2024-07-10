import json
import os
import argparse

import matplotlib.pyplot as plt

from results.analysis.AnalysisReport import AnalysisReport
from results.benchmark.PuzzleAnalysisReport import PuzzleAnalysisReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-benchmark", action="store_true")
    parser.add_argument("--show-analysis", type=str)

    args = parser.parse_args()
    show_benchmark = args.show_benchmark
    analysis = args.show_analysis

    if show_benchmark:
        with open("./benchmark.json", "r") as file:
            benchmark = json.load(file)
        for i, puzzle in enumerate(benchmark):
            print(f"\n=== Puzzle {i+1} ({puzzle['image']}) ===")
            print(f"Options: {json.dumps(puzzle['options'])}")
            print(f"Correct: {puzzle['correct']}")
            plt.rcParams["figure.figsize"] = (4, 4)
            plt.axis('off')
            image_path = f"{os.path.dirname(__file__)}/results/benchmark/final_v2/{puzzle['image']}"
            image = plt.imread(image_path)
            plt.imshow(image)
            plt.tight_layout()
            plt.show()
    elif analysis is not None:
        if analysis == "puzzles":
            puzzle_analysis = PuzzleAnalysisReport()
            puzzle_analysis.generate()
        elif analysis == "models":
            model_analysis = AnalysisReport()
            model_analysis.generate_all(verbose=True)


if __name__ == "__main__":
    main()
