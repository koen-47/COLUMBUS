import os
import argparse


from models.BLIP2Experiment import BLIP2Experiment
from models.FuyuExperiment import FuyuExperiment
from models.LlavaExperiment import LlavaExperiment
from models.InstructBLIPExperiment import InstructBLIPExperiment
from data.Benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)

    args = parser.parse_args()
    model = args.model

    print(f"Running experiment... (model: {model})")
    if model == "blip2-2.7b":
        blip2_experiment = BLIP2Experiment(size="2.7b")
        blip2_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "blip2-6.7b":
        blip2_experiment = BLIP2Experiment(size="6.7b")
        blip2_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "instruct-blip":
        instruct_blip_experiment = InstructBLIPExperiment()
        instruct_blip_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "fuyu":
        fuyu_experiment = FuyuExperiment()
        fuyu_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "llava":
        llava_experiment = LlavaExperiment()
        llava_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    else:
        print("Error: specified model does not exist")


if __name__ == "__main__":
    # benchmark = Benchmark(with_metadata=True)
    # compounds, phrases = benchmark.get_puzzles()

    main()
