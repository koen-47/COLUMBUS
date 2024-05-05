import json
import os
import argparse


from models.BLIP2Experiment import BLIP2Experiment
from models.FuyuExperiment import FuyuExperiment
from models.LlavaExperiment import LlavaExperiment
from models.InstructBLIPExperiment import InstructBLIPExperiment
from models.CLIPExperiment import CLIPExperiment
from models.MistralExperiment import MistralExperiment
from data.Benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)

    args = parser.parse_args()
    model = args.model

    include_description = True
    save_dir = f"{os.path.dirname(__file__)}/results/prompt_{int(include_description)+1}"

    print(f"Running experiment... (model: {model})")
    if model == "blip2-opt-2.7b":
        blip2_experiment = BLIP2Experiment(model_type="opt-2.7b", include_description=include_description)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "blip2-opt-6.7b":
        blip2_experiment = BLIP2Experiment(model_type="opt-6.7b", include_description=include_description)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "blip2-flan-t5":
        blip2_experiment = BLIP2Experiment(model_type="flan-t5-xxl", include_description=include_description)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "instruct-blip":
        instruct_blip_experiment = InstructBLIPExperiment(include_description=include_description)
        instruct_blip_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "fuyu":
        fuyu_experiment = FuyuExperiment(include_description=include_description)
        fuyu_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "llava":
        llava_experiment = LlavaExperiment(include_description=include_description)
        llava_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "clip":
        clip_experiment = CLIPExperiment()
        clip_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "mistral":
        mistral_experiment = MistralExperiment()
        mistral_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    else:
        print("Error: specified model does not exist")


if __name__ == "__main__":
    main()
