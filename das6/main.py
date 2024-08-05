import json
import os
import argparse

from models.BLIP2Experiment import BLIP2Experiment
from models.FuyuExperiment import FuyuExperiment
from models.LlavaExperiment import LlavaExperiment
from models.InstructBLIPExperiment import InstructBLIPExperiment
from models.CLIPExperiment import CLIPExperiment
from models.CogVLMModel import CogVLMModel
from models.QwenVLModel import QwenVLModel
from models.MistralExperiment import MistralExperiment
from data.Benchmark import Benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("prompt_type", type=int)

    args = parser.parse_args()
    model = args.model
    prompt_type = int(args.prompt_type)

    if model == "mistral" and (prompt_type == 1 or prompt_type == 2):
        print(f"Unable to run {model} with prompt {prompt_type}")
        return -1

    save_dir = f"{os.path.dirname(__file__)}/results/prompt_{prompt_type}"

    print(f"Running experiment... (model: {model}, prompt type: {prompt_type})")
    if model == "blip2-opt-2.7b":
        blip2_experiment = BLIP2Experiment(model_type="opt-2.7b", prompt_type=prompt_type)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "blip2-opt-6.7b":
        blip2_experiment = BLIP2Experiment(model_type="opt-6.7b", prompt_type=prompt_type)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "blip2-flan-t5":
        blip2_experiment = BLIP2Experiment(model_type="flan-t5-xxl", prompt_type=prompt_type)
        blip2_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "instruct-blip":
        instruct_blip_experiment = InstructBLIPExperiment(prompt_type=prompt_type)
        instruct_blip_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "fuyu":
        fuyu_experiment = FuyuExperiment(prompt_type=prompt_type)
        fuyu_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "llava-1.5-13b":
        llava_experiment = LlavaExperiment(model_type="13b", prompt_type=prompt_type)
        llava_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "llava-1.6-34b":
        llava_experiment = LlavaExperiment(model_type="34b", prompt_type=prompt_type)
        llava_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "clip":
        clip_experiment = CLIPExperiment()
        clip_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    elif model == "cogvlm":
        cogvlm_experiment = CogVLMModel(prompt_type=prompt_type)
        cogvlm_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "qwenvl":
        qwenvl_experiment = QwenVLModel(prompt_type=prompt_type)
        qwenvl_experiment.run_on_benchmark(save_dir=save_dir)
    elif model == "mistral":
        mistral_experiment = MistralExperiment(prompt_type=prompt_type)
        mistral_experiment.run_on_benchmark(save_dir=f"{os.path.dirname(__file__)}/results")
    else:
        print("Error: specified model does not exist")


if __name__ == "__main__":
    main()

    # print(json.dumps(Benchmark().get_puzzles(), indent=3))
