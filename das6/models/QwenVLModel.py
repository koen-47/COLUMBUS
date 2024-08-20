import json
import os

import requests
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class QwenVLModel(ModelExperiment):
    """
    Class to handle Qwen-VL model experiments.
    """
    def __init__(self, prompt_type=1):
        super().__init__(prompt_type)
        self.name = f"QwenVL"
        self.prompt = self.prompt_templates["base"][str(self.prompt_type)]

        self._load_model()

    def _load_model(self):
        """
            Loads the Qwen-VL model
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            cache_dir=self.models_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            cache_dir=self.models_dir,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

    def run_on_benchmark(self, save_dir):
        """
        Runs the Qwen-VL model on the benchmark and saves it to a directory. This also deletes the model files at
        the end of the run.

        :param save_dir: file path to directory where the results will be saved.
        """
        benchmark = Benchmark(with_metadata=True)
        puzzles = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(puzzles, desc=f"Prompting {self.name}"):
            image = puzzle["image"]
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt
            query = self.tokenizer.from_list_format([
                {'image': image},
                {'text': prompt},
            ])
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            puzzle["output"] = response

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_prompt_{self.prompt_type}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": puzzles
            }, file, indent=3)

        self.delete_downloads()
