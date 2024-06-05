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
    def __init__(self, prompt_type=1):
        super().__init__(prompt_type)
        self.name = f"QwenVL"
        if self.prompt_type == 1:
            self.prompt = "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}"
        if self.prompt_type == 2:
            self.prompt = "You are given a rebus puzzle. " \
                          "It consists of text that is used to convey a word or phrase. " \
                          "It needs to be solved through creative thinking. " \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}"
        elif self.prompt_type == 3:
            self.prompt = "You are given a description of a graph that is used to convey a word or phrase. " \
                          "The nodes are elements that contain text that are manipulated through its attributes. " \
                          "The description is as follows:\n" \
                          "{}\n" \
                          "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                          "(A) {} (B) {} (C) {} (D) {}"
        elif self.prompt_type == 4:
            self.prompt = "You are given a description of a graph that is used to convey a word or phrase. " \
                          "The nodes are elements that contain text that are manipulated through its attributes. " \
                          "The edges define relationships between the nodes. The description is as follows:\n" \
                          "{}\n" \
                          "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                          "(A) {} (B) {} (C) {} (D) {}"

        self._load_model()

    def _load_model(self):
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
        benchmark = Benchmark(with_metadata=True)
        compounds, phrases = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        if self.prompt_type != 4:
            for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
                image = puzzle["image"]
                options = puzzle["options"]
                prompt_format = list(options.values())
                if self.prompt_type == 3 or self.prompt_type == 4:
                    prompt_format = [puzzle["metadata"]] + list(options.values())
                prompt = self.prompt.format(*prompt_format)
                puzzle["prompt"] = prompt

                query = self.tokenizer.from_list_format([
                    {'image': image},
                    {'text': prompt},
                ])
                response, history = self.model.chat(self.tokenizer, query=query, history=None)
                puzzle["output"] = response

            with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds_prompt_{self.prompt_type}.json",
                      "w+") as file:
                json.dump({
                    "metadata": metadata,
                    "results": compounds
                }, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
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

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{self.prompt_type}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)
