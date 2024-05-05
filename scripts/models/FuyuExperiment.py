import json
import os

import requests
from PIL import Image
from tqdm import tqdm
from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig
import torch

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class FuyuExperiment(ModelExperiment):
    def __init__(self, include_description=False):
        super().__init__(include_description)
        self.name = "Fuyu-8b"
        if self.include_description:
            self.prompt = "You are given a rebus puzzle. " \
                          "It consists of text that is used to convey a word or phrase. " \
                          "It needs to be solved through creative thinking. " \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}\n"
        else:
            self.prompt = "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}\n"
        self._load_model()

    def _load_model(self):
        self.processor = FuyuProcessor.from_pretrained(
            "adept/fuyu-8b",
            cache_dir=self.models_dir
        )
        self.model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b",
            cache_dir=self.models_dir,
            device_map={"": 0},
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark()
        compounds, phrases = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            puzzle["prompt"] = prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                       dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=8)
            generated_text = self.processor.batch_decode(generated_ids[:, -8:], skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds_prompt_{int(self.include_description) + 1}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": compounds
            }, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            puzzle["prompt"] = prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                       dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=8)
            generated_text = self.processor.batch_decode(generated_ids[:, -8:], skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{int(self.include_description) + 1}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

        self.delete_downloads()
