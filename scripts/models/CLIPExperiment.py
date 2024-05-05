import json
import os

import requests
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class CLIPExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "CLIP"
        self._load_model()

    def _load_model(self):
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=self.models_dir,
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=self.models_dir
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark()
        compounds, phrases = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = list(puzzle["options"].values())
            inputs = self.processor(text=options, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            generated_answer = options[np.argmax(probs)]
            puzzle["output"] = generated_answer

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": compounds
            }, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = list(puzzle["options"].values())
            inputs = self.processor(text=options, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]
            generated_answer = options[np.argmax(probs)]
            puzzle["output"] = generated_answer

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

        self.delete_downloads()
