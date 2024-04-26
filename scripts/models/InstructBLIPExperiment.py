import os

import requests
import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class InstructBLIPExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "InstructBLIP"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_dir = f"{os.path.dirname(__file__)}/downloads"
        self.prompt = "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                      "(A) {} (B) {} (C) {} (D) {}"
        self._load_model()

    def _load_model(self):
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            cache_dir=self._models_dir,
        ).to(self.device)

        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            cache_dir=self._models_dir
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark()
        compounds, phrases = benchmark.get_puzzles()

        print(f"Starting experiment: {self.name}")
        print(f"Number of compounds: {len(compounds)}")
        print(f"Number of phrases: {len(phrases)}")
        print(f"Results save directory: {save_dir}")

        puzzle = compounds[1]
        image = Image.open(puzzle["image"]).convert("RGB")
        options = puzzle["options"]
        prompt = self.prompt.format(*options.values())
        puzzle["prompt"] = prompt

        print(f"Prompt: {prompt}")

        # url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # prompt = "What is unusual about this image?"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )

        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
