import json
import os

from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class BLIP2Experiment(ModelExperiment):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.name = f"BLIP-2 {size}"
        self._models_dir = f"{os.path.dirname(__file__)}/downloads"
        self.prompt = "Question: Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                      "(A) {} (B) {} (C) {} (D) {} Answer:"
        self._load_model()

    def _load_model(self):
        self.processor = Blip2Processor.from_pretrained(
            f"Salesforce/blip2-opt-{self.size}",
            cache_dir=self._models_dir
        )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/blip2-opt-{self.size}",
            cache_dir=self._models_dir,
            device_map={"": 0},
            torch_dtype=torch.float16
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark()
        compounds, phrases = benchmark.get_puzzles()

        print(f"Starting experiment: {self.name}")
        print(f"Number of compounds: {len(compounds)}")
        print(f"Number of phrases: {len(phrases)}")
        print(f"Results save directory: {save_dir}")

        for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            puzzle["prompt"] = prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                       dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds.json", "w+") as file:
            json.dump(compounds, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            puzzle["prompt"] = prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                       dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases.json", "w+") as file:
            json.dump(phrases, file, indent=3)

        self.delete_downloads()
