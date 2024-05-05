import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class InstructBLIPExperiment(ModelExperiment):
    def __init__(self, include_description=False):
        super().__init__(include_description)
        self.name = "InstructBLIP"
        if self.include_description:
            self.prompt = "<Image> "\
                          "Question: You are given a rebus puzzle. " \
                          "It consists of text that is used to convey a word or phrase. " \
                          "It needs to be solved through creative thinking. " \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "Options: (A) {} (B) {} (C) {} (D) {}. " \
                          "Short answer:"
        else:
            self.prompt = "<Image> Question: Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "Options: (A) {} (B) {} (C) {} (D) {}. Short answer:"
        self._load_model()

    def _load_model(self):
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            cache_dir=self.models_dir,
        ).to(self.device)

        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            cache_dir=self.models_dir
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
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{int(self.include_description) + 1}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

        self.delete_downloads()
