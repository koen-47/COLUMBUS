import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class InstructBLIPExperiment(ModelExperiment):
    def __init__(self, prompt_type=1):
        super().__init__(prompt_type)
        self.name = "InstructBLIP"
        prompt_format = [self.prompt_templates["instructblip"][self.prompt_type],
                         "(A) {} (B) {} (C) {} (D) {}"]
        self.prompt_boilerplate = "<Image> Question: {} Options: {}. Short answer:"
        self.prompt = self.prompt_boilerplate.format(*prompt_format)

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
        benchmark = Benchmark(with_metadata=True)
        puzzles = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(puzzles, desc=f"Prompting {self.name}"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt
            print(f"PHRASE {puzzle['correct']}", prompt)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                num_beams=5,
                max_length=512,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )

            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_prompt_{self.prompt_type}.json",
                  "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": puzzles
            }, file, indent=3)

        # self.delete_downloads()
