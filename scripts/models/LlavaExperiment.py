import json
import os

from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class LlavaExperiment(ModelExperiment):
    def __init__(self, include_description=False):
        super().__init__(include_description)
        self.name = "Llava-1.5-13b"
        if self.include_description:
            self.prompt = "USER: <image>\n" \
                          "You are given a rebus puzzle. " \
                          "It consists of text that is used to convey a word or phrase. " \
                          "It needs to be solved through creative thinking. " \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}\n" \
                          "ASSISTANT:"
        else:
            self.prompt = "USER: <image>\n" \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {}\n" \
                          "ASSISTANT:"
        self._load_model()

    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            cache_dir=self.models_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
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
            inputs = self.processor(prompt, image, return_tensors='pt').to(device=self.device,
                                                                           dtype=torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
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
            inputs = self.processor(prompt, image, return_tensors='pt').to(device=self.device,
                                                                           dtype=torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{int(self.include_description) + 1}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

        self.delete_downloads()
