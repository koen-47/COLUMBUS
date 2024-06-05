import json
import os

from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class BLIP2Experiment(ModelExperiment):
    def __init__(self, model_type, prompt_type=1):
        super().__init__(prompt_type)
        self.model_type = model_type
        self.name = f"BLIP-2 {model_type}"
        if self.prompt_type == 1:
            self.prompt = "Question: Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {} Answer:"
        if self.prompt_type == 2:
            self.prompt = "Question: You are given a rebus puzzle. " \
                          "It consists of text that is used to convey a word or phrase. " \
                          "It needs to be solved through creative thinking. " \
                          "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                          "(A) {} (B) {} (C) {} (D) {} " \
                          "Answer:"
        elif self.prompt_type == 3:
            self.prompt = "Question: You are given a description of a graph that is used to convey a word or phrase. " \
                          "The nodes are elements that contain text that are manipulated through its attributes. " \
                          "The description is as follows:\n" \
                          "{}\n" \
                          "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                          "(A) {} (B) {} (C) {} (D) {} " \
                          "Answer:"
        elif self.prompt_type == 4:
            self.prompt = "Question: You are given a description of a graph that is used to convey a word or phrase. " \
                          "The nodes are elements that contain text that are manipulated through its attributes. " \
                          "The edges define relationships between the nodes. The description is as follows:\n" \
                          "{}\n" \
                          "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                          "(A) {} (B) {} (C) {} (D) {} " \
                          "Answer:"

        self._load_model()

    def _load_model(self):
        self.processor = Blip2Processor.from_pretrained(
            f"Salesforce/blip2-{self.model_type}",
            cache_dir=self.models_dir
        )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/blip2-{self.model_type}",
            cache_dir=self.models_dir,
            device_map={"": 0},
            torch_dtype=torch.float16
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark(with_metadata=True)
        compounds, phrases = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        if self.prompt_type != 4:
            for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
                image = Image.open(puzzle["image"]).convert("RGB")
                options = puzzle["options"]
                prompt_format = list(options.values())
                if self.prompt_type == 3 or self.prompt_type == 4:
                    prompt_format = [puzzle["metadata"]] + list(options.values())
                prompt = self.prompt.format(*prompt_format)
                puzzle["prompt"] = prompt
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                           dtype=torch.float16)
                generated_ids = self.model.generate(**inputs, max_length=256)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                puzzle["output"] = generated_text

            with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds_prompt_{self.prompt_type}.json", "w+") as file:
                json.dump({
                    "metadata": metadata,
                    "results": compounds
                }, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device,
                                                                                       dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{self.prompt_type}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

        # self.delete_downloads()
