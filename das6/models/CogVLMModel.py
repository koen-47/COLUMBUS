import json
import os

import requests
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class CogVLMModel(ModelExperiment):
    def __init__(self, prompt_type=1):
        super().__init__(prompt_type)
        self.name = f"CogVLM"
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
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            cache_dir=self.models_dir
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            cache_dir=self.models_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()

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
                inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[],
                                                                 images=[image], template_version='vqa')  # vqa mode
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                }
                gen_kwargs = {"max_length": 2048, "do_sample": False}

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    puzzle["output"] = self.tokenizer.decode(outputs[0])

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
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[],
                                                             images=[image], template_version='vqa')  # vqa mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": False}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                puzzle["output"] = self.tokenizer.decode(outputs[0])

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases_prompt_{self.prompt_type}.json",
                  "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)

