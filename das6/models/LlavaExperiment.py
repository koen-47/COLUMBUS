import base64
import json
import os

import replicate
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class LlavaExperiment(ModelExperiment):
    def __init__(self, model_type, prompt_type=1):
        super().__init__(prompt_type)
        self.model_type = model_type

        if model_type == "13b":
            self.name = "Llava-1.5-13b"
            self.prompt_boilerplate = "USER: <image>\n{}\nASSISTANT:"
            self.prompt = self.prompt_boilerplate.format(self.prompt_templates["base"][str(self.prompt_type)])
        elif model_type == "34b":
            self.name = "Llava-1.6-34b"
            self.prompt_boilerplate = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n" \
                                      "<image>\n{}\n<|im_end|><|im_start|>assistant\n"
            self.prompt = self.prompt_boilerplate.format(self.prompt_templates["base"][str(self.prompt_type)])

        # self._load_model()

    def _load_model(self):
        if self.model_type == "13b":
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
        elif self.model_type == "34b":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-34b-hf",
                cache_dir=self.models_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config
            )

            self.processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-34b-hf",
                cache_dir=self.models_dir
            )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark(with_metadata=True)
        puzzles = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(puzzles, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt
            inputs = self.processor(prompt, image, return_tensors='pt').to(device=self.device,
                                                                           dtype=torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            if self.model_type == "13b":
                generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
            else:
                generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_prompt_{self.prompt_type}.json",
                  "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": puzzles
            }, file, indent=3)

        # self.delete_downloads()

    def run_on_benchmark_api(self, save_dir):
        benchmark = Benchmark(with_metadata=True)
        puzzles = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(puzzles, desc=f"Prompting {self.name} (phrases)"):
            with open(puzzle["image"], 'rb') as file:
                data = base64.b64encode(file.read()).decode('utf-8')

            image = f"data:application/octet-stream;base64,{data}"
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt

            input = {
                "image": image,
                "prompt": prompt
            }

            output = replicate.run(
                "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
                input=input
            )

            puzzle["output"] = "".join(output)

            with open(f"{save_dir}/{'_'.join(self.name.lower().split())}-api_prompt_{self.prompt_type}.json",
                      "w+") as file:
                json.dump({
                    "metadata": metadata,
                    "results": puzzles
                }, file, indent=3)
