import json
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class MistralExperiment(ModelExperiment):
    """
    Class to handle Mistral model experiments.
    """
    def __init__(self, prompt_type=3):
        super().__init__(prompt_type)
        self.name = "Mistral-7b"
        self.prompt_type = prompt_type
        self.prompt = self.prompt_templates["mistral"][str(self.prompt_type)]

        self._load_model()

    def _load_model(self):
        """
            Loads the Mistral model
        """

        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir=self.models_dir,
            token=os.getenv("MISTRAL_API_KEY")
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir=self.models_dir,
            token=os.getenv("MISTRAL_API_KEY")
        )

    def run_on_benchmark(self, save_dir):
        """
        Runs the Mistral model on the benchmark and saves it to a directory. This also deletes the model files at
        the end of the run.

        :param save_dir: file path to directory where the results will be saved.
        """
        benchmark = Benchmark(with_metadata=True)
        puzzles = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(puzzles, desc=f"Prompting {self.name}"):
            options = puzzle["options"]
            prompt_format = list(options.values())
            if self.prompt_type == 3:
                prompt_format = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif self.prompt_type == 4:
                prompt_format = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            prompt = self.prompt.format(*prompt_format)
            puzzle["prompt"] = prompt

            messages = [
                {"role": "user", "content": prompt}
            ]

            model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
            generated_text = self.tokenizer.batch_decode(generated_ids)[0]
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_prompt_{self.prompt_type}.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": puzzles
            }, file, indent=3)

        self.delete_downloads()