import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class MistralExperiment(ModelExperiment):
    def __init__(self):
        super().__init__(include_description=False)
        self.name = "Mistral-7b"
        self.prompt_nodes_edges = "You are given a description of a graph that is used to convey a word or phrase. " \
                                  "The nodes are elements that contain text that are manipulated through its attributes. " \
                                  "The edges define relationships between the nodes. The description is as follows:\n" \
                                  "{}\n" \
                                  "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                                  "(A) {} (B) {} (C) {} (D) {}"
        self.prompt_nodes = "You are given a description of a graph that is used to convey a word or phrase. " \
                            "The nodes are elements that contain text that are manipulated through its attributes. " \
                            "The description is as follows:\n" \
                            "{}\n" \
                            "Which word/phrase is conveyed in this description from the following options (either A, B, C, or D)?\n" \
                            "(A) {} (B) {} (C) {} (D) {}"
        self._load_model()

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir=self.models_dir,
            token="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"
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir=self.models_dir,
            token="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"
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark(with_metadata=True)
        compounds, phrases = benchmark.get_puzzles()

        metadata = self.get_metadata(benchmark, save_dir)
        print(json.dumps(metadata, indent=3))

        for puzzle in tqdm(compounds, desc=f"Prompting {self.name} (compounds)"):
            options = list(puzzle["options"].values())
            format_prompt_nodes = [puzzle["metadata"]] + options
            prompt_nodes = self.prompt_nodes.format(*format_prompt_nodes)

            messages = [
                {"role": "user", "content": prompt_nodes}
            ]

            model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
            nodes_generated_text = self.tokenizer.batch_decode(generated_ids)[0]
            puzzle["prompt"] = prompt_nodes
            puzzle["output"] = nodes_generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": compounds
            }, file, indent=3)

        for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
            options = list(puzzle["options"].values())
            format_prompt_nodes_edges = [puzzle["metadata"]["nodes_and_edges"]] + options
            prompt_nodes_edges = self.prompt_nodes_edges.format(*format_prompt_nodes_edges)

            messages = [
                {"role": "user", "content": prompt_nodes_edges}
            ]

            model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
            node_edges_generated_text = self.tokenizer.batch_decode(generated_ids)[0]

            format_prompt_nodes = [puzzle["metadata"]["nodes"]] + options
            prompt_nodes = self.prompt_nodes.format(*format_prompt_nodes)

            messages = [
                {"role": "user", "content": prompt_nodes}
            ]

            model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
            nodes_generated_text = self.tokenizer.batch_decode(generated_ids)[0]

            puzzle["prompt"] = {
                "nodes_and_edges": prompt_nodes_edges,
                "nodes": prompt_nodes
            }

            puzzle["output"] = {
                "nodes_and_edges": node_edges_generated_text,
                "nodes": nodes_generated_text
            }

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases.json", "w+") as file:
            json.dump({
                "metadata": metadata,
                "results": phrases
            }, file, indent=3)
