import json
import os
import shutil

import torch


class ModelExperiment:
    """
    Base class to handle model experiments.
    """
    def __init__(self, prompt_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = f"{os.path.dirname(__file__)}/downloads"
        self.prompt_type = prompt_type
        self.name = ""
        self.prompt = ""

        with open(f"{os.path.dirname(__file__)}/../data/misc/prompt_templates.json", "r") as file:
            self.prompt_templates = json.load(file)
        
    def run_on_benchmark(self, save_dir):
        """
        Base function for running a model on the benchmark and saveing it to a directory.

        :param save_dir: file path to directory where the results will be saved.
        """
        pass

    def delete_downloads(self):
        """
        Deletes all models in the model_dir folder.
        """
        models_dir = f"{os.path.dirname(__file__)}/downloads"
        for filename in os.listdir(models_dir):
            file_path = os.path.join(models_dir, filename)
            try:
                print(f"Removing file: {file_path}")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to remove: {file_path}. Reason: {e}")

    def get_metadata(self, benchmark, save_dir):
        """
        Returns a dictionary with some metadata relating to the model experiment being run.

        :param benchmark: benchmark object.
        :param save_dir: file path to directory where the results will be saved.
        :return: dictionary with metadata.
        """
        puzzles = benchmark.get_puzzles()
        return {
            "experiment": self.name,
            "prompt_type": self.prompt_type,
            "prompt_template": self.prompt,
            "n_puzzles": len(puzzles),
            "save_dir": save_dir,
            "models_dir": self.models_dir,
            "device": self.device
        }

