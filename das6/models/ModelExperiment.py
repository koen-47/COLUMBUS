import os
import shutil

import torch


class ModelExperiment:
    def __init__(self, prompt_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = f"{os.path.dirname(__file__)}/downloads"
        self.prompt_type = prompt_type
        self.name = ""
        self.prompt = ""
        
    def run_on_benchmark(self, save_dir):
        pass

    def delete_downloads(self):
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

