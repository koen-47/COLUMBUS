import json
import os
import shutil

import torch
from tqdm import tqdm
from PIL import Image


class ModelExperiment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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

