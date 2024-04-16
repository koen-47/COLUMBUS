import json
import os

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
from tqdm import tqdm

from models.PromptManager import PromptManager

from models.BLIP2Experiment import BLIP2Experiment
from models.FuyuExperiment import FuyuExperiment

blip2 = BLIP2Experiment(size="6.7b")
blip2.run_on_benchmark(f"{os.path.dirname(__file__)}/results/experiments")

# fuyu = FuyuExperiment()
# fuyu.run_on_benchmark(f"{os.path.dirname(__file__)}/results/experiments")
