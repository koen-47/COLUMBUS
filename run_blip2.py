import json
import os

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
from tqdm import tqdm

from models.PromptManager import PromptManager

from models.BLIP2Experiment import BLIP2Experiment

blip2 = BLIP2Experiment()
blip2.run_on_benchmark("./test_compounds.json")
