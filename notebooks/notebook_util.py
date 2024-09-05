import os
import json
import base64
import time
import requests

import pandas as pd
from PIL import Image
import google.generativeai as genai

def show_puzzles_3x3():
    pass

def load_inputs_in_columbus():
    compounds = pd.read_csv("../data/input/ladec_raw_small.csv")
    custom_compounds = pd.read_csv("../data/input/custom_compounds.csv")
    compounds = pd.concat([compounds, custom_compounds]).reset_index(drop=True)

    compounds_in_columbus = []
    phrases_in_columbus = []
    with open("../benchmark.json", "r") as file:
        benchmark = json.load(file)
        for puzzle in benchmark:
            answer = os.path.basename(puzzle["image"]).split(".")[0]
            answer_parts = answer.split("_")
            if answer.endswith("non-icon") or answer.endswith("icon"):
                answer_parts = answer_parts[:-1]
            if answer_parts[-1].isnumeric():
                answer_parts = answer_parts[:-1]
            if len(answer_parts) == 1:
                row = compounds.loc[compounds["stim"] == answer_parts[0]].values.flatten().tolist()
                compounds_in_columbus.append({
                    "word_1": row[0],
                    "word_2": row[1],
                    "compound": row[2],
                    "is_plural": row[3] == 1
                })
            elif len(answer_parts) > 1:
                answer = " ".join(answer_parts)
                phrases_in_columbus.append(answer)
                
    return compounds_in_columbus, phrases_in_columbus


def prompt_gpt4(prompt, image_path, model, api_key, max_retries=30, timeout=10, verbose=False):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    image_contents = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "low"
        }
    }]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": image_contents,
            }
        ],
        "max_tokens": 4000,
        'temperature': 0,
    }

    def _send():
        response_ = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response_.json()

    response = None
    try:
        response = _send()
    except:
        retries = 1
        while retries <= max_retries:
            try:
                response = _send()
                break
            except:
                if verbose:
                    print(f"Error (#{retries}). Resending prompt...")
                retries += 1
                time.sleep(timeout)

    return response["choices"][0]["message"]["content"]


def prompt_gemini(prompt, image_path, model, api_key, max_retries=30, timeout=10, verbose=False):
    img = Image.open(image_path)

    def _send():
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(model_name=model)
        response = genai_model.generate_content([prompt, img])
        return response

    response = None
    try:
        response = _send()
    except:
        retries = 1
        while retries <= max_retries:
            try:
                response = _send()
                break
            except:
                if verbose:
                    print(f"Error (#{retries}). Resending prompt...")
                retries += 1
                time.sleep(timeout)
    
    return response.text
