import json
import os
import time

import openai
from tqdm import tqdm

openai.key = os.getenv("OPENAI_API_KEY")


def make_safe_prompt(prompt, max_retries=30, timeout=30):
    def make_prompt(prompt, temperature=0.7):
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": prompt}
            ],
            temperature=temperature
        )

        return completion.choices[0].message.content

    try:
        return make_prompt(prompt)
    except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout,
            openai.error.ServiceUnavailableError):
        retries = 1
        while retries <= max_retries:
            print(f"\n{retries}. Error. Restarting. Prompt = {prompt}")
            try:
                return make_prompt(prompt)
            except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout):
                time.sleep(timeout)
                retries += 1


def extract_model_result(model_type, output_file_path):
    with open(output_file_path, "r") as file:
        output_file = json.load(file)

    prompt_template = ("I have the following text:\n\"{}\"\n\nPlease map this text to any of the following options. "
                       "Remember that the output can also refer to any of the symbols as well (either A, B, C, D)."
                       "Respond with 'None' if none of the output doesn't sufficiently match any of the options."
                       "Respond with only these options:{}")
    for result in tqdm(output_file["results"], desc=f"Extracting results (model: {model_type})"):
        options = "".join([f"\n{symbol}) {option}" for symbol, option in result["options"].items()])
        if model_type == "llava-1.6-34b":
            output = get_llava_34b_output(result["output"])
        elif model_type == "mistral-7b":
            output = get_mistral_output(result["output"])
        else:
            output = result["output"]
        prompt = prompt_template.format(*[output, options])
        result["raw_output"] = result["output"]
        response = make_safe_prompt(prompt)
        result["output"] = response
    return output_file


def get_llava_34b_output(output):
    return output.split("<|im_start|> assistant\n")[1]


def get_mistral_output(output):
    return output.split("[/INST]")[1].strip()


def extract_qwenvl_results():
    for num_prompt in range(1, 5):
        qwenvl_file = f"../results/analysis/results_v3/prompt_{num_prompt}/qwenvl_prompt_{num_prompt}.json"
        results = extract_model_result("qwenvl", qwenvl_file)
        with open(qwenvl_file, "w") as file:
            json.dump(results, file, indent=3)


def extract_mistral_results():
    for num_prompt in range(3, 5):
        mistral_file = f"../results/analysis/results_v3/mistral-7b_prompt_{num_prompt}.json"
        results = extract_model_result("mistral-7b", mistral_file)
        with open(mistral_file, "w") as file:
            json.dump(results, file, indent=3)


extract_mistral_results()
