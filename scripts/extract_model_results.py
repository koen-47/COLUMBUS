"""
Scripts used to map a model's output to one of the options in the question. This was used for the larger models
that consistently gave long answers.
"""

import json
import os
import time

import openai
from tqdm import tqdm

openai.key = os.getenv("OPENAI_API_KEY")


def make_safe_prompt(prompt, max_retries=30, timeout=30):
    """
    Sends a prompt to the GPT-4o API, with the specified prompt (if given).
    :param prompt: text prompt to send to the API.
    :param max_retries: maximum number of times to retry the API call if there is an error.
    :param timeout: number of seconds to wait between each retry.
    :return: text response from GPT-4o.
    """
    def make_prompt(prompt, temperature=0.7):
        """
        Helper function to send a prompt to the GPT-4o API.
        :param prompt: text prompt to send to the API.
        :param temperature: temperature value used.
        :return: text response from GPT-4o.
        """
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
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
    """
    Extracts a result from a model's output using the GPT-4o API.

    :param model_type: model used for the original output that will be mapped to one of the options.
    :param output_file_path: file path of where to store extracted output.
    :return: results file with extract output for each puzzle.
    """
    with open(output_file_path, "r") as file:
        output_file = json.load(file)

    # Prompt used to extract the output
    prompt_template = ("I have the following text:\n\"{}\"\n\nPlease map this text to any of the following options. "
                       "Remember that the output can also refer to any of the symbols as well (either A, B, C, D)."
                       "Respond with 'None' if none of the output doesn't sufficiently match any of the options."
                       "Respond with only these options:{}")

    for result in tqdm(output_file["results"][:1], desc=f"Extracting results (model: {model_type})"):
        options = "".join([f"\n{symbol}) {option}" for symbol, option in result["options"].items()])

        # Different output extraction method used depending on the model
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
    """
    Get output for Llava-34b.
    :param output: raw output of Llava-34b.
    :return: original output of Llava-34b (before being extracted with GPT-4o)
    """
    return output.split("<|im_start|> assistant\n")[1]


def get_mistral_output(output):
    """
    Get output for Mistral.
    :param output: raw output of Mistral.
    :return: original output of Mistral (before being extracted with GPT-4o)
    """
    return output.split("[/INST]")[1].strip()


def extract_llava_34b_results():
    """
    Extract results for Llava-34b.
    """
    llava_file = f"../results/analysis/results/prompt_1/llava-1.6-34b_prompt_1.json"
    results = extract_model_result("llava-1.6-34b", llava_file)
    with open(llava_file, "w") as file:
        json.dump(results, file, indent=3)


def extract_qwenvl_results():
    """
    Extract results for QwenVL.
    """
    for num_prompt in range(1, 5):
        qwenvl_file = f"../results/analysis/results/prompt_{num_prompt}/qwenvl_prompt_{num_prompt}.json"
        results = extract_model_result("qwenvl", qwenvl_file)
        with open(qwenvl_file, "w") as file:
            json.dump(results, file, indent=3)


def extract_mistral_results():
    """
    Extract results for Mistral.
    """
    for num_prompt in range(3, 5):
        mistral_file = f"../results/analysis/results/mistral-7b_prompt_{num_prompt}.json"
        results = extract_model_result("mistral-7b", mistral_file)
        with open(mistral_file, "w") as file:
            json.dump(results, file, indent=3)
