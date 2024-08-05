import json
import os
import time

import openai

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
                       "Respond with only these options:{}")
    for result in output_file["results"][:1]:
        options = "".join([f"\n{symbol}) {option}" for symbol, option in result["options"].items()])
        if model_type == "llava-1.6-34b":
            output = get_llava_34b_output(result["output"])
        else:
            output = result["output"]
        prompt = prompt_template.format(*[output, options])
        result["raw_output"] = result["output"]
        response = make_safe_prompt(prompt)
        result["output"] = response
        print(json.dumps(result, indent=3))


def get_llava_34b_output(output):
    return output.split("<|im_start|> assistant\n")[1]


extract_model_result("llava-1.6-34b", "../results/analysis/results_v2/prompt_4/llava-1.6-34b_prompt_4.json")
