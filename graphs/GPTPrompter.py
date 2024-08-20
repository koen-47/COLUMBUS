import json
import os
import time

import inflect
import requests
import base64

import openai

openai.key = os.getenv("OPENAI_API_KEY")

p = inflect.engine()


class GPTPrompter:
    """
    Class to prompt GPT-4o (mini) for the operators defined in BeliefGraphGenerator.
    """
    def __init__(self, puzzle, n_examples, model="gpt-4o-mini", verbose=False):
        self._images = [puzzle]
        self._model = model
        self._verbose = verbose
        self._n_examples = n_examples

        # This file contains all the prompts we use in the experiments.
        with open(f"{os.path.dirname(__file__)}/prompts.json", "r") as file:
            self._prompts = json.load(file)

        # We also evaluated few-shot prompting. However, this did not increase performance. In the results of Table 2
        # of the paper, all results are with zero-shot prompting (i.e., n_examples == 0).
        if self._n_examples > 0:
            examples = self._prompts["few_shot_examples"][str(self._n_examples)]
            examples_desc = "".join(f"\n- Image {i+1}: {example["description"]}" for i, example in enumerate(examples))
            preprompt_template = self._prompts["prompt_templates"]["few_shot"]["preprompt"]
            if self._n_examples == 1:
                preprompt_template = preprompt_template["1"]
                self._preprompt = preprompt_template.format(*[examples_desc])
            else:
                preprompt_template = preprompt_template["2+"]
                cardinal_num = p.number_to_words(self._n_examples)
                self._preprompt = preprompt_template.format(*[cardinal_num, examples_desc])
            examples_images = [f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{example["image"]}"
                               for example in examples]
            self._images = examples_images + [puzzle]
        else:
            self._preprompt = self._prompts["prompt_templates"]["zero_shot"]["preprompt"]

    def send_prompt(self, text, image_paths=None, max_retries=10, timeout=10, max_tokens=300):
        """
        Sends a prompt to the GPT-4o (mini) API, with the specified text and image (if given).

        :param text: text prompt to send to the API.
        :param image_paths: file paths to the images to send to the API (if given).
        :param max_retries: maximum number of times to retry the API call if there is an error.
        :param timeout: number of seconds to wait between each retry.
        :param max_tokens: maximum number of tokens that each API call should return.
        :return: JSON object of the returned response from the GPT-4o API.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }

                    ]
                }
            ],
            "max_tokens": max_tokens,
            "logprobs": True
        }

        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    image = base64.b64encode(image_file.read()).decode("utf-8")

                payload["messages"][0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        }
                    }
                )

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
                    if self._verbose:
                        print(f"Error (#{retries}). Resending prompt...")
                    retries += 1
                    time.sleep(timeout)

        if "error" in response:
            retries = 1
            while retries <= max_retries and "error" in response:
                if self._verbose:
                    print(f"Error (#{retries}). Resending prompt...")
                retries += 1
                time.sleep(timeout)
                response = _send()
            return response
        else:
            return response

    def score_statement(self, statement):
        """
        Call to the GPT-4o (mini) API to score a statement (see BeliefGraphGenerator for more information).
        :param statement: string of statement to score.
        :return: assigned truth value and confidence in this truth value.
        """
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["score_statement"]
            text = prompt_template.format(*[self._preprompt, statement])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["score_statement"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(
                *[self._preprompt, ordinal_num, len(self._images), len(self._images), statement])

        response = self.send_prompt(text, image_paths=self._images, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = content["logprob"]
        value = content["token"]
        prob = 10 ** float(prob)
        return value, prob

    def generate_premise_from_hypothesis(self, hypothesis):
        """
        Call to the GPT-4o (mini) API to generate premises for a given hypothesis (see BeliefGraphGenerator for more
        information).

        :param hypothesis: string of hypothesis to generate premises for.
        :return: a list of generated premises (strings).
        """
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["generate_premises"]
            text = prompt_template.format(*[self._preprompt, hypothesis])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["generate_premises"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(
                *[self._preprompt, ordinal_num, len(self._images), hypothesis, len(self._images)])

        response = self.send_prompt(text, image_paths=self._images)
        premises = response["choices"][0]["message"]["content"]
        return premises

    def generate_negated_statement(self, statement):
        """
        Call to the GPT-4o (mini) API to negate a statement (see BeliefGraphGenerator for more information).

        :param statement: string of statement to negate.
        :return: a negated statement of the specified statement.
        """
        prompt_template = self._prompts["prompt_templates"]["generate_negation"]
        text = prompt_template.format(*[statement])
        response = self.send_prompt(text)
        negated = response["choices"][0]["message"]["content"]
        return negated

    def score_rule(self, premises, hypothesis):
        """
        Call to the GPT-4o (mini) API to score a rule (see BeliefGraphGenerator for more information).

        :param premises: list of premises (strings).
        :param hypothesis: hypothesis (string).
        :return: confidence (probability) that the premises imply the hypothesis.
        """
        premises_text = "".join([f"\n- {premise}" for premise in premises if premise != ""])
        text = ""
        if self._n_examples == 0:
            prompt_template = self._prompts["prompt_templates"]["zero_shot"]["score_rule"]
            text = prompt_template.format(*[self._preprompt, premises_text, hypothesis])
        elif self._n_examples > 0:
            prompt_template = self._prompts["prompt_templates"]["few_shot"]["score_rule"]
            ordinal_num = p.number_to_words(p.ordinal(len(self._images)))
            text = prompt_template.format(*[self._preprompt, ordinal_num, len(self._images), len(self._images),
                                            premises_text, hypothesis])

        response = self.send_prompt(text, image_paths=self._images, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = content["logprob"]
        prob = 10 ** float(prob)
        return prob
