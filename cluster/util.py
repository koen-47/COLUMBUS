import base64
import glob
import json
import os
import copy
import time

import networkx as nx
import pandas as pd
import requests

from parsers.patterns.Rule import Rule
import google.generativeai as genai
import PIL.Image
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_node_attributes(graph):
    attrs = {attr: nx.get_node_attributes(graph, attr) for attr in ["text", "is_plural"] + Rule.ALL_RULES}
    node_attrs = {}
    for attr, nodes in attrs.items():
        for node, attr_val in nodes.items():
            if node not in node_attrs:
                node_attrs[node] = {attr: attr_val}
            node_attrs[node][attr] = attr_val
    return node_attrs


def get_edges_from_node(graph, node_id):
    in_edges, out_edges = {}, {}
    for edge in graph.in_edges(node_id, keys=True):
        in_edges[edge] = nx.get_edge_attributes(graph, "rule")[edge]
    for edge in graph.out_edges(node_id, keys=True):
        out_edges[edge] = nx.get_edge_attributes(graph, "rule")[edge]
    return [in_edges, out_edges]


def get_edge_information(graph):
    node_attrs = get_node_attributes(graph)
    edge_attrs = nx.get_edge_attributes(graph, "rule")
    edge_info = {}
    for edge in graph.edges:
        rule = edge_attrs[edge]
        source = node_attrs[edge[0]]
        target = node_attrs[edge[1]]
        edge_info[edge] = (source, rule, target)
    return edge_info


def remove_duplicate_graphs(graphs):
    unique_graphs = []
    for graph in graphs:
        is_duplicate = False
        for unique_graph in unique_graphs:
            if nx.utils.graphs_equal(graph, unique_graph):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_graphs.append(graph)
    return unique_graphs


def count_relational_rules(phrase):
    relational_keywords = [x for xs in Rule.get_all_rules()["relational"].values() for x in xs]
    return sum([1 for word in phrase.split() if word in relational_keywords])


def get_answer_graph_pairs(combine=False):
    from parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
    from parsers.PhraseRebusGraphParser import PhraseRebusGraphParser

    phrases = [os.path.basename(file).split(".")[0]
               for file in glob.glob(f"{os.path.dirname(__file__)}/data/images/*")]
    ladec = pd.read_csv(f"{os.path.dirname(__file__)}/data/misc/ladec_raw_small.csv")
    custom_compounds = pd.read_csv(f"{os.path.dirname(__file__)}/data/misc/custom_compounds.csv")

    compound_parser = CompoundRebusGraphParser()
    phrase_parser = PhraseRebusGraphParser()
    phrase_to_graph = {}
    compound_to_graph = {}
    for phrase in phrases:
        orig_phrase = phrase
        if phrase.endswith("_icon") or phrase.endswith("_non-icon"):
            phrase = "_".join(phrase.split("_")[:-1])
        parts = phrase.split("_")
        index = 0
        if (parts[0] in ladec["stim"].tolist() and len(parts) == 2) or len(parts) == 1:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            row = ladec.loc[ladec["stim"] == phrase_].values.flatten().tolist()
            if len(row) == 0:
                row = custom_compounds.loc[custom_compounds["stim"] == phrase_].values.flatten().tolist()
            c1, c2, is_plural = row[0], row[1], bool(row[2])
            graphs = compound_parser.parse(c1, c2, is_plural)
            phrase = "_".join(orig_phrase.split())
            if orig_phrase.endswith("non-icon"):
                compound_to_graph[phrase] = remove_icons_from_graph(graphs[index])
            else:
                compound_to_graph[phrase] = graphs[index]
        else:
            if parts[-1].isnumeric():
                index = int(parts[-1]) - 1
                parts = parts[:-1]
            phrase_ = " ".join(parts)
            graphs = phrase_parser.parse(phrase_)
            phrase = "_".join(orig_phrase.split())
            if orig_phrase.endswith("non-icon"):
                phrase_to_graph[phrase] = remove_icons_from_graph(graphs[index])
            else:
                phrase_to_graph[phrase] = graphs[index]
    if combine:
        graphs = {}
        graphs.update(phrase_to_graph)
        graphs.update(compound_to_graph)
        return graphs

    return phrase_to_graph, compound_to_graph


def remove_icons_from_graph(graph):
    graph_no_icon = copy.deepcopy(graph)
    graph_no_icon_node_attrs = get_node_attributes(graph_no_icon)
    for attr in graph_no_icon_node_attrs.values():
        if "icon" in attr:
            attr["text"] = list(attr["icon"].keys())[0].upper()
            del attr["icon"]

    for node in graph_no_icon.nodes:
        graph_no_icon.nodes[node].clear()
    nx.set_node_attributes(graph_no_icon, graph_no_icon_node_attrs)
    return graph_no_icon


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt4v(image_path, system_prompt):
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
        "Authorization": f"Bearer {os.getenv("OPENAI_API_KEY")}"
    }

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": image_contents,
            }
        ],
        "max_tokens": 4000,
        'temperature': 0,
    }

    start = time.time()
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    end = time.time()
    if end - start < 4:
        time.sleep(4 - (end - start))

    return response.json()["choices"][0]["message"]["content"]


def query_model(puzzle, index, args):
    prompt_template_dict = {
        "1": "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)?\n(A) {} (B) {} (C) {} (D) {}",
        "2": "You are given a rebus puzzle. It consists of text or icons that is used to convey a word or phrase. It needs to be solved through creative thinking. Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)?\n(A) {} (B) {} (C) {} (D) {}",
        "3": "You are given an image of a rebus puzzle. It consists of text or icons that is used to convey a word or phrase. It needs to be solved through creative thinking. You are also given a description of the graph representation of the puzzle. The nodes are elements that contain text or icons, which are then manipulated through the attributes of their node. The description is as follows:\n{}\nWhich word/phrase is conveyed in this image and description from the following options (either A, B, C, or D)?\n(A) {} (B) {} (C) {} (D) {}",
        "4": "You are given an image of a rebus puzzle. It consists of text or icons that is used to convey a word or phrase. It needs to be solved through creative thinking. You are also given a description of the graph representation of the puzzle. The nodes are elements that contain text or icons, which are then manipulated through the attributes of their node. The edges define spatial relationships between these elements. The description is as follows:\n{}\nWhich word/phrase is conveyed in this image and description from the following options (either A, B, C, or D)?\n(A) {} (B) {} (C) {} (D) {}"
    }

    try:
        image_path = puzzle["image"]
        options = puzzle["options"]
        if index % 50 == 0:
            print(f"Processing {index}th image")
        sub_folder_path = os.path.join(args.folder_path, f"prompt_{args.prompt_type}")
        json_file_path = os.path.join(sub_folder_path, f'{index}.json')
        if os.path.exists(json_file_path):
            print(f"Skipping sub_folder {index} as it already exists")
        else:
            prompt_template = prompt_template_dict[args.prompt_type]
            if args.prompt_type == "3":
                prompt_options = [puzzle["metadata"]["nodes"]] + list(options.values())
            elif args.prompt_type == "4":
                prompt_options = [puzzle["metadata"]["nodes_and_edges"]] + list(options.values())
            else:
                prompt_options = list(options.values())
            prompt = prompt_template.format(*prompt_options)
            gpt_response = Gemini(image_path, prompt)
            with open(json_file_path, 'w') as f:
                json.dump({"gemini_pro_response": gpt_response, 'prompt': prompt, 'answer': puzzle['correct']}, f,
                          indent=4)
    except Exception as e:
        time.sleep(20)
        print(e)


def Gemini(image_path, system_prompt):
    img = PIL.Image.open(image_path)

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content([system_prompt, img])
    return response.text


def query_model_simple_json(data, prompt, index):
    try:
        if index % 10 == 0:
            print(f"Processing {index}th image")
        image_path = data['image']
        response = gpt4v(image_path, prompt)
        data['response'] = response
        data['prompt'] = prompt
        json_file_path = os.path.join('./results/GPT/json_result', f'{index}.json')
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(e)
