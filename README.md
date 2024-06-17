# A Benchmark to Evaluate the Lateral Thinking Skills of Visual Question Answering Models through Rebus Puzzles
This repository presents a benchmark of rebus puzzles designed to challenge the lateral thinking skills of visual question answering (VQA) models. 

This repository has been submitted in conjunction with a thesis for the VU Master of Science degree in Artificial Intelligence. The draft version of the thesis can be found [here](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/thesis_draft.pdf).

## Data Selection and Collection

---

The following files consist of the raw data scraped, downloaded or manually collected (custom) to be used as input for our puzzle generation pipeline:
- Compound words: [source](https://era.library.ualberta.ca/items/dc3b9033-14d0-48d7-b6fa-6398a30e61e4) + [custom](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/saved/custom_compounds.csv) 
- Idioms/phrases: [source](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/saved/idioms_raw.json) + [custom](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/saved/custom_phrases.json)
- Icons: [source](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/saved/icons_v2.json)
- Homophones: [source](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/saved/homophones_v2.json)


##  Puzzle Generation

All files relating to puzzle generation can be found under [graphs](https://github.com/Koen-Kraaijveld/rebus-puzzles/tree/main/graphs). The main ones are as follows:
- [Compound graph parser](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/graphs/parsers/CompoundRebusGraphParser.py): parses a compound word into its graph representation.
- [Phrase graph parser](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/graphs/parsers/PhraseRebusGraphParser.py): parses an idiom/phrase into its graph representation. 
- [Image generation](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/graphs/RebusImageConverterV2.py): generates a rebus puzzle from its graph representation.
- [Distractor generation](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/misc/phrase_similarity.py): generates three distractors for each question given the correct answer.

## Results

All files used to prompt the models in our experiments can be found under the [DAS-6](https://github.com/Koen-Kraaijveld/rebus-puzzles/tree/main/das6) folder. This is a duplicated, smaller version of this repository to use on the DAS-6 cluster. The file used to prompt each model is as follows:

- [CLIP](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/CLIPExperiment.py) (baseline)
- [BLIP-2](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/BLIP2Experiment.py) (OPT 2.7b, OPT 6.7b, Flan-T5-XXL-11b)
- [InstructBLIP](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/InstructBLIPExperiment.py)
- [Fuyu](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/FuyuExperiment.py)
- [QwenVL](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/QwenVLModel.py)
- [CogVLM](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/CogVLMModel.py)
- [Llava](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/LlavaExperiment.py) (1.5-13b, 1.6-34b)
- [Mistral](https://github.com/Koen-Kraaijveld/rebus-puzzles/blob/main/das6/models/MistralExperiment.py)