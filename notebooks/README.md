# Notebooks for COLUMBUS

There are three notebooks that can be used to recreate results from COLUMBUS. 
These are as follows:

- [Puzzle generation](./generate_puzzles.ipynb): generate puzzles directly from COLUMBUS or generate your own using our pipeline.
- [Running open-source models](./run_open_source_models.ipynb): run three open-source models (Fuyu-8b, BLIP-2 Flan-T5-XXL, and Mistral-7b) on COLUMBUS. **NOTE**: this does not involve post-processing the results.
- [Running closed-source models](./run_closed_source_models.ipynb): run four closed-source models (GPT-4o, GPT-4o-mini, Gemini 1.5 Pro + Flash) on COLUMBUS, including their forward and backward chaining variants.  **NOTE**: this does not involve post-processing the results.

All model downloads and results will be stored under the [model downloads](./model_downloads) and [model results](./model_results) folders, respectively.

**NOTE:** make sure you have installed all required packages for the conda environment beforehand (see the [installation](../README.md) instructions) and that the notebooks have the environment activated while running.