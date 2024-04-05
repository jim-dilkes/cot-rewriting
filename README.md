# Project README

## Purpose
This project provides a framework for defining sequences of language model prompts ("prompt strategies") intended to complete reasoning tasks. 
OpenAI chat models (accessed through the OpenAI API) and locally run HuggingFace text generation language models are supported. 
Each experiment has a config .json file in the model_defns directory. This file specifies run details such as which prompt strategy, the prompt text, generation parameters, and which model(s) to use.
New prompt strategies may be defined in [src/prompt_strategies.py](https://github.com/jim-dilkes/cot-rewriting/blob/main/src/prompt_strategies.py).

## Command-Line Arguments
- `--task_name`: Specifies the task to execute. Choices include predefined tasks like "gsm8k", "strategyqa", etc.
- `--model_defns_file`: Path and name of the configuration file inside `./models_defns`, defining the model, prompting strategy, prompt text, generation parameters.
- `--run_identifier`: A string to identify this run. Used in filenames to differentiate results.
- `--num_examples`: Number of examples to process. Can be an integer or "all" to process every example.
- `--async_concurr`: Maximum number of concurrent requests to APIs.
- `--overwrite_results`: If set, existing results and logs will be overwritten.
- `--ambiguous_incorrect`: Instructs the answer extractor to provide an incorrect answer if the response is ambiguous.
- `--seed`: Seed for random number generation.
