import string, os, json, sys, shutil
import argparse
import random

import src.data_utils as data_utils
from src.models import GPTModel

import pandas as pd
import asyncio

import logging


""" TASK SPECIFIC CODE """


# Task specific prompt to generate the correct answer string using the CoT solution
def get_task_answer_prompt(task_name):
    if task_name == "gsm8k":
        return "Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols"
    elif task_name[: len("tracking_shuffled_objects")] == "tracking_shuffled_objects":
        return "Respond with the correct completion of the problem statement. Do not include names. Give only the entity that completes the final sentence."
    elif task_name[: len("coinflip")] == "coinflip":
        return "Respond with the final state of the coin. Do not explain your answer only use heads or tails"
    elif task_name == "strategyqa":
        return "Respond with the answer to the question. Do not explain your answer only use yes or no"
    elif task_name == "prontoqa":
        return "Respond with the answer to the question. Do not explain your answer only use true or false"
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")


# Task data load
def load_task(task_name, task_dir):
    if task_name == "gsm8k":
        questions, answers = data_utils.load_gsm8k("test.jsonl", filedir=task_dir)
    elif (
        task_name == "strategyqa"
        or task_name[: len("tracking_shuffled_objects")] == "tracking_shuffled_objects"
    ):
        questions, answers = data_utils.load_bigbench("task.json", filedir=task_dir)
    elif task_name[: len("coinflip")] == "coinflip":
        questions, answers = data_utils.load_coinflip("task.csv", filedir=task_dir)
    elif task_name == "prontoqa":
        questions, answers = data_utils.load_prontoqa(
            "345hop_random_true.json", filedir=task_dir
        )
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")

    return questions, answers


async def main():
    
    #### LOAD DATA ####    
    task_texts, task_answers = load_task(TASK_NAME, f"data/{TASK_NAME}")

    task_texts_sample = (
        random.sample(task_texts, NUM_EXAMPLES)
        if NUM_EXAMPLES < len(task_texts)
        else task_texts
    )
    task_answers_sample = [task_answers[task_texts.index(t)] for t in task_texts_sample]

    total_examples = len(task_texts_sample)
    print(f"Executing on {total_examples} examples")


    ## CREATE PROMPTING STRATEGY OBJECT
    def create_prompt_strategy_class(models_defns):
        class_name = PROMPT_STRATEGY_CLASS
        module = __import__("src.prompt_strategies", fromlist=[class_name])
        class_ = getattr(module, class_name)
        instance = class_(models_defns, TASK_NAME, **PROMPT_STRATEGY_KWARGS)
        return instance

    prompt_strategy = create_prompt_strategy_class(MODELS_DEFNS)
    
    ## GENERATE ASYNC TASKS AND RUN
    tasks = [
        prompt_strategy.proc_example(example, i, SEMAPHORE)
        for i, example in enumerate(task_texts_sample)
    ]
    results = await asyncio.gather(*tasks)
    print()
    
    # Extract answers
    cot_responses, pred_answers, inputs_outputs = zip(*results)
    final_answers = [a[-1] for a in pred_answers]

    #### RECORD RESULTS ####

    ## Record the results of each example as a table
    results_df = pd.DataFrame(
        columns=[
            "index",
            "question",
            "cot_response",
            "pred_answer",
            "true_answer",
            "correct",
        ]
    )
    results_df["index"] = range(total_examples)
    results_df["question"] = task_texts_sample
    results_df["cot_response"] = cot_responses
    true_answers_procd = [
        str(a).lower().translate(str.maketrans("", "", string.punctuation))
        for a in task_answers_sample
    ]
    results_df["true_answer"] = true_answers_procd
    results_df["pred_answer"] = final_answers
    results_df["all_answers"] = pred_answers
    results_df["correct"] = results_df["true_answer"] == results_df["pred_answer"]
    results_df["correct"] = results_df["correct"].astype(int)

    results_df.to_csv(
        os.path.join(RESULTS_DIR, "answers.tsv"), quotechar='"', sep="\t", index=False
    )
    results_df["io"] = inputs_outputs

    ## Record the results to a JSON file
    df_dict_list = results_df.to_dict("records")
    processed_dict_list = []
    for item in df_dict_list:
        new_item = {}

        # Copy the existing fields
        new_item["example_idx"] = item["index"]
        new_item["question"] = item["question"]
        new_item["true_answer"] = item["true_answer"]
        new_item["predicted_answer"] = item["pred_answer"]
        new_item["correct"] = item["correct"] == 1

        # Zip the 'cot_response' and 'answers' fields together
        response_pairs = [
            {"cot_response": c, "answer": a}
            for c, a in zip(item["cot_response"], item["all_answers"])
        ]

        # Store the zipped list and its length
        new_item["response_pairs"] = response_pairs
        new_item["response_count"] = len(response_pairs)

        new_item["queries"] = []
        for j, io in enumerate(zip(item["io"][0], item["io"][1])):
            new_item["queries"].append(
                {"query_idx": j, "input": io[0], "output": io[1]}
            )

        # Add the new item to the list
        processed_dict_list.append(new_item)

    # Calculate costs
    total_prompt_cost = 0  # cot_model.prompts_cost() + answer_model.prompts_cost()
    total_completion_cost = (
        0  # cot_model.completions_cost() + answer_model.completions_cost()
    )
    total_cost = total_prompt_cost + total_completion_cost

    ## Record experiment details to a JSON file
    # Generate the run details data structure with full model query and response history
    details = {
        "Task": TASK_NAME,
        "Prompt strategy": PROMPT_STRATEGY_CLASS,
        "Prompt strategy kwargs": PROMPT_STRATEGY_KWARGS,
        "Number of examples": total_examples,
        "Number of correct": int(results_df["correct"].sum()),
        "Accuracy": float(results_df["correct"].mean()),
        "Models": MODELS_DEFNS,        
        "Cost": {
            "Prompt": {
                "Total": total_prompt_cost,
                "Per token": total_prompt_cost / total_examples,
            },
            "Completion": {
                "Total": total_completion_cost,
                "Per token": total_completion_cost / total_examples,
            },
            "Total": {
                "Total": total_cost,
                "Per token": total_cost / total_examples,
            },
            "Currency": "USD",
        },
        "Examples": processed_dict_list
    }
    
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(details, f, indent=4)

    print("Accuracy:", results_df["correct"].mean())


# python run.py --task_name tracking_shuffled_objects/three_objects --model_name gpt-3.5-turbo --run_identifier test --prompt_type CoT --system_message_type ChatGPT-default --num_examples 20
if __name__ == "__main__":
    random.seed(1)

    task_choices = [
        "gsm8k",
        "tracking_shuffled_objects/three_objects",
        "tracking_shuffled_objects/five_objects",
        "tracking_shuffled_objects/seven_objects",
        "coinflip/four",
        "coinflip/eight",
        "strategyqa",
        "prontoqa",
    ]

    model_apis = {"gpt-3.5-turbo": "openai"}
    model_choices = model_apis.keys()

    prompt_choices = ["None", "CoT", "CoT-WS"]
    system_message_choices = ["ChatGPT-default", "instruct", "None"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", type=str, default="gsm8k", help="", choices=task_choices
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="",
        choices=model_choices,
    )
    parser.add_argument(
        "--prompt_type", type=str, default="CoT", help="", choices=prompt_choices
    )
    parser.add_argument(
        "--run_identifier", type=str, default="", help="Used in file names"
    )
    parser.add_argument(
        "--num_examples", type=str, default="500", help='Either an integer or "all"'
    )
    parser.add_argument(
        "--async_concurr",
        type=int,
        default=7,
        help="Maximum number of concurrent requests to APIs",
    )
    parser.add_argument(
        "--max_rewrites",
        type=int,
        default=0,
        help="The number of times the solution may be rewritten. 0 means no rewriting.",
    )
    parser.add_argument(
        "--model_defns_file", type=str, default="prompt_answer_extraction"
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite results directory if it exists",
    )


    ## Parse arguments
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TASK_NAME = args.task_name
    RUN_IDENTIFIER = args.run_identifier
    PROMPT_TYPE = args.prompt_type
    NUM_EXAMPLES = 1e9 if args.num_examples == "all" else int(args.num_examples)
    MAX_REWRITES = args.max_rewrites


    ## Model Definitions Config File Import
    models_dir = "models_defns"
    models_filename = args.model_defns_file
    # remove .json from filename if present
    if models_filename[-5:] == ".json":
        models_filename = models_filename[:-5]
    models_file = os.path.join(models_dir, f"{models_filename}.json")
    models_defns_json = json.load(open(models_file, "r"))
    PROMPT_STRATEGY_CLASS = models_defns_json["class"]
    PROMPT_STRATEGY_KWARGS = (
        models_defns_json["kwargs"] if "kwargs" in models_defns_json else {}
    )
    MODELS_DEFNS = models_defns_json["models"]
    MODELS_DEFNS["answer_extractor"]["prompt"] = get_task_answer_prompt(TASK_NAME)

    # Define the complete run name
    FILE_NAME = f"{PROMPT_TYPE}__{PROMPT_STRATEGY_CLASS}" + (
        "" if RUN_IDENTIFIER == "" else f"__{RUN_IDENTIFIER}"
    )
    FILE_NAME = FILE_NAME.replace("/", "-")

    # Make a directory in results and .logs for this run
    results_dir = (
        f"./results/{TASK_NAME.replace('/','_')}/{MODEL_NAME.replace('/','_')}"
    )
    RESULTS_DIR = os.path.join(results_dir, FILE_NAME)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    else:
        if args.overwrite_results:
            print(f"Overwriting results directory: {RESULTS_DIR}")
            shutil.rmtree(RESULTS_DIR)
            os.makedirs(RESULTS_DIR)
        else:
            raise ValueError(f"Run directory already exists: {RESULTS_DIR}")


    ## SET UP LOGGING ##

    # Always overwrite the logs directory
    logs_dir = f"./.logs/{TASK_NAME.replace('/','_')}/{MODEL_NAME.replace('/','_')}"
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    log_file = f"{logs_dir}/{FILE_NAME}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # file handler handles all messages

    # Create a stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(
        logging.WARNING
    )  # stdout handler only handles WARNING, ERROR, and CRITICAL messages

    # Create formatters and add them to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


    ## SET UP ASYNCIO ##
    # Specifies the maximum number of concurrent requests to the OpenAI API
    MAX_CONCURRENT_REQUESTS = (
        args.async_concurr
    )  # gpt-3.5 rate limit possibly hit above 8
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


    ## RUN THE EXPERIMENT ##
    asyncio.run(main())
