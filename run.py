import string, os, json, sys, shutil, time
import argparse
import random

import src.task_utils as task_utils

import pandas as pd
import asyncio

import logging
# logger name for all files in this project
logger_name = "main_logger"

async def main():
    
    #### LOAD DATA ####    
    task_texts, task_answers = task_utils.load_task(TASK_NAME, f"data/{TASK_NAME}")

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
    results_df = pd.DataFrame({
        "index": range(total_examples),
        "question": task_texts_sample,
        "cot_response": cot_responses,
        "true_answer": [str(a).lower().translate(str.maketrans("", "", string.punctuation)) for a in task_answers_sample],
        "pred_answer": final_answers,
        "all_answers": pred_answers,
        "io": inputs_outputs,
    })
    results_df["correct"] = [int(x) for x in (results_df["true_answer"] == results_df["pred_answer"]).tolist()]

    df_dict_list = results_df.to_dict("records")

    # Calculate costs
    total_prompt_cost = 0  # cot_model.prompts_cost() + answer_model.prompts_cost()
    total_completion_cost = (
        0  # cot_model.completions_cost() + answer_model.completions_cost()
    )
    total_cost = total_prompt_cost + total_completion_cost

    details = {
        "Task": TASK_NAME,
        "Models file": MODELS_FILENAME,
        "Prompt strategy": PROMPT_STRATEGY_CLASS,
        "Prompt strategy kwargs": PROMPT_STRATEGY_KWARGS,
        "Run identifier": RUN_IDENTIFIER,
        "Date": time.strftime("%Y-%m-%d", time.localtime()),
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
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "details.json"), "w") as f:
        json.dump(details, f, indent=4)

    examples_json =  [
            {
                "example_idx": item["index"],
                "question": item["question"],
                "true_answer": item["true_answer"],
                "predicted_answer": item["pred_answer"],
                "correct": item["correct"] == 1,
                "response_pairs": [{"cot_response": c, "answer": a} for c, a in zip(item["cot_response"], item["all_answers"])],
                "response_count": len(item["all_answers"]),
                "queries": [{"query_idx": j, "input": io[0], "output": io[1]} for j, io in enumerate(zip(item["io"][0], item["io"][1]))]
            }
            for item in df_dict_list
        ]

    details["examples"] = examples_json
    
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
        "gaokao-physics",
        "logiqa-en",
        "lsat-ar"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", type=str, default="gsm8k", help="", choices=task_choices
    )
    parser.add_argument(
        "--model_defns_file", type=str, default="PromptWithAnswerExtraction/cot_instruct", help="Config file patha and name inside ./models_defns"
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
        "--overwrite_results",
        action="store_true",
        help="Overwrite results and logs directory if they exists",
    )
    parser.add_argument(
        "--ambiguous_incorrect",
        action="store_true",
        help="Prompts the answer extractor to give an incorrect answer if the response is ambiguous",
    )


    ## Parse arguments
    args = parser.parse_args()
    TASK_NAME = args.task_name
    RUN_IDENTIFIER = args.run_identifier
    NUM_EXAMPLES = 1e9 if args.num_examples == "all" else int(args.num_examples)


    ## Model Definitions Config File Import
    models_dir = "models_defns"
    models_file = args.model_defns_file
    MODELS_FILENAME = models_file.split("/")[-1]
    
    # remove .json from filename if present
    if models_file[-5:] == ".json":
        models_file = models_file[:-5]
        
    models_defns_json = json.load(open(os.path.join(models_dir, f"{models_file}.json"), "r"))
    PROMPT_STRATEGY_CLASS = models_defns_json["class"]
    PROMPT_STRATEGY_KWARGS = (
        models_defns_json["kwargs"] if "kwargs" in models_defns_json else {}
    )
    MODELS_DEFNS = models_defns_json["models"]
    MODELS_DEFNS["answer_extractor"]["prompt"] = task_utils.get_task_answer_prompt(TASK_NAME, ambiguous_incorrect=args.ambiguous_incorrect)


    # Define the complete run dir for results and logs
    RUN_DIR = f"{models_file}" + (
        "" if RUN_IDENTIFIER == "" else f"__{RUN_IDENTIFIER}"
    )

    # Make a directory in results for this run
    results_dir = (
        f"./results/{TASK_NAME.replace('/','_')}"
    )
    RESULTS_DIR = os.path.join(results_dir, RUN_DIR)
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
    logs_dir = f"./.logs/{TASK_NAME.replace('/','_')}"
    log_subdir, log_file = os.path.split(RUN_DIR)
    logs_dir = os.path.join(logs_dir, log_subdir)
    log_file = os.path.join(logs_dir, f"{log_file}.log")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
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
