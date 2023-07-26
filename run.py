import string, os, json, sys
import argparse
import random

import src.data_utils as data_utils
from src.models import GPTModel
from src.openai_utils import OpenAIChatMessages

import pandas as pd
import asyncio

import logging


""" TASK SPECIFIC CODE """
# Task specific prompt to generate the correct answer string using the CoT solution
def get_task_answer_prompt(task_name):
    if task_name == 'gsm8k':
        return "Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols"
    elif task_name[:len('tracking_shuffled_objects')] == 'tracking_shuffled_objects':
        return "Respond with the correct completion of the problem statement. Do not explain your answer. Do not repeat the problem statement give only the entity. Examples:\nblue ball\nJason\nThe Great Gatsby\ngoalkeeper"
    elif task_name[:len('coinflip')] == 'coinflip':
        return "Respond with the final state of the coin. Do not explain your answer only use heads or tails"
    elif task_name == 'strategyqa':
        return "Respond with the answer to the question. Do not explain your answer only use yes or no"
    elif task_name == 'prontoqa':
        return "Respond with the answer to the question. Do not explain your answer only use true or false"
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")

def append_task_answer_prompt(task_name, messages_obj:OpenAIChatMessages):
    """ In place udpate of messages_obj with the task specific answer prompt """
    return messages_obj.append(role='user', content=get_task_answer_prompt(task_name))   
    
# Task data load
def load_task(task_name, task_dir):
    if task_name == 'gsm8k':
        questions, answers = data_utils.load_gsm8k('test.jsonl', filedir=task_dir)
    elif task_name == 'strategyqa' or task_name[:len('tracking_shuffled_objects')] == 'tracking_shuffled_objects':
        questions, answers = data_utils.load_bigbench('task.json', filedir=task_dir)
    elif task_name[:len('coinflip')] == 'coinflip':
        questions, answers = data_utils.load_coinflip('task.csv', filedir=task_dir)
    elif task_name == 'prontoqa':
        questions, answers = data_utils.load_prontoqa('345hop_random_true.json', filedir=task_dir)
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")
    
    return questions, answers

""" Async func to process a single example """
# Awaits calls to OpenAI API in generate_async
async def process_example(i, example_text, cot_model, answer_model, task_name):
    # Acquire the semaphore before processing an example
    async with SEMAPHORE:
        ## Generate CoT Solution
        cot_response = await cot_model.generate_async(example_text)
        answer_response = await answer_model.generate_async(f"Problem Statement: {example_text}\nProposed Solution: {cot_response}")
        answer_response = answer_response.strip()

        print(f"\rDone example {i}", end='')
        sys.stdout.flush()
        
        return cot_response, answer_response


async def main():
    ## Load the task
    task_texts, task_answers = load_task(TASK_NAME, f'data/{TASK_NAME}')

    task_texts_sample = random.sample(task_texts, NUM_EXAMPLES) if NUM_EXAMPLES < len(task_texts) else task_texts
    task_answers_sample = [task_answers[task_texts.index(t)] for t in task_texts_sample]

    total_examples = len(task_texts_sample)
    print(f"Executing on {total_examples} examples")

    ## Define the models
    cot_model_name = MODEL_NAME
    cot_temp = 0
    cot_max_tokens = 256
    cot_model = GPTModel(model_name=cot_model_name, 
                         system_message=PROMPT_SYSTEM_MESSAGE, 
                         prompt_message=PROMPT_TEXT,
                         temperature=cot_temp,
                         max_tokens=cot_max_tokens)

    answer_model_name = MODEL_NAME
    answer_temp = 0
    answer_max_tokens = 40
    answer_model = GPTModel(model_name=answer_model_name,
                            system_message=ANSWER_SYSTEM_MESSAGE,
                            prompt_message=get_task_answer_prompt(TASK_NAME),
                            temperature=answer_temp,
                            max_tokens=answer_max_tokens)

    # Create an asyncio task for each example in dataset
    tasks = [process_example(i, example, cot_model, answer_model, TASK_NAME) for i, example in enumerate(task_texts_sample)]

    # Run all the tasks in parallel
    results = await asyncio.gather(*tasks)
    print()
    cot_responses, pred_answers = zip(*results)

    ## Format and record the results            
    results_df = pd.DataFrame(columns=['index','question', 'cot_response', 'pred_answer', 'true_answer', 'correct'])
    results_df['index'] = range(total_examples)
    results_df['question'] = task_texts_sample
    results_df['cot_response'] = cot_responses
    true_answers_procd = [str(a).lower().translate(str.maketrans("", "", string.punctuation)) for a in task_answers_sample]
    pred_answers_procd = [str(a).lower().translate(str.maketrans("", "", string.punctuation)) for a in pred_answers]
    results_df['true_answer'] = true_answers_procd
    results_df['pred_answer'] = pred_answers_procd
    results_df['correct'] = results_df['true_answer'] == results_df['pred_answer']
    results_df['correct'] = results_df['correct'].astype(int)


    results_df.to_csv(os.path.join(RESULTS_DIR, "answers.tsv"), quotechar='"',sep='\t',index=False)

    # Generate the run details data structure
    total_prompt_cost = cot_model.prompts_cost() + answer_model.prompts_cost()
    total_completion_cost = cot_model.completions_cost() + answer_model.completions_cost()
    total_cost = total_prompt_cost + total_completion_cost
    details = {
        "Task": TASK_NAME,
        "Model": MODEL_NAME,
        "Prompt type": PROMPT_TYPE,
        "Prompt": PROMPT_TEXT,
        "System message type": SYSTEM_MESSAGE_TYPE,
        "Prompt system message": PROMPT_SYSTEM_MESSAGE,
        "Answer system message": ANSWER_SYSTEM_MESSAGE,
        "Number of examples": total_examples,
        "Number of correct": int(results_df['correct'].sum()),
        "Accuracy": float(results_df['correct'].mean()),
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
            "Currency": "USD"
        }
    }

    # Write the run details to a JSON file
    with open(os.path.join(RESULTS_DIR, "details.json"), "w") as file:
        json.dump(details, file, indent=4)

    print("Accuracy:", results_df['correct'].mean())
    print()



# python run.py --task_name tracking_shuffled_objects/three_objects --model_name gpt-3.5-turbo --run_identifier test --prompt_type CoT --system_message_type ChatGPT-default --num_examples 20
if __name__ == '__main__':
    random.seed(1)
    
    task_choices = ['gsm8k',
                   'tracking_shuffled_objects/three_objects',
                   'tracking_shuffled_objects/five_objects',
                   'tracking_shuffled_objects/seven_objects',
                   'coinflip/four',
                   'coinflip/eight',
                   'strategyqa',
                   'prontoqa']
    model_choices = ['gpt-3.5-turbo']
    prompt_choices = ['None',
                      'CoT',
                      'CoT-WS']
    system_message_choices = ['ChatGPT-default',
                              'instruct',
                              'instruct-list']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='gsm8k', help='', choices=task_choices)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='', choices=model_choices)
    parser.add_argument('--prompt_type', type=str, default='CoT', help='', choices=prompt_choices)
    parser.add_argument('--system_message_type', type=str, default='ChatGPT-default', help='', choices=system_message_choices)
    parser.add_argument('--run_identifier', type=str, default='', help='Used in file names')
    parser.add_argument('--num_examples', type=str, default='500', help='Either an integer or "all"')
    parser.add_argument('--async_concurr', type=int, default=10, help='Maximum number of concurrent requests to APIs')
    
    
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TASK_NAME = args.task_name
    RUN_IDENTIFIER = args.run_identifier
    PROMPT_TYPE = args.prompt_type
    SYSTEM_MESSAGE_TYPE = args.system_message_type
    NUM_EXAMPLES = 1E9 if args.num_examples == 'all' else int(args.num_examples)
    
    # Define the run name
    FILE_NAME = f'{PROMPT_TYPE}__{SYSTEM_MESSAGE_TYPE}' + ('' if RUN_IDENTIFIER == '' else f'__{RUN_IDENTIFIER}')
    FILE_NAME = FILE_NAME.replace('/', '-')
    
    # Make a directory in results for this run
    results_dir = f"./results/{TASK_NAME.replace('/','_')}/{MODEL_NAME.replace('/','_')}"
    RESULTS_DIR = os.path.join(results_dir, FILE_NAME)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    else:
        raise ValueError(f"Run directory already exists: {RESULTS_DIR}")
    
    # Define prompt and system message text
    if PROMPT_TYPE == 'None':
        PROMPT_TEXT = ""
    elif PROMPT_TYPE == 'CoT':
        PROMPT_TEXT = "Let's think about this step by step."
    elif PROMPT_TYPE == 'CoT-WS':
        PROMPT_TEXT = "Let's think about this step by step and describe how the state of the world, any values, or our knoweledge change at each step."
    
    if SYSTEM_MESSAGE_TYPE == 'ChatGPT-default':
        PROMPT_SYSTEM_MESSAGE = "You are a helpful assistant."
    elif SYSTEM_MESSAGE_TYPE == 'instruct':
        PROMPT_SYSTEM_MESSAGE = "You are an instruction following, problem solving assistant."
    elif SYSTEM_MESSAGE_TYPE == 'instruct-list':
        PROMPT_SYSTEM_MESSAGE = "You are an instruction following, problem solving assistant. Respond with lists \n1.\n2.\n3.\n..."
    
    ANSWER_SYSTEM_MESSAGE = "You are an instruction following, problem solving agent. Only respond with the exact answer. Do not explain your answers. Do not respond with sentences. Give exactly one answer."
    
    # Specifies the maximum number of concurrent requests to the OpenAI API
    MAX_CONCURRENT_REQUESTS = args.async_concurr # gpt-3.5 rate limit possibly hit above 10
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    
     # Make a directory in logs for this run
    logs_dir = f"./.logs/{TASK_NAME.replace('/','_')}/{MODEL_NAME.replace('/','_')}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure logging
    logging.basicConfig(filename=f'{logs_dir}/{FILE_NAME}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    asyncio.run(main())

    