import string, os, json, sys
import argparse
import random

import src.data_utils as data_utils
from src.models import GPTModel, HFInferenceModel
from src.openai_utils import OpenAIChatMessages

import pandas as pd
import asyncio

import logging


""" TASK SPECIFIC CODE """
# Task specific prompt to generate the correct answer string using the CoT solution
def append_task_answer_prompt(task_name, messages_obj:OpenAIChatMessages):
    """ In place udpate of messages_obj with the task specific answer prompt """
    if task_name == 'gsm8k':
        messages_obj.append(role='user', content="Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols")
    elif task_name[:len('tracking_shuffled_objects')] == 'tracking_shuffled_objects':
        # messages_obj.append(role='user', content="Respond with the correct completion of the problem statement. Do not explain your answer. Do not repeat the problem statement give only the entity. Examples:\nblue ball\nJason\nThe Great Gatsby\ngoalkeeper")
        messages_obj.append(role='user', content="Respond with the correct completion of the problem statement. Do not include names. Give only the entity that completes the final sentence.")
    elif task_name[:len('coinflip')] == 'coinflip':
        messages_obj.append(role='user', content="Respond with the final state of the coin. Do not explain your answer only use heads or tails")
    elif task_name == 'strategyqa':
        messages_obj.append(role='user', content="Respond with the answer to the question. Do not explain your answer only use yes or no")
    elif task_name == 'prontoqa':
        messages_obj.append(role='user', content="Respond with the answer to the question. Do not explain your answer only use true or false")
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")      
    
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

def clean_answer(answer):
    return str(answer).lower().translate(str.maketrans("", "", string.punctuation)).strip()

async def generate_and_record(model, messages_obj, inputs_lst, outputs_lst, **kwargs):
    """ Generate a response from the model and record the inputs and outputs """
    response = await model.generate_async(messages_obj, **kwargs)
    inputs_lst.append(messages_obj.get())
    outputs_lst.append(response)
    return response

async def process_example(i, example_text, cot_model, answer_model, task_name):
    # Acquire the semaphore before processing an example
    async with SEMAPHORE:
        all_inputs = []
        all_outputs = []
        messages_obj = OpenAIChatMessages()

        ## Generate CoT Solution
        messages_obj.append(role='system', content=PROMPT_SYSTEM_MESSAGE)
        messages_obj.append(role='user', content=f"{example_text}")
        messages_obj.append(role='user', content=PROMPT_TEXT)
        cot_responses = [await generate_and_record(cot_model, messages_obj, all_inputs, all_outputs, n_sample=1)]
        
        answer_responses = []
        ## Extract the answer from the final CoT solution
        messages_obj.reset()
        messages_obj.append(role='system', content=ANSWER_SYSTEM_MESSAGE)
        messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
        messages_obj.append(role='user', content=f"Proposed Solution:\n{cot_responses[-1]}")

        append_task_answer_prompt(task_name, messages_obj)
        answer = await generate_and_record(answer_model, messages_obj, all_inputs, all_outputs, n_sample=1)
        answer_responses.append(clean_answer(answer))
        
        j=0
        while j < MAX_REWRITES:
            ## Ask whether to rewrite
            # Stick together the CoT responses, writing the response number before each one
            # cot_message = '\n'.join([f"Proposed solution {i+1}: {r}" for i, r in enumerate(cot_responses)])
            cot_answer_message = '\n'.join([f"Proposed solution {i+1}: {r[0]}\nAnswer {i+1}: {r[1]}" for i, r in enumerate(zip(cot_responses, answer_responses))])
            
            ## ASK WHETHER TO REWRITE
            messages_obj.reset()
            messages_obj.append(role='system', content=PROMPT_SYSTEM_MESSAGE)
            messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
            messages_obj.append(role='user', content=cot_answer_message)
            messages_obj.append(role='user', content=ASK_REWRITE_MESSAGE)

            rewrite_response = await generate_and_record(answer_model, messages_obj, all_inputs, all_outputs, n_sample=1)
            
            ## EXTRACT REWRITE RESPONSE
            if EXTRACT_REWRITE_RESPONSE:
                messages_obj.reset()
                messages_obj.append(role='system', content=ANSWER_SYSTEM_MESSAGE)
                messages_obj.append(role='user', content=ASK_REWRITE_MESSAGE)
                messages_obj.append(role='assistant', content=rewrite_response)
                messages_obj.append(role='user', content="Decide whether to attempt the solution again, responding yes if you were incorrect or no if you were correct")
                rewrite_response = await generate_and_record(answer_model, messages_obj, all_inputs, all_outputs, n_sample=1)

            if clean_answer(rewrite_response)[-3:] != 'yes':
                j=MAX_REWRITES
                break

            ## PERFORM REWRITE
            messages_obj.reset()
            messages_obj.append(role='system', content=PROMPT_SYSTEM_MESSAGE)
            messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
            messages_obj.append(role='user', content=cot_answer_message)
            messages_obj.append(role='user', content=REWRITE_MESSAGE)
            
            response = await generate_and_record(cot_model, messages_obj, all_inputs, all_outputs, n_sample=1)
            cot_responses.append(response)
            
            
            ## EXTRACT NEW ANSWER
            messages_obj.reset()
            messages_obj.append(role='system', content=ANSWER_SYSTEM_MESSAGE)
            messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
            messages_obj.append(role='user', content=f"Proposed Solution:\n{cot_responses[-1]}")
            append_task_answer_prompt(task_name, messages_obj)
            
            answer = await generate_and_record(answer_model, messages_obj, all_inputs, all_outputs, n_sample=1)
            answer_responses.append(clean_answer(answer))
            
            j+=1


        print(f"\rDone example {i}", end='')
        sys.stdout.flush()
        
        return cot_responses, answer_responses, (all_inputs, all_outputs)
    
    
    


async def main():
    
    #### LOAD DATA ####
    
    task_texts, task_answers = load_task(TASK_NAME, f'data/{TASK_NAME}')

    task_texts_sample = random.sample(task_texts, NUM_EXAMPLES) if NUM_EXAMPLES < len(task_texts) else task_texts
    task_answers_sample = [task_answers[task_texts.index(t)] for t in task_texts_sample]

    total_examples = len(task_texts_sample)
    print(f"Executing on {total_examples} examples")


    #### DEFINE MODELS ####

    ## Define the models
    cot_model_name = MODEL_NAME
    cot_temp = 0.0001
    cot_max_tokens = 512

    answer_model_name = MODEL_NAME
    answer_temp = 0.0001
    answer_max_tokens = 256

    if MODEL_NAME == 'falcon-40b':
        cot_model = HFInferenceModel(model_name=cot_model_name, temperature=cot_temp, max_tokens=cot_max_tokens)
        answer_model = HFInferenceModel(model_name=answer_model_name, temperature=answer_temp, max_tokens=answer_max_tokens)
    elif MODEL_NAME == 'gpt-3.5-turbo':
        cot_model = GPTModel(model_name=cot_model_name, temperature=cot_temp, max_tokens=cot_max_tokens)
        answer_model = GPTModel(model_name=answer_model_name, temperature=answer_temp, max_tokens=answer_max_tokens)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")
    
    
    #### PREPARE ASYNC TASKS AND RUN ####

    # Create an asyncio task for each example in dataset
    tasks = [process_example(i, example, cot_model, answer_model, TASK_NAME) for i, example in enumerate(task_texts_sample)]

    # Run all the tasks in parallel
    results = await asyncio.gather(*tasks)
    print()
    cot_responses, pred_answers, inputs_outputs = zip(*results)
    final_answers = [a[-1] for a in pred_answers]
    
    
    #### RECORD RESULTS ####
    
    ## Record the results as a table
    results_df = pd.DataFrame(columns=['index','question', 'cot_response', 'pred_answer', 'true_answer', 'correct'])
    results_df['index'] = range(total_examples)
    results_df['question'] = task_texts_sample
    results_df['cot_response'] = cot_responses
    true_answers_procd = [str(a).lower().translate(str.maketrans("", "", string.punctuation)) for a in task_answers_sample]
    results_df['true_answer'] = true_answers_procd
    results_df['pred_answer'] = final_answers
    results_df['all_answers'] = pred_answers
    results_df['correct'] = results_df['true_answer'] == results_df['pred_answer']
    results_df['correct'] = results_df['correct'].astype(int)
    
    results_df.to_csv(os.path.join(RESULTS_DIR, "answers.tsv"), quotechar='"',sep='\t',index=False)
    results_df['io'] = inputs_outputs
    
    ## Record the results to a JSON file
    df_dict_list = results_df.to_dict('records')
    processed_dict_list = []
    for item in df_dict_list:
        new_item = {}
        
        # Copy the existing fields
        new_item['example_idx'] = item['index']
        new_item['question'] = item['question']
        new_item['true_answer'] = item['true_answer']
        new_item['predicted_answer'] = item['pred_answer']
        new_item['correct'] = item['correct']
        
        # Zip the 'cot_response' and 'answers' fields together
        response_pairs = [{'cot_response': c, 'answer': a} for c, a in zip(item['cot_response'], item['all_answers'])]
    
        # Store the zipped list and its length
        new_item['response_pairs'] = response_pairs
        new_item['response_count'] = len(response_pairs)
        
        new_item['queries']=[]
        for j, io in enumerate(zip(item['io'][0],item['io'][1])):
            new_item['queries'].append({'query_idx':j, 'input': io[0], 'output': io[1]})
        
        # Add the new item to the list
        processed_dict_list.append(new_item)
    
    # processed_inputs_outputs = []
    # for io in inputs_outputs:
    #     processed_inputs_outputs.append({'input': io[0], 'output': io[1]})
    
    total_prompt_cost = cot_model.prompts_cost() + answer_model.prompts_cost()
    total_completion_cost = cot_model.completions_cost() + answer_model.completions_cost()
    total_cost = total_prompt_cost + total_completion_cost
    
    results_dict = {
        "Task": TASK_NAME,
        "Model": MODEL_NAME,
        "Prompt type": PROMPT_TYPE,
        "Prompt": PROMPT_TEXT,
        "System message type": SYSTEM_MESSAGE_TYPE,
        "Prompt system message": PROMPT_SYSTEM_MESSAGE,
        "Answer system message": ANSWER_SYSTEM_MESSAGE,
        "Max rewrites": MAX_REWRITES,
        "Rewrite prompt"
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
        },
        'Examples': processed_dict_list,
        # 'Inputs outputs': processed_inputs_outputs
    }
    
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    
    ## Record experiment details to a JSON file
    # Generate the run details data structure
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
    with open(os.path.join(RESULTS_DIR, "details.json"), "w") as f:
        json.dump(details, f, indent=4)

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
    
    model_apis = {
        'gpt-3.5-turbo': 'openai',
        'falcon-40b': 'hf_inference',
        'falcon-40b-gptq': 'hf_inference',
        'llama-2-7b': 'hf_inference'
    }
    model_choices = model_apis.keys()
    
    prompt_choices = ['None',
                      'CoT',
                      'CoT-WS']
    system_message_choices = ['ChatGPT-default',
                              'instruct',
                              'instruct-list',
                              'None']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='gsm8k', help='', choices=task_choices)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='', choices=model_choices)
    parser.add_argument('--prompt_type', type=str, default='CoT', help='', choices=prompt_choices)
    parser.add_argument('--system_message_type', type=str, default='ChatGPT-default', help='', choices=system_message_choices)
    parser.add_argument('--run_identifier', type=str, default='', help='Used in file names')
    parser.add_argument('--num_examples', type=str, default='500', help='Either an integer or "all"')
    parser.add_argument('--async_concurr', type=int, default=7, help='Maximum number of concurrent requests to APIs')
    parser.add_argument('--max_rewrites', type=int, default=0, help='The number of times the solution may be rewritten. 0 means no rewriting.')
    
    
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TASK_NAME = args.task_name
    RUN_IDENTIFIER = args.run_identifier
    PROMPT_TYPE = args.prompt_type
    SYSTEM_MESSAGE_TYPE = args.system_message_type
    NUM_EXAMPLES = 1E9 if args.num_examples == 'all' else int(args.num_examples)
    MAX_REWRITES = args.max_rewrites
    
    # For text completion models
    USE_ROLES = False
    USE_SYSTEM_MESSAGES = False if SYSTEM_MESSAGE_TYPE == 'None' else True
    
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
        # PROMPT_TEXT = "Let's think about this step by step."
        PROMPT_TEXT = "Let's think about this step by step, putting each step on a new row of a table with structure |Step Number|Step description|Change in world state or knowledge|"
    elif PROMPT_TYPE == 'CoT-WS':
        PROMPT_TEXT = "Let's think about this step by step and describe how the state of the world, any values, or our knowledge changes at each step."
        # PROMPT_TEXT = "Let's think about this one step at a time, write concisely and put each step on a new row with structure: \"Step Number\tStep description\tList of available values after this step\""
    
    if SYSTEM_MESSAGE_TYPE == 'ChatGPT-default':
        PROMPT_SYSTEM_MESSAGE = "You are a helpful assistant."
    elif SYSTEM_MESSAGE_TYPE == 'instruct':
        PROMPT_SYSTEM_MESSAGE = "You are an instruction following, problem solving assistant."
        # PROMPT_SYSTEM_MESSAGE = "You are an instruction following, problem solving assistant. Do not do anything that is not explicitly instructed."
    elif SYSTEM_MESSAGE_TYPE == 'instruct-list':
        PROMPT_SYSTEM_MESSAGE = "You are an instruction following, problem solving assistant. Respond with lists \n1.\n2.\n3.\n..."
    elif SYSTEM_MESSAGE_TYPE == 'None':
        PROMPT_SYSTEM_MESSAGE = ""
    
    # ASK_REWRITE_MESSAGE = "Let's think about whether our previous solution is correct and decide if we should reattempt it. Do not explain the decision only respond yes to reattempt or no to terminate"
    # ASK_REWRITE_MESSAGE = "Let's think about whether our previous solution is correct and decide if we should reattempt it. Explain your reasoning. Your response must finish with either the word yes to reattempt or no to terminate"
    ASK_REWRITE_MESSAGE = "Validate your previous response, describing what if anything is wrong with your response. Do not give another solution You're answer could be correct, do not assume it is wrong. If youre answer is correct say so. Do not solve the problem again."
    REWRITE_MESSAGE = "Write a new solution to the problem. Use the previous solution but correct any errors described in the previous response."
    # REWRITE_SYSTEM_MESSAGE = PROMPT_SYSTEM_MESSAGE
    EXTRACT_REWRITE_RESPONSE = True
    
    ANSWER_SYSTEM_MESSAGE = "You are an instruction following, problem solving agent. Only respond with the exact answer. Do not explain your answers. Do not respond with sentences. Give exactly one answer."
    
    
    # Specifies the maximum number of concurrent requests to the OpenAI API
    MAX_CONCURRENT_REQUESTS = args.async_concurr # gpt-3.5 rate limit possibly hit above 8
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

    