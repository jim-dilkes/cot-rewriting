import argparse

from src.gsm8k_utils import load_gsm8k
from src.models import GPTModel
from src.openai_utils import OpenAIChatMessages

import pandas as pd
import asyncio

import logging


def task_answer_prompt(task_name, messages_obj:OpenAIChatMessages):
    if task_name == 'gsm8k':
        messages_obj.append(role='user', content="Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols")
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")        
    
def load_task(task_name, task_dir):
    if task_name == 'gsm8k':
        gsm8k_dataset = load_gsm8k('test.jsonl', filedir=task_dir)
        gsm8k_texts = [example['question'] for example in gsm8k_dataset]
        gsm8k_answers = [example['numerical_answer'] for example in gsm8k_dataset]
        return gsm8k_texts, gsm8k_answers
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")

async def process_example(i, example_text, cot_model, answer_model, task_name):
    # Acquire the semaphore before processing an example
    async with SEMAPHORE:
        messages_obj = OpenAIChatMessages()
        ## Generate CoT Solution
        messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
        messages_obj.append(role='user', content="Describe a step by step solution to this problem, with one step per line")
        cot_response = await cot_model.generate_async(messages_obj.get(), n_sample=1)
        
        messages_obj.reset()
        messages_obj.append(role='user', content=f"Problem Statement:\n{example_text}")
        messages_obj.append(role='user', content=f"Proposed Solution:\n{cot_response}")
        
        task_answer_prompt(task_name, messages_obj)
        
        answer_response = await answer_model.generate_async(messages_obj.get(), n_sample=1)
        answer_response = answer_response.strip()
    
        print(f"Done {i}")
        
        return cot_response, answer_response


async def main():
    ## Load the data
    task_texts, task_answers = load_task(TASK_NAME, f'data/{TASK_NAME}')
    task_texts_sample=task_texts[:4]
    task_answers_sample=task_answers[:4]
    total_examples = len(task_texts_sample)
    
    ## Define the models
    cot_model_name = MODEL_NAME
    cot_temp = 0.7
    cot_max_tokens = 256
    cot_system_message = "You are an instruction following, problem solving agent. Provide concise, practical responses to instructions. Do not explain responses. Structure:\n1. Step number 1\n2. Step number 2\n3. Step number 3\n..."
    # cot_system_message = "You are a helpful asssistant."
    cot_model = GPTModel(model_name=cot_model_name, system_message=cot_system_message, temperature=cot_temp, max_tokens=cot_max_tokens)
    
    answer_model_name = MODEL_NAME
    answer_temp = 0
    answer_max_tokens = 20
    answer_system_message = "You are an instruction following, problem solving agent. Only respond with numbers, without currencies or units or symbols. Do not explain your answers. You give exactly one numerical answer."
    answer_model = GPTModel(model_name=answer_model_name, system_message=answer_system_message, temperature=answer_temp, max_tokens=answer_max_tokens)

    # Create an asyncio task for each example in dataset
    tasks = [process_example(i, example, cot_model, answer_model, TASK_NAME) for i, example in enumerate(task_texts_sample)]

    # Run all the tasks in parallel
    results = await asyncio.gather(*tasks)
    cot_responses, answers = zip(*results)

    ## Format and record the results            
    results_df = pd.DataFrame(columns=['index','question', 'answer', 'cot_response', 'answer_response', 'correct'])
    results_df['index'] = range(total_examples)
    results_df['question'] = task_texts_sample
    results_df['answer'] = task_answers_sample
    results_df['cot_response'] = cot_responses
    results_df['answer_response'] = answers
    results_df['correct'] = results_df['answer'] == results_df['answer_response']
        
    results_df.to_csv(f'.results/{RUN_NAME}_gsm8k_results.tsv',quotechar='"',sep='\t',index=False)
    
    print("Accuracy:", results_df['correct'].mean())
    
    total_prompt_cost = cot_model.prompts_cost() + answer_model.prompts_cost()
    total_completion_cost = cot_model.completions_cost() + answer_model.completions_cost()
    total_cost = total_prompt_cost + total_completion_cost
    
    print(f"\nTotal prompt cost: ${total_prompt_cost:.5f}, per example: ${total_prompt_cost/total_examples:.5f}")
    print(f"Total completion cost: ${total_completion_cost:.5f}, per example: ${total_completion_cost/total_examples:.5f}")
    print(f"Total cost: ${total_cost:.5f}, per example: ${total_cost/total_examples:.5f}")


        
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='gsm8k', help='', choices=['gsm8k'])
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='', choices=['gpt-3.5-turbo'])
    parser.add_argument('--run_name', type=str, default='default', help='Used in file names')
    parser.add_argument('--async_concurr', type=str, default='10', help='Maximum number of concurrent requests to APIs')
    
    # python run.py --task_name gsm8k --model_name gpt-3.5-turbo run_name --test
    
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    TASK_NAME = args.task_name
    RUN_NAME = args.run_name
    
    # Determines the maximum number of concurrent requests to the OpenAI API
    MAX_CONCURRENT_REQUESTS = int(args.async_concurr) # gpt-3.5 rate limit possibly hit above 10
    SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    log_dir = f'./logs/{TASK_NAME}'
    
    # Configure logging
    logging.basicConfig(filename=f'{log_dir}/{RUN_NAME}.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    asyncio.run(main())

    