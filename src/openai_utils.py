import openai
import tiktoken
import os
import backoff
import httpx
import json
import time

import logging

# Get a logger instance for the current file
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
MAX_TRIES = 5

@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=MAX_TRIES)
def completions_with_backoff(**kwargs):
# Adapted from https://github.com/princeton-nlp/tree-of-thought-llm
    # print(f"Making request to OpenAI API: {kwargs}")
    response = openai.Completion.create(**kwargs)
    # print(f"Response: {response}")
    return response

@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=MAX_TRIES)
def chat_with_backoff(**kwargs):
    # print(f"Making request to OpenAI API: {kwargs}")
    response = openai.ChatCompletion.create(**kwargs)
    # print(f"Response: {response}")
    return response

@backoff.on_exception(backoff.expo, (httpx.ReadTimeout, openai.error.OpenAIError), max_tries=MAX_TRIES)
async def chat_with_backoff_async(**kwargs):

    request_id = hash(frozenset(str(kwargs))) % 10000000000 # Short hash of kwargs to use as a request ID
    
    logger.debug(f"Making request {request_id} to OpenAI API: {kwargs}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post('https://api.openai.com/v1/chat/completions', headers={'Authorization': f'Bearer {api_key}'}, json=kwargs)
        except httpx.ReadTimeout as e:
            logger.error(f"Request {request_id} timed out: {e}")
            raise e
        
    if response.status_code != 200:
        logger.error(f"Response {request_id} ({response}): {response.text}")
        raise openai.error.OpenAIError(response.text)
    
    response_text = json.loads(response.text)
    logger.debug(f"Response {request_id} ({response}): {response_text}")
    return response_text


def get_tokenizer(model_name):
    return tiktoken.encoding_for_model(model_name)


def structure_message(role, content):
    if role not in {'system', 'user', 'assistant'}:
        raise ValueError(f"Unknown OpenAI chat role: {role}")
    return {'role': role, 'content': content}


class OpenAIChatMessages:
    """ Class to manage messages for OpenAI chat API """
    def __init__(self):
        self.messages = []
    
    def append(self, role, content):
        self.messages.append(structure_message(role, content))
        
    def prepend(self, role, content):
        [structure_message(role, content)].extend(self.messages)
        
    def get(self):
        return self.messages
        
    def reset(self):
        self.messages = []
        
        
def token_type_cost(n_tokens, token_type, prompt_cost, completion_cost):
    if token_type == 'prompt':
        return n_tokens * prompt_cost
    elif token_type == 'completion':
        return n_tokens * completion_cost
    else:
        raise ValueError(f"Unknown token type: {token_type}")
        
def cost(n_tokens, model_name, token_type):
    if model_name == 'gpt-3.5-turbo':
        prompt_cost = 0.0015/1000
        completion_cost = 0.002/1000
        return token_type_cost(n_tokens, token_type, prompt_cost, completion_cost)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
        