import openai
import tiktoken
import os
import backoff
import httpx, ssl
import json
import random
import logging

# Get a logger instance for the current file
logger_name = "main_logger"
logger = logging.getLogger(logger_name)

key_pool_envvars = ['OPENAI_API_KEY', 'OPENAI_API_KEY_2']
key_pool = [os.getenv(envvar, "") for envvar in key_pool_envvars]

for i, key in enumerate(key_pool):
    if key == "":
        print(f"Warning: {key_pool_envvars[i]} is not set")

last_used_api_key = ""
    
MAX_TRIES = 10

@backoff.on_exception(backoff.expo, (httpx.RequestError, ssl.SSLError, TimeoutError, openai.error.OpenAIError), max_tries=MAX_TRIES)
async def chat_with_backoff_async(**kwargs):

    request_id = hash(frozenset(str(kwargs))) % 10000000000 # Short hash of kwargs to identify requests
    
    logger.debug(f"Making request {request_id} to OpenAI API: {kwargs}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Select an API key that wasnt last used
            global last_used_api_key
            use_api_key = random.choice([key for key in key_pool if key != last_used_api_key])
            last_used_api_key = use_api_key
            response = await client.post('https://api.openai.com/v1/chat/completions', headers={'Authorization': f'Bearer {use_api_key}'}, json=kwargs)
        except httpx.ReadTimeout as e:
            logger.error(f"Request {request_id} timed out: {e}")
            raise e
        except Exception as e:
            logger.error(f"Request {request_id} encountered an error: {e}")
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
        
    
    def format_single_prompt(self, join_char='\n', use_roles=False, use_system_messages=False):
        # Format messages as a single prompt, using only the content joined by join_char
        msgs = self.messages
        if not use_system_messages:
            msgs = [msg for msg in msgs if msg['role'] != 'system']
            
        if use_roles:
            formatted_messages = join_char.join([f"{m['role']}: {m['content']}" for m in msgs])
        else:
            formatted_messages = join_char.join([m['content'] for m in msgs])
        
        return formatted_messages
            
        
    def get(self, chat_model=True, join_char='\n', use_roles=False, use_system_messages=False):
        return self.messages if chat_model else self.format_single_prompt()
        
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
    
        