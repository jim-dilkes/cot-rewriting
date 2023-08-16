from src import openai_utils
from abc import ABC, abstractmethod
from transformers import pipeline
import torch


OAI_CHAT_MODELS = {'gpt-3.5-turbo-0613','gpt-4-0613'}
OAI_LEGACY_MODELS = {'text-davinci-003'}
HF_MODELS = {'gpt2','gpt2-xl','meta-llama/Llama-2-7b-hf'}

HF_GENERATOR_CACHE = {}

def get_cached_hf_generator(model_name):
    """
    Only want to load each model once.
    Return a cached model instance if it exists; otherwise, create, cache, and return a new instance.
    """
    global HF_GENERATOR_CACHE
    if model_name not in HF_GENERATOR_CACHE:
        if model_name in HF_MODELS:
            HF_GENERATOR_CACHE[model_name] = pipeline(
                                        "text2text-generation",
                                        model=model_name, 
                                        torch_dtype=torch.float16,
                                        device_map="auto")
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    return HF_GENERATOR_CACHE[model_name]

def structure_message(role:str, content:str) -> dict:
    if role not in {'system', 'user', 'assistant'}:
        raise ValueError(f"Unknown chat role: {role}")
    return {'role': role, 'content': content}


def get_model(model_name:str, **kwargs):
    if model_name in OAI_CHAT_MODELS | OAI_LEGACY_MODELS:
        return GPTModelInstance(model_name=model_name, **kwargs)
    elif model_name in {'gpt2','gpt2-xl','meta-llama/Llama-2-7b-hf'}:
        return HFModelInstance(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Create an ABC for Models
class ModelInstance(ABC):

    
    def __init__(self, model_name:str, system_message=None, prompt=None, temperature=0.7, max_tokens=256):
        
        self.temperature = temperature
        self.max_tokens = max_tokens

        if model_name not in OAI_CHAT_MODELS | OAI_LEGACY_MODELS | HF_MODELS:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.model_name = model_name
        self.chat_model = model_name in OAI_CHAT_MODELS

    @abstractmethod
    async def generate_async(self, content:str, n_sample:int):
        pass


class GPTModelInstance(ModelInstance):

    def __init__(self, model_name: str, system_message=None, prompt=None, temperature=0.7, max_tokens=256):
        super().__init__(model_name, system_message, prompt, temperature, max_tokens)
        
        self.system_message = structure_message('system', system_message) # System message to prepend to all queries
        self.prompt_message = structure_message('user', prompt) # Prompt message to append to all queries
        
        self.tokenizer = openai_utils.get_tokenizer(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def generate_async(self,  content:str, n_sample:int=1, logit_bias:dict={}):
        
        query_messages = [self.system_message, structure_message('user', content), self.prompt_message]
        
        if self.model_type == 'chat':
            response = await openai_utils.chat_with_backoff_async(model=self.model_name,
                                                      messages=query_messages,
                                                      temperature=self.temperature,
                                                      max_tokens=self.max_tokens,
                                                      logit_bias=logit_bias,
                                                      n=n_sample)
        
        self.prompt_tokens += response['usage']['prompt_tokens']
        self.completion_tokens += response['usage']['completion_tokens']
        
        return query_messages, response['choices'][0]['message']['content']
    
    def prompts_cost(self):
        return openai_utils.cost(self.prompt_tokens, self.model_name, 'prompt') 
    
    def completions_cost(self):
        return openai_utils.cost(self.completion_tokens, self.model_name, 'completion')
    
    def total_cost(self):
        return self.prompts_cost() + self.completions_cost()
    
    
class HFModelInstance(ModelInstance):
    def __init__(self, model_name: str, system_message=None, prompt=None, temperature=0.7, max_tokens=256):
        super().__init__(model_name, system_message, prompt, temperature, max_tokens)
        
        self.system_message = system_message
        self.prompt_message = prompt
        
        self.generator = get_cached_hf_generator(model_name)
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    async def generate_async(self,  content:str, n_sample:int=1):
        query_text = "\n\n".join([self.system_message, content, self.prompt_message])
        
        response = self.generator(query_text,
                                  num_return_sequences=n_sample,
                                  do_sample=True,
                                  temperature=self.temperature,
                                  max_length=2056,
                                    )        
        response = response[0]['generated_text']
        response = response[len(query_text):] # Remove the query text from the response     
        
        return query_text, response[0]['generated_text']
    