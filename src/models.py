from abc import ABC, abstractmethod
from src import openai_utils



class LanguageModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, input, temperature:float, max_tokens:int, n_sample:int):
        pass
    

class GPTModel(LanguageModel):
    def __init__(self, model_name:str, system_message=None, temperature=0.7, max_tokens=256):
        self.model_name = model_name
        self.tokenizer = openai_utils.get_tokenizer(model_name)
        
        self.system_message_obj = [openai_utils.structure_message('system',system_message)] if system_message else []
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.prompt_tokens = 0
        self.completion_tokens = 0

        if model_name in {'gpt-3.5-turbo'}:
            self.model_name = model_name
            self.openai_model_type = 'chat'
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def generate(self,  input, n_sample:int=1, logit_bias:dict={}):
        if self.openai_model_type == 'chat':
            response = openai_utils.chat_with_backoff(model=self.model_name,
                                                      messages=self.system_message_obj + input,
                                                      temperature=self.temperature,
                                                      max_tokens=self.max_tokens,
                                                      logit_bias=logit_bias,
                                                      n=n_sample)
            
        self.prompt_tokens += response['usage']['prompt_tokens']
        self.completion_tokens += response['usage']['completion_tokens']
        
        return response['choices'][0]['message']['content']
    
    async def generate_async(self,  input:list, n_sample:int=1, logit_bias:dict={}):
        
        if self.openai_model_type == 'chat':
            response = await openai_utils.chat_with_backoff_async(model=self.model_name,
                                                      messages=self.system_message_obj + input,
                                                      temperature=self.temperature,
                                                      max_tokens=self.max_tokens,
                                                      logit_bias=logit_bias,
                                                      n=n_sample)
        
        self.prompt_tokens += response['usage']['prompt_tokens']
        self.completion_tokens += response['usage']['completion_tokens']
        
        return response['choices'][0]['message']['content']
    
    
    def prompts_cost(self):
        return openai_utils.cost(self.prompt_tokens, self.model_name, 'prompt')
    
    def completions_cost(self):
        return openai_utils.cost(self.completion_tokens, self.model_name, 'completion')
    
    def total_cost(self):
        return self.prompts_cost() + self.completions_cost()
    
    
    
    
    