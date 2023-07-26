from src import openai_utils, hf_inference_utils


class GPTModel():
    def __init__(self, model_name:str, use_system_message=True, temperature=0.7, max_tokens=256):
        self.model_name = model_name
        self.tokenizer = openai_utils.get_tokenizer(model_name)
        self.use_system_message = use_system_message
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.prompt_tokens = 0
        self.completion_tokens = 0

        if model_name in {'gpt-3.5-turbo'}:
            self.model_name = model_name
            self.model_type = 'chat'
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    async def generate_async(self,  messages_input:openai_utils.OpenAIChatMessages, n_sample:int=1, logit_bias:dict={}):
        text_input = messages_input.get(chat_model=True)
        if self.model_type == 'chat':
            response = await openai_utils.chat_with_backoff_async(model=self.model_name,
                                                      messages=text_input,
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
    