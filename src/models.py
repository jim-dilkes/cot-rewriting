from src import openai_utils
from abc import ABC, abstractmethod
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch


OAI_CHAT_MODELS = {"gpt-3.5-turbo-0613", "gpt-4-0613"}
OAI_LEGACY_MODELS = {"text-davinci-003"}
HF_LLAMA_CHAT_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
}
HF_MODELS = {
    "gpt2",
    "gpt2-xl",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
}


HF_GENERATOR_CACHE = {}


def get_cached_hf_generator(model_name):
    """
    Only want to load each model once.
    Return a cached model instance if it exists; otherwise, create, cache, and return a new instance.
    """

    if model_name not in HF_GENERATOR_CACHE:
        if model_name in HF_MODELS | HF_LLAMA_CHAT_MODELS:
            llama_70b_prefix = "meta-llama/Llama-2-70b"
            if model_name[: len(llama_70b_prefix)] == llama_70b_prefix:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map="auto",
                )

            HF_GENERATOR_CACHE[model_name] = pipeline(
                "text-generation",
                # return_full_text=True,
                model=model,
                tokenizer=AutoTokenizer.from_pretrained(model_name),
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    return HF_GENERATOR_CACHE[model_name]


def structure_message(role: str, content: str) -> dict:
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Unknown chat role: {role}")
    return {"role": role, "content": content}


def get_model(model_name: str, **kwargs):
    if model_name in OAI_CHAT_MODELS | OAI_LEGACY_MODELS:
        return GPTModelInstance(model_name=model_name, **kwargs)
    elif model_name in HF_LLAMA_CHAT_MODELS:
        return HfLlamaChatModelInstance(model_name=model_name, **kwargs)
    elif model_name in HF_MODELS:
        return HfModelInstance(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# Create an ABC for Models
class ModelInstance(ABC):
    def __init__(
        self,
        model_name: str,
        system_message=None,
        prompt=None,
        temperature=0.7,
        max_tokens=256,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        if (
            model_name
            not in OAI_CHAT_MODELS
            | OAI_LEGACY_MODELS
            | HF_MODELS
            | HF_LLAMA_CHAT_MODELS
        ):
            raise ValueError(f"Unknown model name: {model_name}")

        self.model_name = model_name
        self.is_chat_model = model_name in OAI_CHAT_MODELS

    @abstractmethod
    async def generate_async(self, content: str, n_sample: int):
        pass


class GPTModelInstance(ModelInstance):
    def __init__(
        self,
        model_name: str,
        system_message=None,
        prompt=None,
        temperature=0.7,
        max_tokens=256,
    ):
        super().__init__(model_name, system_message, prompt, temperature, max_tokens)

        self.system_message = structure_message(
            "system", system_message
        )  if system_message != "" else None # System message to prepend to all queries
        self.prompt_message = structure_message(
            "user", prompt
        )  if prompt != "" else None  # Prompt message to append to all queries

        self.tokenizer = openai_utils.get_tokenizer(model_name)

    async def generate_async(
        self, content: str, n_sample: int = 1, logit_bias: dict = {}
    ):
        query_messages = [
            self.system_message,
            structure_message("user", content),
            self.prompt_message,
        ]
        query_messages = [m for m in query_messages if m is not None]

        if self.is_chat_model:
            response = await openai_utils.chat_with_backoff_async(
                model=self.model_name,
                messages=query_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logit_bias=logit_bias,
                n=n_sample,
            )

        return {
                "input": query_messages,
                "response": response["choices"][0]["message"]["content"],
                "token_counts": response["usage"]
        }

    def prompts_cost(self):
        return openai_utils.cost(self.prompt_tokens, self.model_name, "prompt")

    def completions_cost(self):
        return openai_utils.cost(self.completion_tokens, self.model_name, "completion")

    def total_cost(self):
        return self.prompts_cost() + self.completions_cost()


class HfModelInstance(ModelInstance):
    def __init__(
        self,
        model_name: str,
        system_message=None,
        prompt=None,
        temperature=0.7,
        max_tokens=256,
    ):
        super().__init__(model_name, system_message, prompt, temperature, max_tokens)

        self.system_message = system_message
        self.prompt_message = prompt

        self.generator = get_cached_hf_generator(model_name)

    def format_input(self, system_message: str, content: str, prompt_message: str):
        return "\n\n".join([system_message, content, prompt_message])

    async def generate_async(self, content: str, n_sample: int = 1):
        query_text = self.format_input(
            self.system_message, content, self.prompt_message
        )

        response = self.generator(
            query_text,
            num_return_sequences=n_sample,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            repetition_penalty=1.1,
        )
        response = response[0]["generated_text"]
        response = response[
            len(query_text):
        ]  # Remove the query text from the response

        
        return {
                "input": query_text,
                "response": response,
                "token_counts": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
        }


class HfLlamaChatModelInstance(HfModelInstance):
    # The chat versions of the Llama2 models are fine tuned to use a specific prompt format.
    def format_input(self, system_message: str, content: str, prompt_message: str):
        return f"<s><<SYS>>\n{system_message}\n<</SYS>>\n\n[INST]{content}\n\n{prompt_message}[/INST] "
