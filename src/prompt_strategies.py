from abc import ABC, abstractmethod
import src.models as md
import string
import asyncio


async def generate_and_record(model, content, inputs_lst, response_lst, **kwargs):
    """ Generate a response from the model and record the inputs and outputs """
    input, response = await model.generate_async(content, **kwargs)
    inputs_lst.append(input)
    response_lst.append(response)
    return response

def clean_answer(answer, task_name):
    if task_name=='gsm8k':
        # Dont include "."
        punctuation = string.punctuation.replace('.', '')
    else:
        punctuation = string.punctuation
    return str(answer).lower().translate(str.maketrans("", "", punctuation)).strip()


class AbstractPromptStrategy(ABC):
    def __init__(self, model_messages_json, task_name):
        self.models = self.define_models(model_messages_json)
        self.task_name = task_name

    def define_models(self, model_messages_json):
        models = {}
        for model_tag in self.required_model_tags:
            try:
                model_inputs = model_messages_json[model_tag]
            except KeyError as e:
                raise KeyError(
                    f"Model tag {model_tag} required for {self.__class__.__name__} prompt strategy"
                ) from e
                
            model_name = model_inputs.pop('model_name')
            kwargs = model_inputs
            
            models[model_tag] = md.get_model(model_name, **kwargs)
            
        return models

    @abstractmethod
    def proc_example(self, example):
        pass
    

class PromptWithAnswerExtraction(AbstractPromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = ['cot_generator','answer_extractor']
        super().__init__(model_messages_json, task_name)

            
    async def proc_example(self, example:str, semaphore: asyncio.Semaphore):
        async with semaphore:
            all_inputs_lst = []
            all_responses_lst = []
            answers_lst = []
            cot_responses_lst = []

            ## Generate CoT Solution
            cot_solution = await generate_and_record(self.models['cot_generator'],
                                                        example,
                                                        all_inputs_lst,
                                                        all_responses_lst)
            cot_responses_lst.append(cot_solution)
            
            ## Extract the answer from the final CoT solution
            answer_content = f"Problem Statement: {example}\n\nProposed Solution: {cot_solution}"
            answer = await generate_and_record(self.models['answer_extractor'],
                                                                    answer_content,
                                                                    all_inputs_lst,
                                                                    all_responses_lst)
            answers_lst.append(clean_answer(answer, self.task_name))
            
            return cot_responses_lst, answers_lst, (all_inputs_lst, all_responses_lst)
        