from abc import ABC, abstractmethod
import src.models as md
import string
import asyncio
import sys


async def generate_and_record(model, content, **kwargs):
    """Generate a response from the model and record the inputs and outputs"""
    input, response = await model.generate_async(content, **kwargs)
    return {"input": input, "response": response}


def clean_answer(answer, task_name=None):
    if task_name == "gsm8k":
        # Dont include "."
        punctuation = string.punctuation.replace(".", "")
    else:
        punctuation = string.punctuation
    return str(answer).lower().translate(str.maketrans("", "", punctuation)).strip()


class PromptStrategy(ABC):
    def __init__(self, model_messages_json, task_name):
        self.models = self.define_models(model_messages_json)
        self.task_name = task_name

        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        try:
            self.required_model_tags  # Ensure that require_model_tags is defined
        except AttributeError as e:
            raise AttributeError(
                f"required_model_tags must be defined for {self.__class__.__name__} prompt strategy"
            ) from e

    def define_models(self, model_messages_json):
        models = {}
        for model_tag in self.required_model_tags:
            try:
                model_inputs = model_messages_json[model_tag].copy()
            except KeyError as e:
                raise KeyError(
                    f"Model tag {model_tag} required for {self.__class__.__name__} prompt strategy"
                ) from e

            model_name = model_inputs.pop("model_name")
            kwargs = model_inputs

            models[model_tag] = md.get_model(model_name, **kwargs)

        return models

    @abstractmethod
    def proc_example(self, example:str) -> tuple:
        pass

    async def generate_answer(self,query: str):
        n_sample=1
        """Generates a single answer to a query and records the inputs and outputs"""
        cot_input_response = await generate_and_record(
            self.models["cot_generator"],
            query,
            n_sample=n_sample,
        )
        cot_response = cot_input_response["response"]
        
        answer_content = f"Problem Statement: {query}\n\nSolution: {cot_response}"
        answer_input_response = await generate_and_record(
            self.models["answer_extractor"],
            answer_content,
            n_sample=n_sample,
        )        
        answer = clean_answer(answer_input_response["response"], self.task_name)
        
        inputs_responses = [cot_input_response, answer_input_response]
        
        # Return all input, response pairs and the answer
        return cot_response, answer, inputs_responses


class PromptWithAnswerExtraction(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = ["cot_generator", "answer_extractor"]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        
        async with semaphore:
            prompt_tokens = 0
            completion_tokens = 0

            cot_response, answer, inputs_responses = await self.generate_answer(example)

            return_dict = {
                    "cot_responses": [cot_response],
                    "answers": [answer],
                    "all_io": inputs_responses,
                    "tokens": {"prompts": prompt_tokens, "completions": completion_tokens},
                    }
            print(return_dict)

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()
            return return_dict


class SolveValidateRewrite(PromptStrategy):
    def __init__(self, model_messages_json, task_name, max_rewrites=2):
        self.max_rewrites = max_rewrites
        self.required_model_tags = [
            "cot_generator",
            "answer_extractor",
            "validator",
            "decider",
        ]
        super().__init__(model_messages_json, task_name)

    async def proc_example(self, example: str, example_num: int, semaphore: asyncio.Semaphore):
        async with semaphore:
            answers_lst = []
            cot_responses_lst = []
            all_inputs_responses = []
            prompt_tokens = 0
            completion_tokens = 0

            cot_response, answer, inputs_responses = await self.generate_answer(example)
            cot_responses_lst.append(cot_response)
            answers_lst.append(answer)
            all_inputs_responses.extend(inputs_responses)
            

            j = 0
            while j < self.max_rewrites:
                ## Validate the solution and decide whether to rewrite
                cot_answer_content = "\n".join(
                    [
                        f"Proposed solution {i+1}: {r[0]}\nAnswer {i+1}: {r[1]}"
                        for i, r in enumerate(zip(cot_responses_lst, answers_lst))
                    ]
                )

                validation_content = f"Problem statement: {example}\n\n{cot_answer_content}"
                validation_input_response = await generate_and_record(self.models["validator"], validation_content)
                all_inputs_responses.append(validation_input_response)


                decision_content = f"Solution validation: {validation_input_response['response']}"
                decision_input_response = await generate_and_record(self.models["decider"], decision_content)
                all_inputs_responses.append(decision_input_response)

                # if clean_answer(decision_response)[-3:] != "yes":
                if clean_answer(decision_input_response['response'])[-3:] != "no":
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"Problem Statement: {example}\n\nPrevious erroneous attempts: {cot_answer_content}"
                cot_response, answer, inputs_responses = await self.generate_answer(cot_content)
                cot_responses_lst.append(cot_response)
                answers_lst.append(answer)
                all_inputs_responses.extend(inputs_responses)                

                j += 1

            return_dict = {
                    "cot_responses": cot_responses_lst,
                    "answers": answers_lst,
                    "all_io": all_inputs_responses,
                    "tokens": {"prompts": prompt_tokens, "completions": completion_tokens},
                    }

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()

            return return_dict


class GoalExtraction(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = [
            "cot_generator",
            "answer_extractor",
            "goal_extractor"
        ]
        super().__init__(model_messages_json, task_name)
        
        
    async def proc_example(self, example: str, example_num: int, semaphore: asyncio.Semaphore):
        
        async with semaphore:
            all_inputs_responses = []
            prompt_tokens = 0
            completion_tokens = 0

            goal_responses = await generate_and_record(self.models["goal_extractor"],example)
            goal_response = goal_responses["response"]
            all_inputs_responses.append({"input":goal_responses["input"],"response":goal_response})
            
            solution_content = f"{example}\n\n{goal_response}"
            # Updates cot_responses_lst and answers_lst in place
            cot_response, answer, inputs_responses = await self.generate_answer(solution_content)     
            all_inputs_responses.extend(inputs_responses)

            return_dict = {
                    "cot_responses": [cot_response],
                    "answers": [answer],
                    "all_io": all_inputs_responses,
                    "tokens": {"prompts": prompt_tokens, "completions": completion_tokens},
                    }
            print(return_dict)

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()
            return return_dict
