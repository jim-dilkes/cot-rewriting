from abc import ABC, abstractmethod
import src.models as md
import string
import asyncio
import sys


async def generate_and_record(model, content, inputs_lst, response_lst, **kwargs):
    """Generate a response from the model and record the inputs and outputs"""
    input, response = await model.generate_async(content, **kwargs)
    inputs_lst.append(input)
    response_lst.append(response)
    return response


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
                model_inputs = model_messages_json[model_tag]
            except KeyError as e:
                raise KeyError(
                    f"Model tag {model_tag} required for {self.__class__.__name__} prompt strategy"
                ) from e

            model_name = model_inputs.pop("model_name")
            kwargs = model_inputs

            models[model_tag] = md.get_model(model_name, **kwargs)

        return models

    @abstractmethod
    def proc_example(self, example):
        pass

    async def generate_answer(
        self,
        query: str,
        all_inputs: list,
        all_responses: list,
        cot_responses: list,
        answers: list,
        n_sample=1,
    ):
        """Generates a single answer to a query and records the inputs and outputs"""
        cot_solution = await generate_and_record(
            self.models["cot_generator"],
            query,
            all_inputs,
            all_responses,
            n_sample=n_sample,
        )
        cot_responses.append(cot_solution)

        answer_content = f"{cot_solution}"
        answer = await generate_and_record(
            self.models["answer_extractor"],
            answer_content,
            all_inputs,
            all_responses,
            n_sample=n_sample,
        )
        answers.append(clean_answer(answer, self.task_name))


class PromptWithAnswerExtraction(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = ["cot_generator", "answer_extractor"]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            all_inputs_lst = []
            all_responses_lst = []
            answers_lst = []
            cot_responses_lst = []

            # Updates cot_responses_lst and answers_lst in place
            await self.generate_answer(
                example,
                all_inputs_lst,
                all_responses_lst,
                cot_responses_lst,
                answers_lst,
            )

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()

            return cot_responses_lst, answers_lst, (all_inputs_lst, all_responses_lst)


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
            all_inputs_lst = []
            all_responses_lst = []
            answers_lst = []
            cot_responses_lst = []

            # Updates cot_responses_lst and answers_lst in place
            await self.generate_answer(
                example,
                all_inputs_lst,
                all_responses_lst,
                cot_responses_lst,
                answers_lst,
            )

            j = 0
            while j < self.max_rewrites:
                ## Validate the solution and decide whether to rewrite
                cot_answer_content = "\n".join(
                    [
                        f"Proposed solution {i+1}: {r[0]}\nAnswer {i+1}: {r[1]}"
                        for i, r in enumerate(zip(cot_responses_lst, answers_lst))
                    ]
                )

                validation_content = (
                    f"Problem statement: {example}\n\n{cot_answer_content}"
                )
                validation_response = await generate_and_record(
                    self.models["validator"],
                    validation_content,
                    all_inputs_lst,
                    all_responses_lst,
                )

                decision_content = f"Solution validation: {validation_response}"
                decision_response = await generate_and_record(
                    self.models["decider"],
                    decision_content,
                    all_inputs_lst,
                    all_responses_lst,
                )

                if clean_answer(decision_response)[-3:] != "yes":
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"Problem Statement: {example}\n\nPrevious erroneous attempts: {cot_answer_content}"
                await self.generate_answer(
                    cot_content,
                    all_inputs_lst,
                    all_responses_lst,
                    cot_responses_lst,
                    answers_lst,
                )

                j += 1

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()

            return cot_responses_lst, answers_lst, (all_inputs_lst, all_responses_lst)
