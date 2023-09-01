from abc import ABC, abstractmethod
import src.models as md
import string
import asyncio
import sys


def clean_answer(answer, task_name=None):
    if task_name == "gsm8k":
        # Dont remove "."
        punctuation = string.punctuation.replace(".", "")
    else:
        punctuation = string.punctuation
    return str(answer).lower().translate(str.maketrans("", "", punctuation)).strip()


class PromptStrategy(ABC):
    def __init__(self, model_messages_json, task_name):
        self.models = self.define_models(model_messages_json)
        self.task_name = task_name

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
    def proc_example(self, example: str) -> tuple:
        pass

    async def generate_answer(self, query: str, cot_model="cot_generator", answer_model="answer_extractor"):
        n_sample = 1
        """Generates a single answer to a query and records the inputs and outputs"""
        cot_input_response = await self.models[cot_model].generate_async(
            query,
            n_sample=n_sample,
        )
        cot_response = cot_input_response["response"]

        answer_content = f"Problem Statement: {query}\n\nAnswer: {cot_response}"
        answer_input_response = await self.models[answer_model].generate_async(
            answer_content,
            n_sample=n_sample,
        )

        return (
            cot_response,
            clean_answer(answer_input_response["response"], self.task_name),
            (cot_input_response, answer_input_response),
        )


class PromptWithAnswerExtraction(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = ["cot_generator", "answer_extractor"]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            cot_response, answer, gen_respose_dicts = await self.generate_answer(example)
            
        print(f"\rDone example {example_num}", end="")
        sys.stdout.flush()
        
        return {
            "cot_responses": [cot_response],
            "answers": [answer],
            "query_details": gen_respose_dicts,
        }


class SolveValidateRewrite(PromptStrategy):
    def __init__(self, model_messages_json, task_name, rewrite_trigger='no', max_rewrites=2):
        self.max_rewrites = max_rewrites
        self.rewrite_trigger = rewrite_trigger
        self.required_model_tags = [
            "cot_generator",
            "answer_extractor",
            "validator",
            "decider",
            "rewriter"
        ]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            answers_lst = []
            cot_responses_lst = []
            gen_respose_dicts = []

            cot_response, answer, response_dict = await self.generate_answer(example)
            cot_responses_lst.append(cot_response)
            answers_lst.append(answer)
            gen_respose_dicts.extend(response_dict)

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
                val_response_dict = await self.models["validator"].generate_async(
                    validation_content
                )
                gen_respose_dicts.append(val_response_dict)

                decision_content = (
                    f"Solution validation: {val_response_dict['response']}"
                )
                decis_response_dict = await self.models["decider"].generate_async(
                    decision_content
                )
                gen_respose_dicts.append(decis_response_dict)

                if clean_answer(decis_response_dict["response"])[-3:] != self.rewrite_trigger:
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"Problem Statement: {example}\n\n: Validation of prior incorrect solution: {val_response_dict['response']}"
                cot_response, answer, response_dict = await self.generate_answer(
                    cot_content, cot_model="rewriter"
                )
                cot_responses_lst.append(cot_response)
                answers_lst.append(answer)
                gen_respose_dicts.extend(response_dict)

                j += 1

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()
            
            return {
                "cot_responses": cot_responses_lst,
                "answers": answers_lst,
                "query_details": gen_respose_dicts,
            }


class GoalExtraction(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = [
            "cot_generator",
            "answer_extractor",
            "goal_extractor",
        ]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            gen_respose_dicts = []

            goal_responses = await self.models["goal_extractor"].generate_async(example)
            gen_respose_dicts.append(goal_responses)

            solution_content = f"{example}\n\n{goal_responses['response']}"
            cot_response, answer, inputs_responses = await self.generate_answer(
                solution_content
            )
            gen_respose_dicts.extend(inputs_responses)

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()
            return {
                "cot_responses": [cot_response],
                "answers": [answer],
                "query_details": gen_respose_dicts,
            }



class GoalApproachRewrite(PromptStrategy):
    def __init__(self, model_messages_json, task_name, rewrite_trigger='no', max_rewrites=2):
        self.max_rewrites = max_rewrites
        self.rewrite_trigger = rewrite_trigger
        self.required_model_tags = [
            "goal_extractor",
            "cot_generator",
            "answer_extractor",
            "validator",
            "decider",
            "rewriter"
        ]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            answers_lst = []
            cot_responses_lst = []
            gen_respose_dicts = []
            
            # Goal and approach generation
            goal_responses = await self.models["goal_extractor"].generate_async(example)
            goal_approach_text = goal_responses['response']
            gen_respose_dicts.append(goal_responses)

            solution_content = f"Problem statement:{example}\n\nGoal and approach: {goal_approach_text}"

            cot_response, answer, response_dict = await self.generate_answer(solution_content)
            cot_responses_lst.append(cot_response)
            answers_lst.append(answer)
            gen_respose_dicts.extend(response_dict)

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
                val_response_dict = await self.models["validator"].generate_async(
                    validation_content
                )
                gen_respose_dicts.append(val_response_dict)

                decision_content = (
                    f"Solution validation: {val_response_dict['response']}"
                )
                decis_response_dict = await self.models["decider"].generate_async(
                    decision_content
                )
                gen_respose_dicts.append(decis_response_dict)

                if clean_answer(decis_response_dict["response"])[-3:] != self.rewrite_trigger:
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"{solution_content}\n\n: Validation of prior incorrect solution: {val_response_dict['response']}"
                cot_response, answer, response_dict = await self.generate_answer(
                    cot_content, cot_model="rewriter"
                )
                cot_responses_lst.append(cot_response)
                answers_lst.append(answer)
                gen_respose_dicts.extend(response_dict)

                j += 1

            print(f"\rDone example {example_num}", end="")
            sys.stdout.flush()
            
            return {
                "cot_responses": cot_responses_lst,
                "answers": answers_lst,
                "query_details": gen_respose_dicts,
            }
            


class SampleTree(PromptStrategy):
    def __init__(self, model_messages_json, task_name):
        self.required_model_tags = [
            "goal_extractor",
            "approach_generator",
            "step_executor",
            "answer_extractor",
        ]
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            gen_respose_dicts = []
            
            text_problem_statement = f"### Problem Statement ###\n{example}"
            
            # Goal and approach generation
            goal_response = await self.models["goal_extractor"].generate_async(text_problem_statement)
            gen_respose_dicts.append(goal_response)
            text_goal = f"### Goal ###\n{goal_response['response']}"

            approach_response = await self.models["approach_generator"].generate_async(text_problem_statement+"\n"+text_goal)
            gen_respose_dicts.append(approach_response)
            
            # Extract each row of the approach into a list
            lst_approach = approach_response['response'].split("\n")

            text_previous_steps = ""
            for text_step in lst_approach:
                
                input = f"{text_problem_statement}\n### Previous Steps ###\n{text_previous_steps}### Step to Execute ###\n{text_step}"
                
                step_response = await self.models["step_executor"].generate_async(input)
                gen_respose_dicts.append(step_response)
                
                format_step = step_response["response"].replace('\\n', ' ')
                text_previous_steps += f"{format_step}\n"
            
        
        text_cot = f"{text_problem_statement}\n{text_goal}### Steps ###\n{text_previous_steps}"
        answer_response = await self.models["answer_extractor"].generate_async(text_cot)
        gen_respose_dicts.append(answer_response)

        print(f"\rDone example {example_num}", end="")
        sys.stdout.flush()
            
        return {
            "cot_responses": [text_cot],
            "answers": [clean_answer(answer_response["response"], self.task_name)],
            "query_details": gen_respose_dicts,
        }
            
            
            
