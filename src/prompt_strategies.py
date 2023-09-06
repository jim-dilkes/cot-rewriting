from abc import ABC, abstractmethod
import src.models as md
import string
import asyncio
import sys
import regex as re

def clean_answer(answer, task_name=None):
    if task_name == "gsm8k":
        # Dont remove "."
        punctuation = string.punctuation.replace(".", "")
    else:
        punctuation = string.punctuation
    return str(answer).lower().translate(str.maketrans("", "", punctuation)).strip()


class PromptStrategy(ABC):
    def __init__(self, model_messages_json, task_name):
        self.task_name = task_name

        try:
            self.required_model_tags  # Ensure that require_model_tags is defined
        except AttributeError as e:
            raise AttributeError(
                f"required_model_tags must be defined for {self.__class__.__name__} prompt strategy"
            ) from e
        try:
            self.optional_model_tags
        except AttributeError:
            self.optional_model_tags = []

        self.models = self.define_models(model_messages_json)

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
            
        for model_tag in self.optional_model_tags:
            # Attempt to get the model, but don't raise an error if it doesn't exist
            try:
                model_inputs = model_messages_json[model_tag].copy()
            except KeyError:
                continue
            model_name = model_inputs.pop("model_name")
            kwargs = model_inputs
            models[model_tag] = md.get_model(model_name, **kwargs)

        return models

    @abstractmethod
    def proc_example(self, example: str) -> tuple:
        pass

    async def generate_answer(self, query: str, cot_model="cot_generator", answer_model="answer_extractor"):

        """Generates a single answer to a query and records the inputs and outputs"""
        cot_input_response = await self.models[cot_model].generate_async(
            query
        )
        cot_response = cot_input_response["response"][0]

        answer_content = f"Problem Statement: {query}\n\nAnswer: {cot_response}"
        answer_input_response = await self.models[answer_model].generate_async(
            answer_content
        )

        return (
            cot_response,
            clean_answer(answer_input_response["response"][0], self.task_name),
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

                validation_content = (f"Problem statement: {example}\n\n{cot_answer_content}")
                val_response_dict = await self.models["validator"].generate_async(validation_content)
                gen_respose_dicts.append(val_response_dict)
                val_response_dict = val_response_dict['response'][0]

                decision_content = (f"Solution validation: {val_response_dict}")
                decis_response_dict = await self.models["decider"].generate_async(decision_content)
                gen_respose_dicts.append(decis_response_dict)
                decis_response_dict = decis_response_dict['response'][0]
                
                if clean_answer(decis_response_dict)[-3:] != self.rewrite_trigger:
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"Problem Statement: {example}\n\n: Validation of prior incorrect solution: {val_response_dict}"
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
            goal_responses = goal_responses['response'][0]

            solution_content = f"{example}\n\n{goal_responses}"
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
            goal_responses = await self.models["goal_extractor"].generate_async(example)[0]
            gen_respose_dicts.append(goal_responses)
            goal_responses = goal_responses['response'][0]

            solution_content = f"Problem statement:{example}\n\nGoal and approach: {goal_responses}"

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
                val_response_dict = await self.models["validator"].generate_async(validation_content)
                gen_respose_dicts.append(val_response_dict)
                val_response_dict = val_response_dict['response'][0]

                decision_content = (f"Solution validation: {val_response_dict}")
                decis_response_dict = await self.models["decider"].generate_async(decision_content)
                gen_respose_dicts.append(decis_response_dict)
                decis_response_dict = decis_response_dict['response'][0]

                if clean_answer(decis_response_dict)[-3:] != self.rewrite_trigger:
                    break

                ## If we are rewriting, generate a new solution, using the previous solution as context
                cot_content = f"{solution_content}\n\n: Validation of prior incorrect solution: {val_response_dict['response']}"
                cot_response, answer, response_dict = await self.generate_answer(cot_content, cot_model="rewriter")
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
    def __init__(self, model_messages_json, task_name, n_step_samples=1, n_approach_samples=1):
        self.required_model_tags = [
            "goal_extractor",
            "approach_generator",
            "step_executor",
            "answer_extractor",
        ]
        self.optional_model_tags = [
            "approach_selector",
            "step_selector"
        ]
        self.n_step_samples=n_step_samples
        self.n_approach_samples=n_approach_samples
        super().__init__(model_messages_json, task_name)

    async def proc_example(
        self, example: str, example_num: int, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            gen_respose_dicts = []

            ## Goal and approach generation
            # Goal
            text_problem_statement = f"### Problem Statement ###\n{example}"
            goal_response = await self.models["goal_extractor"].generate_async(text_problem_statement)
            gen_respose_dicts.append(goal_response)
            goal_response = goal_response['response'][0]
            # Approach
            text_goal = f"### Goal ###\n{goal_response}"
            approach_responses_lst = await self.models["approach_generator"].generate_async(text_problem_statement+"\n"+text_goal, n_sample=self.n_approach_samples)
            gen_respose_dicts.append(approach_responses_lst)
            approach_responses_lst = approach_responses_lst['response']

            # approach_response is a string including numbers in braces {1}, {2}, etc.
            # we want to split this into a list of steps, remove the newlines, then stick each step together with a newline
            approach_responses_lst = [re.sub(r"\n", " ", step) for step in approach_responses_lst]
            approach_responses_lst = [re.sub(r"\{[2-9]+\}", "\n", step) for step in approach_responses_lst]
            # Remove duplicate responses from approach_responses_lst
            approach_responses_lst = list(dict.fromkeys(approach_responses_lst))            
            
            if len(approach_responses_lst) > 1 and "approach_selector" in self.models:
                # Join the samples, but with ascending letters at the start of each line {A}, {B}, etc.
                text_approach_samples = "\n".join([f"{{{chr(65+i)}}}: {r}" for i, r in enumerate(approach_responses_lst)])
                input = f"{text_problem_statement}\n### Goal ###\n{text_goal}\n### Options to Analyse ###{text_approach_samples}"
                approach_selection_response = await self.models["approach_selector"].generate_async(input)
                gen_respose_dicts.append(approach_selection_response)
                approach_selection_response = approach_selection_response['response'][0]
                
                # Extract the selected approach from the samples: the last instance of {X} in the response
                selected_approach = re.findall(r"\{([A-Z])\}", approach_selection_response)
                # If none found, use A
                selected_approach = selected_approach[-1] if selected_approach else "A"
                # Extract the selected approach, by converting slection from A,B,C... to 0,1,2...
                # If the selected approach is not between 0 and len(approach_response), use 0
                selected_approach_idx = ord(selected_approach)-65
                selected_approach_idx = selected_approach_idx if selected_approach_idx < len(approach_responses_lst) else 0
                approach_response_str = approach_responses_lst[selected_approach_idx]
            else:
                approach_response_str = approach_responses_lst[0]

            # Extract each row of the approach into a list
            lst_approach = approach_response_str.split("\n")
            # If the number of steps is greater than 5, keep the last 5
            lst_approach = lst_approach[-min(len(lst_approach), 5):]
            lst_approach[-1] = f"{lst_approach[-1]} Conclude by stating the answer to the problem."

            text_previous_steps = ""
            for text_step in lst_approach:

                input = f"{text_problem_statement}\n### Previous Steps ###\n{text_previous_steps}### Step to Execute ###\n{text_step}"

                step_response = await self.models["step_executor"].generate_async(input, n_sample=self.n_step_samples)
                gen_respose_dicts.append(step_response)

                # Remove duplicate responses from step_samples_lst
                step_samples_lst = step_response['response'] # Returns a list if n_sample > 1 # TODO: change this to always return list
                step_samples_lst = list(dict.fromkeys(step_samples_lst))

                # Query the model to analyse and select the best step if multiple are sampled
                if len(step_samples_lst) > 1 and "step_selector" in self.models:
                    # Join the samples, but with ascending letters at the start of each line {A}, {B}, etc.
                    text_step_samples = "\n".join([f"{{{chr(65+i)}}}: {r}" for i, r in enumerate(step_samples_lst)])
                    input = f"### Previous Steps ###\n{text_previous_steps}### Step to Execute ###\n{text_step}\n### Options to Analyse ###{text_step_samples}"
                    step_selection_response = await self.models["step_selector"].generate_async(input)
                    gen_respose_dicts.append(step_selection_response)
                    step_selection_response = step_selection_response['response'][0]
                    
                    # Extract the selected step from the samples: the last instance of {X} in the response
                    selected_step = re.findall(r"\{([A-Z])\}", step_selection_response)
                    # If none found, use A
                    selected_step = selected_step[-1] if selected_step else "A"
                    # Extract the selected step from], by converting slection from A,B,C... to 0,1,2...
                    # If the selected step is not between 0 and len(step_samples_lst), use 0
                    selected_step_idx = ord(selected_step)-65
                    selected_step_idx = selected_step_idx if selected_step_idx < len(step_samples_lst) else 0
                    text_selected_step = step_samples_lst[selected_step_idx]
                else:
                    text_selected_step = step_samples_lst[0]

                format_step = text_selected_step.replace('\\n', ' ')
                text_previous_steps += f"{format_step}\n"


        text_cot = f"{text_problem_statement}\n{text_goal}### Steps ###\n{text_previous_steps}"
        answer_response = (await self.models["answer_extractor"].generate_async(text_cot))
        gen_respose_dicts.append(answer_response)
        answer_response = answer_response['response'][0]

        print(f"\rDone example {example_num}", end="")
        sys.stdout.flush()

        return {
            "cot_responses": [text_cot],
            "answers": [clean_answer(answer_response, self.task_name)],
            "query_details": gen_respose_dicts,
        }
            
            
            
