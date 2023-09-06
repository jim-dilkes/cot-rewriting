import json
import os
import csv

AGIEVAL_TASKS = ['gaokao-physics','logiqa-en','lsat-ar']

# Task specific prompt to generate the correct answer string using the CoT solution
def get_task_answer_prompt(task_name, ambiguous_incorrect=False):
    if ambiguous_incorrect: # For ambiguous responses, prompt for an answer that will be marked as incorrect
        if task_name == "gsm8k":
            return "Respond with the given single value that is the answer to the problem. Do not explain your answer or include symbols. If there is no answer or multiple answers respond with NA."
        elif task_name.startswith("logical_deduction"):
            return "Respond with exactly the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E. If there is no answer or multiple answers respond with NA. A/B/C/D/E/NA:"
        elif task_name.startswith("tracking_shuffled_objects"):
            if task_name.endswith("multi"):
                return "Respond with exactly the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E. If there is no answer or multiple answers respond with NA. A/B/C/D/E/NA:"
            else:
                return "Respond with the correct given completion of the problem statement. Do not include names. Give only the entity that completes the final sentence. If the answer is ambiguous, respond with NA."
        elif task_name.startswith("coinflip"):
            return "Respond with the given final state of the coin. Do not explain your answer only use heads or tails. If there is no answer or multiple answers respond with NA."
        elif task_name == "strategyqa":
            return "Respond with the given answer to the question. Do not explain your answer only use yes or no. If there is no answer or multiple answers respond with NA."
        elif task_name == "prontoqa":
            return "Extract the answer from the response. Do not explain your answer only use True or False. Respond with NA if an inconclusive answer is given. True/False:"
        elif task_name in AGIEVAL_TASKS:
            return "Respond with the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E. If there is no answer or multiple answers respond with F."
        elif task_name == "aqua-rat":
            return "Respond with the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E. If there is no answer or multiple answers respond with NA."
        elif task_name == "navigate":
            return "Respond \{yes\} if you do return to the starting. Respond \{no\} if you do not return to the starting point. If there is no answer or multiple answers respond with X."
        else:
            raise ValueError(f"Unknown benchmark name: {task_name}")
    else: 
        if task_name == "gsm8k":
            return "Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols."
        elif task_name.startswith("logical_deduction"):
            return "Respond with exactly the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E."
        elif task_name.startswith("tracking_shuffled_objects"):
            if task_name.endswith("multi"):
                return "Respond with exactly the given multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E."
            else:
                return "Respond with the correct completion of the problem statement. Do not include names. Give only the entity that completes the final sentence."
        elif task_name.startswith("coinflip"):
            return "Respond with the final state of the coin. Do not explain your answer only use heads or tails."
        elif task_name == "strategyqa":
            return "Respond with the answer to the question. Do not explain your answer only use yes or no."
        elif task_name == "prontoqa":
            return "Respond with the answer to the question. Do not explain your answer only use true or false."
        elif task_name in AGIEVAL_TASKS:
            return "Respond with the multiple choice answer to the question. Do not explain your answer only use A/B/C/D/E."
        elif task_name == "aqua-rat":
            return "Respond with the given multiple choice answer to the question. Do not explain your answer only use yes or no."
        elif task_name == "navigate":
            return "Respond {yes} if you do return to the starting. Respond {no} if you do not return to the starting point."
        else:
            raise ValueError(f"Unknown benchmark name: {task_name}")
        

# Task data load
def load_task(task_name, task_dir):
    if task_name == "gsm8k":
        return load_gsm8k("test.jsonl", task_dir)
    elif (
        task_name in ("strategyqa")
        or task_name.startswith("tracking_shuffled_objects")
        or task_name.startswith("logical_deduction")
    ):
        if not task_name.endswith("_multi"):
            return load_bigbench("task.json", task_dir)
        return load_bigbench_multichoice("task.json", task_dir[:-len("_multi")])
    elif task_name == "navigate":
        return load_navigate("task.json", task_dir)
    elif task_name[: len("coinflip")] == "coinflip":
        return load_coinflip("task.csv", task_dir)
    elif task_name == "prontoqa":
        return load_prontoqa("345hop_random_true.json", task_dir)
    elif task_name in AGIEVAL_TASKS:
        return load_agieval("task.jsonl", task_dir)
    elif task_name == "aqua-rat":
        return load_aqua_rat("test.json", task_dir)
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")
    

def load_aqua_rat(filename, filedir):
    file_path = os.path.join(filedir, filename)
    with open(file_path, "r") as f:
        examples = [json.loads(l) for l in f]
        
    questions = []
    answers = []
    for ex in examples:
        options = " ".join([f"({o[0]}){o[2:]}" for o in ex['options']])
        
        questions.append(f"{ex['question']} {options}")
        answers.append(ex['correct'])
        
    print(f"{len(questions)} aqua-rat examples loaded from {file_path}")
    return questions, answers
        

def load_gsm8k(filename, filedir="data/gsm8k"):
    file_path = os.path.join(filedir, filename)
    with open(file_path, "r") as f:
        examples = [json.loads(l) for l in f]

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"])
        ex.update(
            numerical_answer=ex["answer"].split("####")[-1].strip().replace(",", "")
        )

    questions = [ex["question"] for ex in examples]
    answers = [ex["numerical_answer"] for ex in examples]

    print(f"{len(questions)} gsm8k examples loaded from {file_path}")
    return questions, answers

def load_navigate(filename, filedir):
    questions, answers = load_bigbench(filename, filedir)
    
    task_prefix = "If you follow these instructions, do you return to the starting point?"
    questions = [f"{task_prefix} {q} Do you return to the starting point (yes/no)?" for q in questions]
    answers = [f"{'yes' if str(a)=='True' else 'no'}" for a in answers]
    
    return questions, answers
    

def load_bigbench(filename, filedir):
    file_path = os.path.join(filedir, filename)
    with open(file_path, "r") as f:
        examples = json.load(f)["examples"]

    questions = []
    answers = []
    for ex in examples:
        questions.append(ex["input"])
        for answer, value in ex["target_scores"].items():
            if value == 1:
                answers.append(answer)
                break

    print(f"{len(questions)} bigbench examples loaded from {file_path}")
    return questions, answers

def load_bigbench_multichoice(filename, filedir):
    file_path = os.path.join(filedir, filename)
    with open(file_path, "r") as f:
        examples = json.load(f)["examples"]

    questions = []
    answers = []
    for ex in examples:
        options_str = '\n'.join(f"{chr(65 + i)}) {option}" for i, option in enumerate(ex['target_scores'].keys()))
        question_string = f"{ex['input']}\n\n{options_str}"
        questions.append(question_string)
        correct_option_index = [i for i, score in enumerate(ex['target_scores'].values()) if score == 1][0]
        correct_option_letter = chr(65 + correct_option_index)
        answers.append(correct_option_letter)

    print(f"{len(questions)} bigbench examples loaded from {file_path}")
    return questions, answers


def load_coinflip(filename, filedir):
    descriptions = []
    final_states = []

    file_path = os.path.join(filedir, filename)

    with open(file_path, "r") as csvfile:
        for row in csv.DictReader(csvfile):
            descriptions.append(row["description"])
            final_states.append(row["final_state"])

    print(f"{len(descriptions)} coin flip examples loaded from {file_path}")
    return descriptions, final_states


def load_prontoqa(filename, filedir):
    questions = []
    answers = []

    file_path = os.path.join(filedir, filename)
    with open(file_path, "r") as f:
        examples = json.load(f)

    for ex in examples.values():
        for sub_ex in ex.values():
            questions.append(f"{sub_ex['question']}\n{sub_ex['query']}")
            answers.append(sub_ex["answer"])

    print(f"{len(questions)} prontoqa examples loaded from {file_path}")
    return questions, answers


def load_agieval(filename, filedir):
    file_path = os.path.join(filedir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]

    for ex in examples:
        ex.update(question=ex["passage"] + "\n\n" + ex["question"] + "\n\n" + "\n".join(ex['options']))
        ex.update(answer=ex["label"])

    questions = [ex["question"] for ex in examples]
    answers = [ex["answer"] for ex in examples]

    print(f"{len(questions)} AGIEval examples loaded from {file_path}")
    return questions, answers
