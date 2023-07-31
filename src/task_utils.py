import json
import os
import csv


# Task specific prompt to generate the correct answer string using the CoT solution
def get_task_answer_prompt(task_name):
    if task_name == "gsm8k":
        return "Respond with a single value that is the answer to the problem. Do not explain your answer or include symbols"
    elif task_name[: len("tracking_shuffled_objects")] == "tracking_shuffled_objects":
        return "Respond with the correct completion of the problem statement. Do not include names. Give only the entity that completes the final sentence."
    elif task_name[: len("coinflip")] == "coinflip":
        return "Respond with the final state of the coin. Do not explain your answer only use heads or tails"
    elif task_name == "strategyqa":
        return "Respond with the answer to the question. Do not explain your answer only use yes or no"
    elif task_name == "prontoqa":
        return "Respond with the answer to the question. Do not explain your answer only use true or false"
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")


# Task data load
def load_task(task_name, task_dir):
    if task_name == "gsm8k":
        questions, answers = load_gsm8k("test.jsonl", filedir=task_dir)
    elif (
        task_name == "strategyqa"
        or task_name[: len("tracking_shuffled_objects")] == "tracking_shuffled_objects"
    ):
        questions, answers = load_bigbench("task.json", filedir=task_dir)
    elif task_name[: len("coinflip")] == "coinflip":
        questions, answers = load_coinflip("task.csv", filedir=task_dir)
    elif task_name == "prontoqa":
        questions, answers = load_prontoqa("345hop_random_true.json", filedir=task_dir)
    else:
        raise ValueError(f"Unknown benchmark name: {task_name}")

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