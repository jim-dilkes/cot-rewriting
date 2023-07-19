import json
import os
import csv


def load_gsm8k(filename, filedir='data/gsm8k'):
    file_path = os.path.join(filedir, filename)
    with open(file_path, 'r') as f:
        examples = [json.loads(l) for l in f]

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"])
        ex.update(numerical_answer=ex["answer"].split("####")[-1].strip().replace(",", ""))

    questions = [ex['question'] for ex in examples]
    answers = [ex['answer'] for ex in examples]

    print(f"{len(questions)} gsm8k examples loaded from {file_path}")
    return questions, answers


def load_bigbench(filename, filedir):
    file_path = os.path.join(filedir, filename)
    with open(file_path, 'r') as f:
        examples = json.load(f)['examples']

    questions = []
    answers = []
    for ex in examples:
        questions.append(ex['input'])
        for answer, value in ex['target_scores'].items():
            if value == 1:
                answers.append(answer)
                break
    
    print(f"{len(questions)} bigbench examples loaded from {file_path}")
    return questions, answers


def load_coinflip(filename, filedir):
    descriptions = []
    final_states = []
    
    file_path = os.path.join(filedir, filename)

    with open(file_path, 'r') as csvfile:
        for row in csv.DictReader(csvfile):
            descriptions.append(row['description'])
            final_states.append(row['final_state'])

    print(f"{len(descriptions)} coin flip examples loaded from {file_path}")
    return descriptions, final_states


def load_prontoqa(filename, filedir):
    questions = []
    answers = []
    
    file_path = os.path.join(filedir, filename)
    with open(file_path, 'r') as f:
        examples = json.load(f)

    for ex in examples.values():
        for sub_ex in ex.values():
            questions.append(f"{sub_ex['question']}\n{sub_ex['query']}")
            answers.append(sub_ex['answer'])
    
    print(f"{len(questions)} prontoqa examples loaded from {file_path}")
    return questions, answers