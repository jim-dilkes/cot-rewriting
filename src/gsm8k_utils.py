import json

def load_gsm8k(filename, filedir='data/gsm8k'):
    with open(f'{filedir}/{filename}') as f:
        examples = [json.loads(l) for l in f]

    for i, ex in enumerate(examples):
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"])
        ex.update(numerical_answer=ex["answer"].split("####")[-1].strip().replace(",", ""))
        ex.update(index=i)

    print(f"{len(examples)} gsm8k examples loaded from {filename}")
    return examples
