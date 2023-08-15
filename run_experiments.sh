#!/bin/bash
taskChoices=(
    "gsm8k"
    # "tracking_shuffled_objects/three_objects"
    # "tracking_shuffled_objects/five_objects"
    # "tracking_shuffled_objects/seven_objects"
    # "coinflip/eight"
    # "strategyqa"
    # "prontoqa"
    # "lsat-ar"
    # "logiqa-en"
)

numExamples=5
asyncConcurr=1

# Uncomment the below line to enable overwriting results
overwriteResults="--overwrite_results"
# overwriteResults=""

runIdentifier="test_HF"
modelDefnsFiles=(
    "PromptWithAnswerExtraction/llama7b_cot_instruct"
    # "GoalExtraction/gpt35_goal_approach"
    # "GoalExtraction/gpt35_goal_approach_sbs"
    # "PromptWithAnswerExtraction/gpt35_cot_instruct"
    # "GoalExtraction/gpt35_goal_answertype"
)

for task in "${taskChoices[@]}"; do
    for modelDefns in "${modelDefnsFiles[@]}"; do
        command="python run.py --task_name $task --model_defns_file $modelDefns  --run_identifier $runIdentifier  --num_examples $numExamples --async_concurr $asyncConcurr $overwriteResults"
        echo $command
        $command
    done
done
