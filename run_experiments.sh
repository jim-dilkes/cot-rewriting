#!/bin/bash
taskChoices=(
    "gsm8k"
    # "tracking_shuffled_objects/three_objects"
    # "tracking_shuffled_objects/five_objects"
    # "tracking_shuffled_objects/seven_objects"
    # "coinflip/eight"
    # "strategyqa"
    "prontoqa"
    "lsat-ar"
    "logiqa-en"
)

numExamples=300
asyncConcurr=7

# Uncomment the below line to enable overwriting results
# overwriteResults="--overwrite_results"
overwriteResults=""

ambiguousIncorrect="--ambiguous_incorrect"
# ambiguousIncorrect=""

runIdentifier="stg4"
modelDefnsFiles=(
"PromptWithAnswerExtraction/gpt35_cot_sbs"
"PromptWithAnswerExtraction/gpt35_cot_instruct"
"GoalExtraction/gpt35_goal_approach_sbs"
"GoalExtraction/gpt35_goal_approach_instruct"
"GoalExtraction/gpt35_approach_sbs"
"GoalExtraction/gpt35_goal_sbs"
"GoalExtraction/gpt35_goal_approach_sbs_gapattern"
"GoalExtraction/gpt35_goal_approach_promptpattern"
"GoalExtraction/gpt35_goal_approach_bothpattern"
"SolveValidateRewrite/gpt35_all_instruct"
"SolveValidateRewrite/gpt35_all_pattern"
)

for task in "${taskChoices[@]}"; do
    for modelDefns in "${modelDefnsFiles[@]}"; do
        command="python run.py --task_name $task --model_defns_file $modelDefns  --run_identifier $runIdentifier  --num_examples $numExamples --async_concurr $asyncConcurr $overwriteResults $ambiguousIncorrect"
        echo $command
        $command
    done
done
