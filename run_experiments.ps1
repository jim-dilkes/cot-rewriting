$taskChoices =  
            # "gsm8k",
            #    "tracking_shuffled_objects/three_objects"
            #    "tracking_shuffled_objects/five_objects",
            #    "tracking_shuffled_objects/seven_objects",
            #    "coinflip/eight",
            #    "strategyqa",
               "prontoqa"
                # "lsat-ar",
                # "logiqa-en"

$numExamples = 300
$asyncConcurr = 6

$overwriteResults = "--overwrite_results"
# $overwriteResults = ""

$ambiguousIncorrect = "--ambiguous_incorrect"
# $ambiguousIncorrect = ""

$randomSeed = 1

$runIdentifier = "stg4"
$modelDefnsFiles = 
"PromptWithAnswerExtraction/gpt35_no_prompt"
# "PromptWithAnswerExtraction/gpt35_cot_sbs"
# "PromptWithAnswerExtraction/gpt35_cot_instruction"
# #
# # "GoalExtraction/gpt35_goal_approach_sbs"
# "GoalExtraction/gpt35_goal_approach_instruct"
# "GoalExtraction/gpt35_approach_sbs"
# "GoalExtraction/gpt35_goal_sbs"
# "GoalExtraction/gpt35_goal_approach_sbs_gapattern"
# "GoalExtraction/gpt35_goal_approach_promptpattern"
# "GoalExtraction/gpt35_goal_approach_bothpattern"
# #
# "SolveValidateRewrite/gpt35_all_instruct"
# "SolveValidateRewrite/gpt35_all_pattern"

foreach ($task in $taskChoices) {
    foreach ($modelDefns in $modelDefnsFiles) {
        $command = "python run.py --task_name $task --model_defns_file $modelDefns  --run_identifier $runIdentifier  --num_examples $numExamples --async_concurr $asyncConcurr --seed $randomSeed $overwriteResults $ambiguousIncorrect"
        Write-Host $command
        Invoke-Expression $command
    }
}
 