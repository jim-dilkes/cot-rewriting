$taskChoices =  
            #"tracking_shuffled_objects/three_objects",
            # "gsm8k"
            #    "tracking_shuffled_objects/three_objects",
            #    "tracking_shuffled_objects/five_objects",
            #    "tracking_shuffled_objects/seven_objects",
            #    "coinflip/eight",
            #    "strategyqa",
               "prontoqa"
                "lsat-ar"
                # "logiqa-en"

$numExamples = 250
$asyncConcurr = 7

# $overwriteResults = "--overwrite_results"
$overwriteResults = ""

$runIdentifier = "stg3"
$modelDefnsFiles = 
"SolveValidateRewrite/gpt35_validate_pattern"
# "SolveValidateRewrite/gpt35_validate_rewrite_pattern"
# "GoalExtraction/gpt35_goal_approach",
# "GoalExtraction/gpt35_goal_approach_sbs",
# "PromptWithAnswerExtraction/gpt35_cot_instruct"
# "GoalExtraction/gpt35_goal_answertype"

foreach ($task in $taskChoices) {
    foreach ($modelDefns in $modelDefnsFiles) {
        $command = "python run.py --task_name $task --model_defns_file $modelDefns  --run_identifier $runIdentifier  --num_examples $numExamples --async_concurr $asyncConcurr $overwriteResults"
        Write-Host $command
        Invoke-Expression $command
    }
}
