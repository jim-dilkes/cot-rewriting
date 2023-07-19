$taskChoices =  "gsm8k"
            # "gsm8k",
            #    "tracking_shuffled_objects/three_objects",
            #    "tracking_shuffled_objects/five_objects",
            #    "tracking_shuffled_objects/seven_objects",
            #    "coinflip/four",
            #    "strategyqa",
            #    "prontoqa"

$modelChoices = "gpt-3.5-turbo"

$promptChoices = "None",
                  "CoT",
                 "CoT-WS"
# $promptChoices = "None"

$systemMessageChoices = "instruct-list",
                    "ChatGPT-default",
                    "instruct"
# $systemMessageChoices = "CoT-list"

$numExamples = "500"

$asyncConcurr = 5

foreach ($task in $taskChoices) {
    foreach ($model in $modelChoices) {
        foreach ($prompt in $promptChoices) {
            foreach ($systemMessage in $systemMessageChoices) {
                $command = "run.py --task_name $task --model_name $model --prompt_type $prompt --system_message_type $systemMessage --num_examples $numExamples --async_concurr $asyncConcurr"
                Write-Host $command
                python run.py --task_name $task --model_name $model --prompt_type $prompt --system_message_type $systemMessage --num_examples $numExamples --async_concurr $asyncConcurr
            }
        }
    }
}
