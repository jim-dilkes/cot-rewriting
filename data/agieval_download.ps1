# Code: gpt-4
# Define the repo and subdirs
$repo = "https://github.com/microsoft/AGIEval.git"
$subdir =  "data/v1"

$tasks = @(
    "lsat-ar",
    "logiqa-en"
)

# Clone the repo into a temp directory
$tempDir = "AGIEval-temp"
$tempTaskDir = "$tempDir/$subdir"

git clone --no-checkout $repo $tempDir
# Enable sparse checkout
git -C $tempDir config core.sparsecheckout true
# Define which directories you want to checkout
"data/v1" | Out-File -FilePath "$tempDir/.git/info/sparse-checkout" -Encoding ascii
# Checkout
git -C $tempDir checkout

# Loop over each subdir and checkout
for ($i=0; $i -lt $tasks.Length; $i++) {
    # Destination path
    $destPath = "$($tasks[$i])"
    
    # Create destination directory if it doesn't exist
    if (!(Test-Path -Path $destPath)) {
        New-Item -ItemType Directory -Path $destPath
    }

    # Copy file to working directory
    Copy-Item -Path "$tempTaskDir/$($tasks[$i]).jsonl" -Destination "$destPath/task.jsonl"
}

# Cleanup temp directory
Remove-Item -Path $tempDir -Recurse -Force