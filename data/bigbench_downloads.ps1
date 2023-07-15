# Code: gpt-4
# Define the repo and subdirs
$repo = "https://github.com/google/BIG-bench.git"
$subdirs = @(
    "bigbench/benchmark_tasks/strategyqa",
    "bigbench/benchmark_tasks/tracking_shuffled_objects"
)

# Define destination directories
$destDirs = @(
    "strategyqa",
    "tracking_shuffled_objects"
)

# Clone the repo into a temp directory
$tempDir = "BIG-bench-temp"
git clone --no-checkout $repo $tempDir

# Loop over each subdir and checkout
for ($i=0; $i -lt $subdirs.Length; $i++) {
    # Enable sparse checkout
    git -C $tempDir config core.sparsecheckout true

    # Define which directories you want to checkout
    $subdirs[$i] | Out-File -FilePath "$tempDir/.git/info/sparse-checkout" -Encoding ascii

    # Checkout
    git -C $tempDir checkout

    # Copy subdir to working directory
    Copy-Item -Path "$tempDir/$($subdirs[$i])" -Destination $destDirs[$i] -Recurse
}

# Cleanup temp directory
Remove-Item -Path $tempDir -Recurse -Force