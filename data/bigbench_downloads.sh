#!/bin/bash

# Code: gpt4  (not tested)

# Define the repo and subdirs
repo="https://github.com/google/BIG-bench.git"
subdirs=("bigbench/benchmark_tasks/strategyqa" "bigbench/benchmark_tasks/tracking_shuffled_objects")

# Define destination directories
destDirs=("strategyqa" "tracking_shuffled_objects")

# Clone the repo into a temp directory
tempDir="BIG-bench-temp"
git clone --no-checkout $repo $tempDir

# Loop over each subdir and checkout
for index in ${!subdirs[*]}; do
    # Enable sparse checkout
    git -C $tempDir config core.sparseCheckout true

    # Define which directories you want to checkout
    echo ${subdirs[$index]} > "$tempDir/.git/info/sparse-checkout"

    # Checkout
    git -C $tempDir checkout

    # Copy subdir to working directory
    cp -R "$tempDir/${subdirs[$index]}" "${destDirs[$index]}"
done

# Cleanup temp directory
rm -rf $tempDir
