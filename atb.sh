#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_runs> <number_of_clusters>"
    exit 1
fi

# Number of runs
NUM_RUNS=$1
NUM_CLUSTERS=$2

# Array of MAX_ITERATIONS values
MAX_ITERATIONS_VALUES=(1000 10000 100000 1000000)

# Output file for results
OUTPUT_FILE="results/results.txt"

# Clear the output file
> $OUTPUT_FILE

# Clear the results folder
rm -rf results/*

# Compile the sequential and parallel k-means code
mkdir -p build
gcc -o build/seq src/seq.c
mpicc -o build/par src/par.c

# Function to run the sequential k-means code
run_sequential() {
    local max_iterations=$1
    local clusters=$2
    local total_time=0

    for ((i=0; i<NUM_RUNS; i++)); do
        start_time=$(date +%s%N)
        build/seq $clusters $max_iterations > temp_output.txt
        exec_time=$(grep "Execution time" temp_output.txt | awk '{print $3}')
        total_time=$(echo "$total_time + $exec_time" | bc -l)
    done

    avg_time=$(echo "$total_time / $NUM_RUNS" | bc -l)
    echo "Sequential, $max_iterations, $avg_time" >> $OUTPUT_FILE
}

# Function to run the parallel (MPI) k-means code
run_parallel() {
    local max_iterations=$1
    local clusters=$2
    local total_time=0

    for ((i=0; i<NUM_RUNS; i++)); do
        start_time=$(date +%s%N)
        mpirun -n 4 build/par 4 $clusters $max_iterations > temp_output.txt
        exec_time=$(grep "Execution time" temp_output.txt | awk '{print $3}')
        total_time=$(echo "$total_time + $exec_time" | bc -l)
    done

    avg_time=$(echo "$total_time / $NUM_RUNS" | bc -l)
    echo "Parallel, $max_iterations, $avg_time" >> $OUTPUT_FILE
}

# Run the k-means code for each value of MAX_ITERATIONS
for max_iterations in "${MAX_ITERATIONS_VALUES[@]}"; do
    run_sequential $max_iterations $NUM_CLUSTERS
    run_parallel $max_iterations $NUM_CLUSTERS
done

rm temp_output.txt

# Generate the graph
python3 generate_graph.py $OUTPUT_FILE
