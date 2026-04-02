#!/bin/bash

# run_experiments.sh
# Automates multi-seed, multi-domain generalization methods across all matrix pairs
# Settings evaluated: MNIST <-> USPS <-> SVHN

METHODS=("baseline" "target_supervised" "fine_tuning" "coral" "dann" "irm" "mc_dropout")
SOURCES=("MNIST" "USPS" "SVHN")
SEEDS=(42 123 2026)

EPOCHS=10
LR="1e-3"

mkdir -p logs

echo "Starting Paper-Quality Domain Generalization Full-Matrix Suite..."

for seed in "${SEEDS[@]}"; do
    echo "========================================================"
    echo "INITIALIZING SEED [ $seed ]"
    echo "========================================================"

    for source in "${SOURCES[@]}"; do
        if [ "$source" == "MNIST" ]; then
            TARGETS="USPS SVHN"
        elif [ "$source" == "USPS" ]; then
            TARGETS="MNIST SVHN"
        elif [ "$source" == "SVHN" ]; then
            TARGETS="MNIST USPS"
        fi
        
        echo "--------------------------------------------------------"
        echo "Starting Runs for SOURCE [ $source ] adapting to [ $TARGETS ]"
        
        for method in "${METHODS[@]}"; do
            run_id="${method}_${source}_seed${seed}"
            echo "Running -> $run_id"
            
            ./.venv/bin/python -u main.py --method "$method" --source "$source" --targets $TARGETS \
                --seed "$seed" --epochs "$EPOCHS" --lr "$LR" > "logs/${run_id}.log" 2>&1
        done
    done
done

echo "--------------------------------------------------------"
echo "All structured empirical experiments complete!"
echo "Run: python aggregate_results.py to view tabular Mean ± StdDev results."
