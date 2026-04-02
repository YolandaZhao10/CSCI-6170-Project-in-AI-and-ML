#!/bin/bash

# tune.sh
# Automates a systematic nested sweep of method-specific hyperparameters
# without combinatorial explosion across non-applicable arguments.

SOURCES=("MNIST" "USPS")
EPOCHS=10
LRS=("1e-3")

mkdir -p logs/tuning

echo "Starting Systematic Hyperparameter Sweep across domain pairs..."

for source in "${SOURCES[@]}"; do
    if [ "$source" == "MNIST" ]; then
        TARGETS="USPS SVHN"
    elif [ "$source" == "USPS" ]; then
        TARGETS="MNIST SVHN"
    fi

    # 1. Baseline Sweep
    for lr in "${LRS[@]}"; do
        for bs in 64 128; do
            run_name="baseline_src${source}_lr${lr}_bs${bs}"
            echo "--------------------------------------------------------"
            echo "Tuning -> $run_name"
            ./.venv/bin/python -u main.py --method "baseline" --source "$source" --targets $TARGETS \
                --epochs "$EPOCHS" --lr "$lr" --batch_size "$bs" > "logs/tuning/${run_name}.log" 2>&1
        done
    done

    # 2. CORAL Sweep
    for lr in "${LRS[@]}"; do
        for penalty in 0.1 1.0 10.0; do
            run_name="coral_src${source}_lr${lr}_penalty${penalty}"
            echo "--------------------------------------------------------"
            echo "Tuning -> $run_name"
            ./.venv/bin/python -u main.py --method "coral" --source "$source" --targets $TARGETS \
                --epochs "$EPOCHS" --lr "$lr" --penalty_weight "$penalty" > "logs/tuning/${run_name}.log" 2>&1
        done
    done

    # 3. DANN Sweep
    for lr in "${LRS[@]}"; do
        for penalty in 0.1 1.0; do
            for gamma in 5.0 10.0; do
                run_name="dann_src${source}_lr${lr}_penalty${penalty}_gamma${gamma}"
                echo "--------------------------------------------------------"
                echo "Tuning -> $run_name"
                ./.venv/bin/python -u main.py --method "dann" --source "$source" --targets $TARGETS \
                    --epochs "$EPOCHS" --lr "$lr" --penalty_weight "$penalty" --dann_gamma "$gamma" > "logs/tuning/${run_name}.log" 2>&1
            done
        done
    done

    # 4. IRM Sweep
    for lr in "${LRS[@]}"; do
        for penalty in 1.0 10.0 100.0; do
            for warmup in 0 2; do
                run_name="irm_src${source}_lr${lr}_penalty${penalty}_warmup${warmup}"
                echo "--------------------------------------------------------"
                echo "Tuning -> $run_name"
                ./.venv/bin/python -u main.py --method "irm" --source "$source" --targets $TARGETS \
                    --epochs "$EPOCHS" --lr "$lr" --penalty_weight "$penalty" --irm_warmup_epochs "$warmup" > "logs/tuning/${run_name}.log" 2>&1
            done
        done
    done

    # 5. MC Dropout Sweep
    for lr in "${LRS[@]}"; do
        for dropout in 0.2 0.5 0.7; do
            for passes in 5 10; do
                run_name="mcdropout_src${source}_lr${lr}_drop${dropout}_passes${passes}"
                echo "--------------------------------------------------------"
                echo "Tuning -> $run_name"
                ./.venv/bin/python -u main.py --method "mc_dropout" --source "$source" --targets $TARGETS \
                    --epochs "$EPOCHS" --lr "$lr" --dropout_p "$dropout" --mc_passes "$passes" > "logs/tuning/${run_name}.log" 2>&1
            done
        done
    done
done

echo "Tuning Complete. Check configs in logs/tuning/ directory!"
