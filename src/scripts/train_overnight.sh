#!/bin/bash
# Simple overnight training script for LoRA fine-tuning

# Exit immediately if a command fails
set -e

# Activate your virtual environment
source .venv/bin/activate

# Run LoRA training, log both stdout and stderr to a file
nohup python src/lora_train.py > training.log 2>&1 &

echo "Training started in the background. Logs are being written to training.log"
echo "Check progress with: tail -f training.log"
