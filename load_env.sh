#!/bin/bash

# Check if .env file exists
if [ -f .env ]; then
    # Load environment variables from .env file
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded from .env file"
else
    echo "Warning: .env file not found. Please create one using .env.template as a guide"
    exit 1
fi

# Verify required variables are set
required_vars=("WANDB_API_KEY" "HF_TOKEN")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "Error: The following required environment variables are missing:"
    printf '%s\n' "${missing_vars[@]}"
    echo "Please set them in your .env file"
    exit 1
fi

echo "All required environment variables are set" 