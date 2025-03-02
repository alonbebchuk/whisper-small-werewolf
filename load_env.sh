#!/bin/bash

# Check if .env file exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables loaded from .env file"
else
    echo "Warning: .env file not found. Please create one using .env.template as a guide"
    exit 1
fi

# Verify required variables are set
missing_vars=""

for var in WANDB_API_KEY HF_TOKEN; do
    if [ -z "${!var}" ]; then
        missing_vars="$missing_vars $var"
    fi
done

if [ -n "$missing_vars" ]; then
    echo "Error: The following required environment variables are missing:"
    echo "$missing_vars"
    echo "Please set them in your .env file"
    exit 1
fi

echo "All required environment variables are set"
