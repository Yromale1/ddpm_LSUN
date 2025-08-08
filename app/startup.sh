#!/bin/bash

set -e
MODEL_FILE="${MODEL_FILE:-checkpoint_epoch_50.pth}"
MODEL_DIR="models"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/Yromale/ddpm/resolve/main/${MODEL_FILE}"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "No token provided -> assuming the model is public"
        curl -L "$MODEL_URL" -o "$MODEL_PATH"
    else
        echo "Download model"
        curl -L -H "Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -o "$MODEL_PATH"
    fi
else
    echo "Model already downloaded"
fi

echo "Starting application"
python app.py