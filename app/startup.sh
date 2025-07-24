#!/bin/bash

set -e
MODEL_FILE="${MODEL_FILE:-checkpoint_epoch_100.pth}"
MODEL_DIR="models"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/<username>/<repo>/resolve/main/${MODEL_PATH}"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "No rights to access the model"
        exit 1
    else
        echo "Download model"
        curl -L -H "Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -o "$MODEL_PATH"
    fi
else
    echo "Model already downloaded"
fi

echo "Starting application"
python app.py