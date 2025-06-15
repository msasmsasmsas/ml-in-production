#!/bin/bash

# Script to archive the PyTorch model for TorchServe

set -e

EXTRA_FILES="model/imagenet_classes.json"
MODEL_NAME="resnet50"
VERSION="1.0"
MODEL_FILE="model/model.py"
HANDLER="model/handler.py"
OUTPUT_DIR="/home/model-server/model-store"

echo "Archiving model: ${MODEL_NAME}"

torch-model-archiver \
    --model-name ${MODEL_NAME} \
    --version ${VERSION} \
    --model-file ${MODEL_FILE} \
    --handler ${HANDLER} \
    --extra-files ${EXTRA_FILES} \
    --export-path ${OUTPUT_DIR}

echo "Model archived successfully: ${OUTPUT_DIR}/${MODEL_NAME}.mar"
