#!/bin/bash

BASEDIR=$(dirname "$0")

DEFAULT_FLARG_FILE=$BASEDIR/../federated_learning/arguments.default
FLARG_FILE=$BASEDIR/../federated_learning/arguments.py

OUT_DIR=$BASEDIR/../out

if [ -f "$FLARG_FILE" ]; then
    echo "FL args file already exist"
else 
    cp $DEFAULT_FLARG_FILE $FLARG_FILE
fi

if [ -d "$OUT_DIR" ]; then
    echo "out dir already exist"
else 
    mkdir $OUT_DIR
fi