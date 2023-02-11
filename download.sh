#!/bin/bash

# download data set if not present
FILE=data_set.zip
if [ ! -f "$FILE" ]; then
    echo "$FILE does not exist --> DOWNLOADING"
    zenodo_get 10.5281/zenodo.7629141
fi

# extract if data_set folder not present
FOLDER=./data_set
if [ ! -d "$FOLDER" ]; then
    echo "$FOLDER not present --> EXTRACTING"
    unzip data_set.zip
fi






