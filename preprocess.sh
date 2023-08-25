#!/bin/bash

# exit when any command fails
set -e

# fix raw data
python3 ./preprocess/fix_position_data.py
python3 ./preprocess/fix_outliers.py
rm -r ./data_set/raw_data/environment0/data_position
rm -r ./data_set/raw_data/environment1/data_position
rm -r ./data_set/raw_data/environment2/data_position
rm -r ./data_set/raw_data/environment3/data_position
python3 ./preprocess/fix_nlos_type.py
rm -r ./data_set/raw_data/environment0/data_outliers
rm -r ./data_set/raw_data/environment1/data_outliers
rm -r ./data_set/raw_data/environment2/data_outliers
rm -r ./data_set/raw_data/environment3/data_outliers
python3 ./preprocess/fix_offset.py
rm -r ./data_set/raw_data/environment0/data_nlos
rm -r ./data_set/raw_data/environment1/data_nlos
rm -r ./data_set/raw_data/environment2/data_nlos
rm -r ./data_set/raw_data/environment3/data_nlos
python3 ./preprocess/range_error_A6.py
python3 ./preprocess/fix_final_loc2_A6.py
rm -r ./data_set/raw_data/environment0/data_offset
rm -r ./data_set/raw_data/environment1/data_offset
rm -r ./data_set/raw_data/environment2/data_offset
rm -r ./data_set/raw_data/environment3/data_offset

# create JSON files and move them to data_set dir
python3 ./preprocess/save_raw_data_to_json.py
mv environment0.json ./data_set/environment0/data.json
mv environment1.json ./data_set/environment1/data.json
mv environment2.json ./data_set/environment2/data.json
mv environment3.json ./data_set/environment3/data.json

# remove data_final folders
rm -r ./data_set/raw_data/environment0/data_final
rm -r ./data_set/raw_data/environment1/data_final
rm -r ./data_set/raw_data/environment2/data_final
rm -r ./data_set/raw_data/environment3/data_final

