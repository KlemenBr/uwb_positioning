#!/bin/bash

# exit when any command fails
set -e

# fix raw data
python3 ./preprocess/fix_position_data.py
python3 ./preprocess/fix_outliers.py
rm -r ./data_set/raw_data/location0/data_position
rm -r ./data_set/raw_data/location1/data_position
rm -r ./data_set/raw_data/location2/data_position
rm -r ./data_set/raw_data/location3/data_position
python3 ./preprocess/fix_nlos_type.py
rm -r ./data_set/raw_data/location0/data_outliers
rm -r ./data_set/raw_data/location1/data_outliers
rm -r ./data_set/raw_data/location2/data_outliers
rm -r ./data_set/raw_data/location3/data_outliers
python3 ./preprocess/fix_offset.py
rm -r ./data_set/raw_data/location0/data_nlos
rm -r ./data_set/raw_data/location1/data_nlos
rm -r ./data_set/raw_data/location2/data_nlos
rm -r ./data_set/raw_data/location3/data_nlos
python3 ./preprocess/range_error_A6.py
python3 ./preprocess/fix_final_loc2_A6.py
rm -r ./data_set/raw_data/location0/data_offset
rm -r ./data_set/raw_data/location1/data_offset
rm -r ./data_set/raw_data/location2/data_offset
rm -r ./data_set/raw_data/location3/data_offset

# create JSON files and move them to data_set dir
python3 ./preprocess/save_raw_data_to_json.py
mv location0.json ./data_set/location0/data.json
mv location1.json ./data_set/location1/data.json
mv location2.json ./data_set/location2/data.json
mv location3.json ./data_set/location3/data.json

# remove data_final folders
rm -r ./data_set/raw_data/location0/data_final
rm -r ./data_set/raw_data/location1/data_final
rm -r ./data_set/raw_data/location2/data_final
rm -r ./data_set/raw_data/location3/data_final

