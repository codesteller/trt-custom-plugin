#!/bin/bash

# STEPS TO RUN
# Download dataset from https://www.kaggle.com/biaiscience/dogs-vs-cats
# then copy this script and make_validation.py to that folder and run

unzip dogs-vs-cats.zip -d dogs_cats
data_dir="dogs_cats/train/"
cd data_dir

mkdir -p "valid/cats"
mkdir -p "valid/dogs"
mkdir -p "train/cats"
mkdir -p "train/dogs"

mv train/cat.* cats/
mv train/dog.* dogs/

python make_validation.py