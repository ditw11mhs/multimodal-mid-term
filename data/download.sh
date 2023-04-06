#!/bin/bash

# Download the file and save it as data.zip
wget -O data.zip "https://zenodo.org/record/3431873/files/CHAOS_Train_Sets.zip?download=1"

# Unzip the file
unzip data.zip

# Remove the downloaded zip file
rm data.zip
