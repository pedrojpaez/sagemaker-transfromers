#! /bin/sh
 
export SM_NUM_GPUS="1"
echo 'export SM_NUM_GPUS="1"' >> ~/.bashrc
 
export SM_NUM_CPUS="8"
echo 'export SM_NUM_CPUS="8"' >> ~/.bashrc

export SM_OUTPUT_DATA_DIR="output/data"
echo 'export SM_OUTPUT_DATA_DIR="output/data"' >> ~/.bashrc

export SM_MODEL_DIR="output/model"
echo 'export SM_MODEL_DIR="output/model"' >> ~/.bashrc

export SM_CHANNEL_TRAIN="data/train"
echo 'export SM_CHANNEL_TRAIN="data/train"' >> ~/.bashrc

export SM_CHANNEL_TEST="data/test"
echo 'export SM_CHANNEL_TEST="data/test"' >> ~/.bashrc

