# Itactivul

## Directory Structure
- baselines: the implementation of baselines and scripts for running baselines
- dataset: NVD and bug report datasets
- models: BiLSTM_Att models

## Prepare Requirements
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## Train
train and test in NVD dataset

`python run.py --model TextRNN_Att --run train`

## Test
test in bug report datasets

`python run.py --model TextRNN_Att --data chromium.txt --run test`

## Run Baselines

```
  cd ./baselines
  
  python classification.py
```
