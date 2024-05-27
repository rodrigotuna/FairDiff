#!/bin/bash
## CORA
python main.py train=train_vm dataset=cora
python main.py train=train_vm +experiment=cora_fair
#python main.py train=train_vm +experiment=cora_focal
python main.py train=train_vm +experiment=cora_fair_focal

## FB
#python main.py train=train_vm dataset=facebook
python main.py train=train_vm +experiment=fb_fair
#python main.py train=train_vm +experiment=fb_focal
python main.py train=train_vm +experiment=fb_fair_focal

## NBA
#python main.py train=train_vm dataset=nba
python main.py train=train_vm +experiment=nba_fair
#python main.py train=train_vm +experiment=nba_focal
python main.py train=train_vm +experiment=nba_fair_focal

##OK97
python main.py train=train_vm dataset=oklahoma97
python main.py train=train_vm +experiment=ok97_fair
python main.py train=train_vm +experiment=ok97_focal
python main.py train=train_vm +experiment=ok97_fair_focal

## UNC28
python main.py train=train_vm dataset=unc28
python main.py train=train_vm +experiment=un28_fair
python main.py train=train_vm +experiment=un28_focal
python main.py train=train_vm +experiment=un28_fair_focal
