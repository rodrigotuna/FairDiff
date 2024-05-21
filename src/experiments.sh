#!/bin/bash
## CORA
python main.py dataset=cora
python main.py +experiment=cora_fair
python main.py +experiment=cora_focal
python main.py +experiment=cora_fair_focal

## FB
python main.py dataset=facebook
python main.py +experiment=fb_fair
python main.py +experiment=fb_focal
python main.py +experiment=fb_fair_focal

## NBA
python main.py dataset=nba
python main.py +experiment=nba_fair
python main.py +experiment=nba_focal
python main.py +experiment=nba_fair_focal

##OK97
python main.py dataset=oklahoma97
python main.py +experiment=ok97_fair
python main.py +experiment=ok97_focal
python main.py +experiment=ok97_fair_focal

## UNC28
python main.py dataset=unc28
python main.py +experiment=un28_fair
python main.py +experiment=un28_focal
python main.py +experiment=un28_fair_focal
