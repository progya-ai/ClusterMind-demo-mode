#!/bin/bash
cd /home/claude/cm
python3 run_generate.py qwen1.5b full contract 0,1,2
python3 run_generate.py llama1b full contract 0,1,2
python3 run_generate.py qwen1.5b tfidf contract 0
python3 run_generate.py llama1b tfidf contract 0
python3 run_generate.py qwen1.5b tfidf original 0
python3 run_generate.py qwen1.5b full original 0
python3 run_generate.py qwen7b full contract 0
echo ALLDONE
