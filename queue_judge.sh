#!/bin/bash
cd /home/claude/cm
while ! grep -q ALLDONE queue_gen.log; do sleep 60; done
python3 run_actions_llm.py qwen1.5b
python3 run_actions_llm.py llama1b
python3 run_judge.py qwen1.5b corrupt
python3 run_judge.py qwen1.5b gen:qwen1.5b_full_contract:0
python3 run_judge.py llama1b corrupt
python3 run_judge.py llama1b gen:llama1b_full_contract:0
python3 run_judge.py qwen7b corrupt
python3 run_judge.py qwen7b gen:qwen1.5b_full_contract:0
python3 run_judge.py qwen7b gen:llama1b_full_contract:0
echo JUDGEDONE
