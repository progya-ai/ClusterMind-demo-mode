"""Convert the extracted machine-subset CSVs to parquet (column names from the official README)."""
import sys, os, pandas as pd
os.makedirs("data", exist_ok=True)
I = pd.read_csv(sys.argv[1], header=None, names="job_name task_name inst_name worker_name inst_id status start_time end_time machine".split())
S = pd.read_csv(sys.argv[2], header=None, names="job_name task_name worker_name inst_id machine gpu_name cpu_usage gpu_wrk_util avg_mem max_mem avg_gpu_wrk_mem max_gpu_wrk_mem read write read_count write_count".split())
I.to_parquet("data/instance.parquet"); S.to_parquet("data/sensor.parquet")
print(len(I), len(S))
