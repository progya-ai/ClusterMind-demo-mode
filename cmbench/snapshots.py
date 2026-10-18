"""
Cluster snapshot construction from Alibaba cluster-trace-gpu-v2020
(pai_instance_table + pai_sensor_table).

A snapshot is the typed state of a small set of machines at a trace time t.
For each machine m we derive, from the instance table, the set of instances
active at t:  start_time <= t  and  (end_time > t  or  (end_time is NaN and
status == 'Running')).  Instances whose end_time is missing for any other
status are excluded (their lifetime is unknown).

Per-machine typed metrics (the ONLY metrics that appear in evidence):
  running_instances : count of active instances                     [count]
  gpu_util_pct      : sum of lifetime-average gpu_wrk_util over active
                      instances that have sensor rows (100 = one GPU)   [%]
  gpu_mem_gb        : sum of lifetime-average avg_gpu_wrk_mem over the
                      same instances                                   [GB]
  failed_last_hour  : instances on m with status 'Failed' and
                      end_time in (t-3600, t]                        [count]

Disclosed approximation: sensor metrics in v2020 are per-instance lifetime
averages, not time-resolved samples, so gpu_util_pct / gpu_mem_gb are
*attributed load* at t, not an instantaneous reading. There is no
temperature anywhere in the trace; we do not synthesize one.
"""
import numpy as np
import pandas as pd

DATA = "data"


def load_tables():
    I = pd.read_parquet(f"{DATA}/instance.parquet")
    S = pd.read_parquet(f"{DATA}/sensor.parquet")
    I = I.dropna(subset=["machine"])  # rows without start_time can still carry a failure end_time
    S = S[["worker_name", "gpu_wrk_util", "avg_gpu_wrk_mem"]].copy()
    S["gpu_wrk_util"] = S["gpu_wrk_util"].fillna(0.0)
    S["avg_gpu_wrk_mem"] = S["avg_gpu_wrk_mem"].fillna(0.0)
    I = I.merge(S, on="worker_name", how="left")
    return I


def active_mask(I, t):
    started = I["start_time"] <= t
    ended_after = I["end_time"] > t
    still_running = I["end_time"].isna() & (I["status"] == "Running")
    return started & (ended_after | still_running)


def machine_state(I, t, machines=None):
    """Typed state for every machine (or the given ones) at time t."""
    sub = I if machines is None else I[I["machine"].isin(machines)]
    act = sub[active_mask(sub, t)]
    g = act.groupby("machine")
    st = pd.DataFrame({
        "running_instances": g.size(),
        "gpu_util_pct": g["gpu_wrk_util"].sum(min_count=1),
        "gpu_mem_gb": g["avg_gpu_wrk_mem"].sum(min_count=1),
    })
    fail = sub[(sub["status"] == "Failed") & (sub["end_time"] > t - 3600) & (sub["end_time"] <= t)]
    st = st.join(fail.groupby("machine").size().rename("failed_last_hour"), how="outer")
    all_m = pd.Index(sub["machine"].unique() if machines is None else machines, name="machine")
    st = st.reindex(all_m)
    st["running_instances"] = st["running_instances"].fillna(0).astype(int)
    st["failed_last_hour"] = st["failed_last_hour"].fillna(0).astype(int)
    st["gpu_util_pct"] = st["gpu_util_pct"].fillna(0.0).round(1)
    st["gpu_mem_gb"] = st["gpu_mem_gb"].fillna(0.0).round(1)
    return st


def build_snapshots(I, n_snapshots=40, n_nodes=8, seed=0, t_lo=2.5e6, t_hi=6.4e6, frac_failure_anchored=0.5):
    """Stratified node sampling so every snapshot contains idle, busy-high-util,
    busy-low-util nodes and (where available) nodes with recent failures."""
    rng = np.random.default_rng(seed)
    machines_all = I["machine"].unique()
    # sensor rows exist only for t >= ~2.0e6 in this subset; half of the snapshot
    # times are anchored shortly after a real failure so failure questions are non-degenerate
    n_f = int(round(n_snapshots * frac_failure_anchored))
    fend = I.loc[(I["status"] == "Failed") & I["end_time"].between(t_lo, t_hi), "end_time"].values
    t_fail = rng.choice(fend, size=n_f, replace=False) + rng.uniform(60, 1800, size=n_f)
    t_unif = rng.uniform(t_lo, t_hi, size=n_snapshots - n_f)
    times = np.sort(np.concatenate([t_fail, t_unif])).round()
    snaps = []
    for si, t in enumerate(times):
        st = machine_state(I, t, machines_all)
        # only machines that have ever hosted work before t (i.e. exist)
        seen = I.loc[I["start_time"] <= t, "machine"].unique()
        st = st.loc[st.index.isin(seen)]
        idle = st[st.running_instances == 0].index.values
        busy_hi = st[(st.running_instances > 0) & (st.gpu_util_pct >= 50)].index.values
        busy_lo = st[(st.running_instances > 0) & (st.gpu_util_pct <= 10)].index.values
        failed = st[st.failed_last_hour > 0].index.values
        pick = []
        def take(pool, k):
            pool = [m for m in pool if m not in pick]
            if len(pool) == 0:
                return
            for m in rng.choice(pool, size=min(k, len(pool)), replace=False):
                pick.append(m)
        take(failed, 2); take(idle, 2); take(busy_hi, 2); take(busy_lo, 1)
        rest = [m for m in st.index.values if m not in pick]
        take(rest, n_nodes - len(pick))
        pick = list(rng.permutation(pick))
        rows = []
        for k, m in enumerate(pick, start=1):
            r = st.loc[m]
            rows.append({
                "node": f"Node-{k}", "machine": m,
                "running_instances": int(r.running_instances),
                "gpu_util_pct": float(r.gpu_util_pct),
                "gpu_mem_gb": float(r.gpu_mem_gb),
                "failed_last_hour": int(r.failed_last_hour),
            })
        snaps.append({"snapshot_id": f"S{si:03d}", "t": int(t), "nodes": rows})
    return snaps
