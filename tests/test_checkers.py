import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmbench.evidence import Evidence
from cmbench.checkers import check_v0, check_v1, check_v1s

REC = [
 {"node":"Node-1","running_instances":0,"gpu_util_pct":0.0,"gpu_mem_gb":0.0,"failed_last_hour":0},
 {"node":"Node-2","running_instances":21,"gpu_util_pct":96.2,"gpu_mem_gb":3.8,"failed_last_hour":2},
 {"node":"Node-3","running_instances":4,"gpu_util_pct":602.4,"gpu_mem_gb":53.9,"failed_last_hour":0},
]
EV = Evidence(REC, 2702883)

def v(a, f=check_v1): return f(a, EV)["verdict"]

def test_v0_known_failures():
    # the four failure modes reproduced on the original code
    assert v("Node-1 looks fine.", check_v0) == "supported"
    assert v("Node-2 is at 97% with 85 GB and 21 instances.", check_v0) == "supported"
    assert v("Node-15 has 150% utilization.", check_v0) == "unsupported"  # bound to Node-1 by substring

def test_v1_no_claim_is_unverifiable():
    assert v("Node-1 looks fine.") == "unverifiable"

def test_v1_every_claim_must_hold():
    assert v("Node-2 has 96.2% GPU utilization and 3.8 GB of GPU memory.") == "supported"
    assert v("Node-2 has 96.2% GPU utilization and 85 GB of GPU memory.") == "unsupported"

def test_v1_metric_binding():
    assert v("Node-2 has 3.8% GPU utilization.") == "unsupported"   # right number, wrong metric

def test_v1_rounding_and_tolerance():
    assert v("Node-3 is at about 602% GPU utilization.") == "supported"
    assert v("Node-3 uses roughly 54 GB of GPU memory.") == "supported"
    assert v("Node-2 had 3 failed instances in the last hour.") == "unsupported"  # counts are exact

def test_v1_node_ids_are_whole_tokens():
    assert v("Node-13 has 0% GPU utilization.") == "unsupported"
    assert v("Node-1 has 0% GPU utilization.") == "supported"

def test_v1_abstention():
    assert v("Node-13 is not present in the telemetry.") == "abstain"
    assert v("Node-2 is not present in the telemetry.") == "unsupported"

def test_v1_timestamp_and_legend_not_claims():
    assert v("Node-2 at t=2702883s shows 96.2% (100% = one GPU).") == "supported"

def test_v1s_state_claims():
    assert v("Node-1 is idle.", check_v1s) == "supported"
    assert v("Node-2 is idle.", check_v1s) == "unsupported"
    assert v("Node-1 is busy.", check_v1s) == "unsupported"
    assert v("Node-2 is not idle.", check_v1s) == "supported"

from cmbench.checkers import check_v2

def test_v2_superlatives_and_targets():
    assert check_v2("Node-3 is the busiest with 602.4% GPU utilization.", EV)["verdict"] == "supported"
    assert check_v2("Node-2 is the busiest in terms of GPU usage.", EV)["verdict"] == "unsupported"
    assert check_v2("Node-1 has 0% GPU utilization.", EV, "What is the GPU utilization of Node-2?")["verdict"] == "unsupported"
    assert check_v2("Node-13 is not present in the telemetry.", EV, "What is the GPU utilization of Node-13?")["verdict"] == "supported"
    assert check_v2("Node-1 has 0% GPU utilization.", EV, "What is the GPU utilization of Node-13?")["verdict"] == "unsupported"
    assert check_v2("Node-1 is idle.", EV, "Which nodes are idle?")["verdict"] == "supported"
