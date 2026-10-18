"""Build the controlled answer sets (dev = seed 1, test = seed 2)."""
import json
from cmbench.corrupt import build
s = json.load(open("data/snapshots.json"))
json.dump(build(s, seed=1), open("data/corruptions_dev.json", "w"), indent=1)
json.dump(build(s, seed=2), open("data/corruptions_test.json", "w"), indent=1)
print("ok")
