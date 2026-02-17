# run_pipeline.py
import yaml
import json
from orchestrator import generate_snapshot

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

snapshot = generate_snapshot(config)
print("\nGenerated JSON preview:")
print(json.dumps(snapshot, indent=2)[:1000] + "...")  # preview