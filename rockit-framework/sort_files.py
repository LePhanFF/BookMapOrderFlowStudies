import os
import json

def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

directory = "data/json_snapshots/2025-12-17"
files = [f for f in os.listdir(directory) if f.endswith('.json')]

file_times = []
for f in files:
    path = os.path.join(directory, f)
    with open(path, 'r') as file:
        data = json.load(file)
        et_time = data['current_et_time']
        minutes = time_to_minutes(et_time)
        file_times.append((minutes, path))

file_times.sort(key=lambda x: x[0])

sorted_paths = [path for _, path in file_times]

for path in sorted_paths:
    print(path)