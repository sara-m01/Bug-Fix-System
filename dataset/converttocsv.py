import json
import pandas as pd


with open("balanced_dataset.json", "r") as f:
    data = json.load(f)


rows = []
for item in data:
    code = item["code"]
    label = str(item["label"]) 
    bug_type = item["bug_type"]
    bug_severity = item["bug_severity"]
    error_line = str(item["error_line"])
    
    input_text = f"Language: {item['language']}\nCode:\n{code}\nError Line: {error_line}"
    target_text = f"Label: {label}\nBug Type: {bug_type}\nSeverity: {bug_severity}"

    rows.append({"input": input_text, "target": target_text})


df = pd.DataFrame(rows)
df.to_csv("autotrain_dataset.csv", index=False)
