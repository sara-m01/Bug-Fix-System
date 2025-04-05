import json
import random
from collections import defaultdict
import numpy as np


def generate_balanced_dataset(target_samples_per_language=50):
    
    languages = list(CODE_SAMPLES.keys())
    samples_per_bug_type = max(1, target_samples_per_language // len(CODE_SAMPLES["python"]))

    dataset = []
    language_bug_counts = defaultdict(lambda: defaultdict(int))

    for language in languages:
        for sample in CODE_SAMPLES[language]:
            for _ in range(samples_per_bug_type):
                buggy_entry = {
                    "language": language,
                    "code": sample["code"],
                    "label": 1,
                    "error_line": sample["error_line"],
                    "bug_type": sample["bug_type"],
                    "fix": sample["fix"],
                    "bug_severity": BUG_SEVERITY[sample["bug_type"]],
                    "error_message": ERROR_MESSAGES[sample["bug_type"]].get(language, "Error"),
                    "output_behavior": OUTPUT_MESSAGES["before"][sample["bug_type"]]
                }
                dataset.append(buggy_entry)
                language_bug_counts[language][sample["bug_type"]] += 1

                
                fixed_entry = {
                    "language": language,
                    "code": sample["fix"],
                    "label": 0,
                    "error_line": -1,
                    "bug_type": "None",
                    "fix": "None",
                    "bug_severity": "None",
                    "error_message": "None",
                    "output_behavior": OUTPUT_MESSAGES["after"][sample["bug_type"]]
                }
                dataset.append(fixed_entry)

    
    print("\nDataset Balance Verification:")
    for lang in languages:
        print(f"\n{lang.upper()}:")
        for bug_type in CODE_SAMPLES[lang]:
            count = language_bug_counts[lang][bug_type["bug_type"]]
            print(f"  {bug_type['bug_type']}: {count} buggy + {count} clean samples")

  
    random.shuffle(dataset)
    return dataset

def generate_instruction_dataset(buggy_samples_only=True):
    
    base_dataset = generate_balanced_dataset(40)  

    instruction_dataset = []
    length_discrepancies = []

    for example in base_dataset:
        if not buggy_samples_only or example["label"] == 1:
           
            code_len = len(example["code"])
            fix_len = len(example["fix"]) if example["fix"] != "None" else 0
            length_ratio = fix_len / code_len if code_len > 0 else 1
            length_discrepancies.append(length_ratio)

            
            if length_ratio > 3:
                continue

            instruction_entry = {
                "instruction": "Analyze this code, identify any bugs, and provide a fixed version.",
                "input": f"Language: {example['language']}\nBug Type: {example['bug_type']}\nCode:\n{example['code']}",
                "output": (f"Bug Analysis:\n"
                          f"- Type: {example['bug_type']}\n"
                          f"- Severity: {example['bug_severity']}\n"
                          f"- Error: {example['error_message']}\n\n"
                          f"Fixed Code:\n{example['fix']}"),
                "metadata": {
                    "language": example["language"],
                    "bug_type": example["bug_type"],
                    "severity": example["bug_severity"],
                    "original_label": example["label"]
                }
            }
            instruction_dataset.append(instruction_entry)

   
    print("\nInstruction Dataset Quality:")
    print(f"- Total samples: {len(instruction_dataset)}")
    print(f"- Avg code length: {np.mean([len(x['input']) for x in instruction_dataset]):.1f} chars")
    print(f"- Avg fix length: {np.mean([len(x['output']) for x in instruction_dataset]):.1f} chars")
    print(f"- Length ratio (fix/code): {np.mean(length_discrepancies):.2f}:1")

    return instruction_dataset

def save_quality_controlled_datasets():
    """Generate and save all datasets with strict quality control"""
   
    balanced_data = generate_balanced_dataset(50) 
    with open('balanced_dataset.json', 'w') as f:
        json.dump(balanced_data, f, indent=2)

    
    instruction_data = generate_instruction_dataset(buggy_samples_only=True)
    with open('instruction_dataset.json', 'w') as f:
        json.dump(instruction_data, f, indent=2)

    
    expanded_data = generate_balanced_dataset(100) 
    with open('expanded_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)

    
    print("\n=== Dataset Generation Complete ===")
    print(f"- Balanced samples: {len(balanced_data)} (50 per language)")
    print(f"- Instruction samples: {len(instruction_data)} (buggy only)")
    print(f"- Expanded samples: {len(expanded_data)} (100 per language)")

if __name__ == "__main__":
    save_quality_controlled_datasets()