import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

def load_and_visualize(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Plot for Language Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='language', hue='label')
    plt.title('Samples per Language (0=Clean, 1=Buggy)')
    plt.xticks(rotation=45)

    # Plot for Bug Type Distribution
    plt.subplot(1, 2, 2)
    buggy_samples = df[df['label'] == 1]
    sns.countplot(data=buggy_samples, y='bug_type', hue='language')
    plt.title('Bug Types by Language')
    plt.tight_layout()
    plt.show()

    # Plot for Severity Analysis
    plt.figure(figsize=(8, 4))
    severity_order = ['High', 'Medium', 'Low', 'None']
    sns.countplot(data=df, x='bug_severity', hue='label', order=severity_order)
    plt.title('Bug Severity Distribution')
    plt.show()

   
    print("\n=== Balance Verification ===")
    print(f"Total samples: {len(df)}")
    print(f"Buggy/Clean ratio: {len(df[df['label']==1])}:{len(df[df['label']==0])}")

    
    cross_tab = pd.crosstab(
        index=buggy_samples['language'],
        columns=buggy_samples['bug_type'],
        margins=True
    )
    print("\nBug Type x Language Matrix:")
    print(cross_tab)


print("Visualizing BALANCED Dataset:")
load_and_visualize("balanced_dataset.json")

print("\nVisualizing INSTRUCTION Dataset (Buggy Only):")
with open("instruction_dataset.json") as f:
    instr_data = json.load(f)
instr_df = pd.DataFrame([x['metadata'] for x in instr_data])
plt.figure(figsize=(10, 5))
sns.countplot(data=instr_df, x='language', hue='bug_type')
plt.title('Instruction Dataset: Bug Types per Language')
plt.show()