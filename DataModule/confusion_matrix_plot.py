'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the false positive and false negative CSV files
false_positive_df = pd.read_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FP_subclass.csv')
false_negative_df = pd.read_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FN_subclass.csv')

# Extract class information
fp_classes = false_positive_df['True Class'].values
fn_classes = false_negative_df['True Class'].values

# Count occurrences of each class in FP and FN
fp_counts = pd.Series(fp_classes).value_counts()
fn_counts = pd.Series(fn_classes).value_counts()

# Determine all unique classes
all_classes = sorted(set(fp_classes) | set(fn_classes))

# Create a DataFrame for plotting
data = {
    'Class': all_classes,
    'FP': [fp_counts.get(cls, 0) for cls in all_classes],
    'FN': [fn_counts.get(cls, 0) for cls in all_classes],
    'Suspicious': ['Suspicious' if i % 2 == 0 else 'Non-Suspicious' for i in range(len(all_classes))]  # Adjust based on actual criteria
}
df = pd.DataFrame(data)

# Pivot for heatmap
df_fp = df.pivot(index='Class', columns='Suspicious', values='FP').fillna(0)
df_fn = df.pivot(index='Class', columns='Suspicious', values='FN').fillna(0)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.heatmap(df_fp, annot=True, fmt="g", cmap="YlGnBu", ax=axes[0])
axes[0].set_title('False Positives (FP)')
axes[0].set_xlabel('Suspicious / Non-Suspicious')
axes[0].set_ylabel('Class')

sns.heatmap(df_fn, annot=True, fmt="g", cmap="YlGnBu", ax=axes[1])
axes[1].set_title('False Negatives (FN)')
axes[1].set_xlabel('Suspicious / Non-Suspicious')

plt.tight_layout()
plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the false positive and false negative CSV files
false_positive_df = pd.read_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FP_subclass.csv')
false_negative_df = pd.read_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FN_subclass.csv')

# Check the columns and the first few rows to ensure the correct structure
print(false_positive_df.head())
print(false_negative_df.head())

# Extract class information
fp_classes = false_positive_df['True Class'].values
fn_classes = false_negative_df['True Class'].values
fp_labels = false_positive_df['Inference Label'].values
fn_labels = false_negative_df['Inference Label'].values

# Count occurrences of each class and label
fp_counts = pd.crosstab(fp_classes, fp_labels)
fn_counts = pd.crosstab(fn_classes, fn_labels)

# Determine all unique classes and labels
all_classes = sorted(set(fp_classes) | set(fn_classes))
all_labels = sorted(set(fp_labels) | set(fn_labels))

# Ensure that all possible labels are present in the DataFrame
df_fp = pd.DataFrame(fp_counts).reindex(index=all_classes, columns=all_labels, fill_value=0)
df_fn = pd.DataFrame(fn_counts).reindex(index=all_classes, columns=all_labels, fill_value=0)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.heatmap(df_fp, annot=True, fmt="g", cmap="YlGnBu", ax=axes[0])
axes[0].set_title('False Positives (FP)')
axes[0].set_xlabel('Inference Label')
axes[0].set_ylabel('True Class')

sns.heatmap(df_fn, annot=True, fmt="g", cmap="YlGnBu", ax=axes[1])
axes[1].set_title('False Negatives (FN)')
axes[1].set_xlabel('Inference Label')

plt.tight_layout()
plt.show()
