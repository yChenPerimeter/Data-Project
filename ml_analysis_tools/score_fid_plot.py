import re
import matplotlib.pyplot as plt

# Initialize a dictionary to store extracted epoch numbers and FID values
score_dict = {}

# Path to your text file
dir_path = '/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/production_O21CVPL00001_13_01_16_v1'
file_path = f'{dir_path}/sorted_scores.txt'

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Use regular expressions to find the epoch number and FID value
        epoch_match = re.search(r"--epoch (\d+)", line)
        fid_match = re.search(r"FID: ([\d\.]+)", line)
        
        # If both epoch number and FID value were found, convert them to the correct type and store them
        if epoch_match and fid_match:
            epoch = int(epoch_match.group(1))
            fid = float(fid_match.group(1))
            score_dict[epoch] = fid

# Sort the dictionary by its keys (epochs) and prepare the lists for plotting
sorted_epochs = sorted(score_dict.keys())
sorted_fids = [score_dict[epoch] for epoch in sorted_epochs]
# Plotting

# Setting figure dimensions to match LaTeX text width
fig_width = 6  # Approximate text width in inches for a single-column layout
fig_height = fig_width / 1.618  # Golden ratio for aesthetic aspect ratio

plt.figure(figsize=(fig_width, fig_height))

# plt.figure(figsize=(20, 6))
# plt.plot([1,2,10], [1.1,1.2,10.2], marker='o', linestyle='-', color='b', markersize=5)
plt.plot(sorted_epochs, sorted_fids, linestyle='-', color='b')
plt.xlabel('Number of Epochs')
plt.ylabel('FID Value')
plt.grid(True)

# Setting x-axis ticks to show every 5 epochs
# Calculate the range dynamically based on the min and max epoch values
plt.xticks(range(min(sorted_epochs), max(sorted_epochs) + 1, 5))
# Optionally, add a tight layout to ensure everything fits without overlap
plt.tight_layout()


# Save the figure
plt.savefig(f'{dir_path}/fid_by_epoch.png')  # Save as PNG with high resolution
plt.show()
