import matplotlib.pyplot as plt
import numpy as np

# Generating sample data (replace these with your arrays)
import pickle

lambda_1 = pickle.load(open('lambda_1.0_source_mask', 'rb'))
lambda_2 = pickle.load(open('lambda_0.2_source_mask', 'rb'))
lambda_5 = pickle.load(open('lambda_0.5_source_mask', 'rb'))
lambda_8 = pickle.load(open('lambda_0.8_source_mask', 'rb'))
lambda_0 = pickle.load(open('lambda_0.0_source_mask', 'rb'))

print(np.mean(lambda_0))
print(np.mean(lambda_2))
print(np.mean(lambda_5))
print(np.mean(lambda_8))
print(np.mean(lambda_1))

# Creating subplots with shared x-axis and same axis scale
fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

# Plotting histograms for each dataset on separate subplots
axes[0].hist(lambda_1, bins=20, color='skyblue', alpha=0.7)
axes[1].hist(lambda_8, bins=20, color='salmon', alpha=0.7)
axes[2].hist(lambda_5, bins=20, color='palegreen', alpha=0.7)
axes[3].hist(lambda_2, bins=20, color='orchid', alpha=0.7)
axes[4].hist(lambda_0, bins=20, color='gold', alpha=0.7)

for i in range(5):
    axes[i].set_ylim(0, 450)  # Setting the same y-axis scale for all subplots
    axes[i].set_xlim(6, 19)  # Setting the same y-axis scale for all subplots


# Set common labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.suptitle('Histogram for Source Masking : Aids (Seed 4586)')

# Show plot
plt.tight_layout()
plt.savefig('new.png')
