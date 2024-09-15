import matplotlib.pyplot as plt

'''
the new prelimary result table
| Pruned Level | Wanda | SparseGPT | Magnitude | Movement |
|-----------|-----------------|-----------------|-----------------|-----------------|
| 0.01      | NA              |NA               |NA               | 5.677           |
| 0.05      | NA              |NA               |NA               | 5.714           |
| 0.10      | 5.696           |5.696            |5.806            | 5.806           |
| 0.20      | 5.817           |5.799            |6.020            | 6.020           |
| 0.30      | 5.999           |5.963            |6.669            | 6.668           |
| 0.40      | 6.387           |6.311            |8.601            | 8.5943          |
| 0.50      | 7.257           |7.234            |17.285           | 17.247          |
| 0.60      | 10.691          |10.442           |559.987          | 554.727         |
| 0.70      | 84.905          |27.214           |48414.551        | 51841.121       |
| 0.80      | 5782.432        |182.463          |132175.578       | 135494.797      |
| 0.90      | 19676.668       |3198.101         |317879.250       | 301472.500      |
| 0.95      | 28309.178       |4088.413         |273552.281       | 273629.750      |
| 0.99      | 108234.484      |16869.203        |222543.047       | 214966.484      |
'''

# Data
pruned_levels = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
wanda_perplexities = [None, None, 5.696, 5.817, 5.999, 6.387, 7.257, 10.691, 84.905, 5782.432, 19676.668, 28309.178, 108234.484]
sparsegpt_perplexities = [None, None, 5.696, 5.799, 5.963, 6.311, 7.234, 10.442, 27.214, 182.463, 3198.101, 4088.413, 16869.203125]
magnitude_perplexities = [None, None, 5.806, 6.020, 6.669, 8.601, 17.285, 559.987, 48414.551, 132175.578, 317879.250, 273552.281, 222543.047]
movement_perplexities = [5.677, 5.714, 5.806, 6.020, 6.668, 8.5943, 17.247, 554.727, 51841.121, 135494.797, 301472.500, 273629.750, 214966.484]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pruned_levels, wanda_perplexities, marker='o', linestyle='-', color='b', label='Wanda')
plt.plot(pruned_levels, sparsegpt_perplexities, marker='x', linestyle='--', color='r', label='SparseGPT')
plt.plot(pruned_levels, magnitude_perplexities, marker='s', linestyle='-.', color='g', label='Magnitude')
plt.plot(pruned_levels, movement_perplexities, marker='^', linestyle=':', color='purple', label='Movement')

# Annotate each point
for i, (x, y) in enumerate(zip(pruned_levels, wanda_perplexities)):
    if y is not None:
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='right')

for i, (x, y) in enumerate(zip(pruned_levels, sparsegpt_perplexities)):
    if y is not None:
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='left')

for i, (x, y) in enumerate(zip(pruned_levels, magnitude_perplexities)):
    if y is not None:
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='center')

for i, (x, y) in enumerate(zip(pruned_levels, movement_perplexities)):
    if y is not None:
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='center')

plt.yscale('log')  # Set y-axis to logarithmic scale for better visualization
plt.xlabel('Pruned Level')
plt.ylabel('Perplexity')
plt.title('Pruned Level vs. Perplexity for llama-v1-7B')
plt.grid(True)
plt.legend()
plt.show()