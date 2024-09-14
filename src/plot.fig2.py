import matplotlib.pyplot as plt

'''
the new prelimary result table for the effectiveness of the weights (Magnitude)
| Pruned Level | Magnitude | Opposite Magnitude |
|-----------|------------------|------------------|
| 0.01      | NA               | 24377.635        |
| 0.05      | NA               | 25804.920        |
| 0.10      | 5.806            | 104948.891       |
| 0.20      | 6.020            | 352772.500       |
| 0.30      | 6.669            | 335747.406       |
| 0.40      | 8.601            | 260632.641       |
| 0.50      | 17.285           | 227413.484       |
| 0.60      | 559.987          | 185086.078       |
| 0.70      | 48414.551        | 273153.688       |
| 0.80      | 132175.578       | 188488.000       |
| 0.90      | 317879.250       | 185304.016       |
| 0.95      | 273552.281       | NA               |
| 0.99      | 222543.047       | NA               |
'''

# Data
pruned_levels = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
magnitude_perplexities = [None, None, 5.806, 6.020, 6.669, 8.601, 17.285, 559.987, 48414.551, 132175.578, 317879.250, 273552.281, 222543.047]
opposite_magnitude_perplexities = [24377.635, 25804.920, 104948.891, 352772.500, 335747.406, 260632.641, 227413.484, 185086.078, 273153.688, 188488.000, 185304.016, None, None]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pruned_levels, magnitude_perplexities, marker='s', linestyle='-.', color='g', label='Magnitude')
plt.plot(pruned_levels, opposite_magnitude_perplexities, marker='o', linestyle='-', color='b', label='Opposite Magnitude')

# Annotate each point
for i, (x, y) in enumerate(zip(pruned_levels, magnitude_perplexities)):
    if y is not None:  # Only annotate if y value exists
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='center')

for i, (x, y) in enumerate(zip(pruned_levels, opposite_magnitude_perplexities)):
    if y is not None:  # Only annotate if y value exists
        plt.text(x, y, f'{y:.1f}', fontsize=9, ha='center')

plt.yscale('log')  # Set y-axis to logarithmic scale for better visualization
plt.xlabel('Pruned Level')
plt.ylabel('Perplexity')
plt.title('the effectiveness of the weights (Magnitude)')
plt.grid(True)
plt.legend()
plt.show()