import matplotlib.pyplot as plt

'''
TODO: ADD the new prelimary result table (from the machine side)
| Pruned Level | aigc_technique2 | aigc_technique3 | aigc_technique6 |
|----------|----------|----------|----------|
| 0.50 | 193740.406 | 266826.094 | 294350.188 |
| 0.60 | 110879.422 | 244139.875 | 138577.469 |
| 0.70 | 174815.859 | 453267.031 | 171725.375 |
| 0.80 | 287734.844 | 570346.750 | 186493.797 |
| 0.90 | 157028.844 | 384411.375 | 298142.469 |
| 0.95 | 90220.781  | 455298.469 | 187259.063 |
| 0.99 | 991519.125 | 206585.391 | 70452.703  |

TODO: ADD the new prelimary result table (from the human side)
| Pruned Level | Wanda | SparseGPT | Magnitude | Movement |
|-----------|-----------------|-----------------|-----------------|-----------------|
| 0.50      | 7.257           |7.234            |17.285           | 17.247          |
| 0.60      | 10.691          |10.442           |559.987          | 554.727         |
| 0.70      | 84.905          |27.214           |48414.551        | 51841.121       |
| 0.80      | 5782.432        |182.463          |132175.578       | 135494.797      |
| 0.90      | 19676.668       |3198.101         |317879.250       | 301472.500      |
| 0.95      | 28309.178       |4088.413         |273552.281       | 273629.750      |
| 0.99      | 108234.484      |16869.203        |222543.047       | 214966.484      |
'''

# Data for machine side
pruned_levels_machine = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
aigc_technique2 = [193740.406, 110879.422, 174815.859, 287734.844, 157028.844, 90220.781, 991519.125]
aigc_technique3 = [266826.094, 244139.875, 453267.031, 570346.750, 384411.375, 455298.469, 206585.391]
aigc_technique6 = [294350.188, 138577.469, 171725.375, 186493.797, 298142.469, 187259.063, 70452.703]

# Data for human side
pruned_levels_human = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
wanda_perplexities = [7.257, 10.691, 84.905, 5782.432, 19676.668, 28309.178, 108234.484]
sparsegpt_perplexities = [7.234, 10.442, 27.214, 182.463, 3198.101, 4088.413, 16869.203125]
magnitude_perplexities = [17.285, 559.987, 48414.551, 132175.578, 317879.250, 273552.281, 222543.047]
movement_perplexities = [17.247, 554.727, 51841.121, 135494.797, 301472.500, 273629.750, 214966.484]

# Plot
plt.figure(figsize=(10, 6))

# Machine side plots
plt.plot(pruned_levels_machine, aigc_technique2, marker='o', linestyle='-', color='b', label='AIGC Technique 2 (Machine)')
plt.plot(pruned_levels_machine, aigc_technique3, marker='x', linestyle='--', color='r', label='AIGC Technique 3 (Machine)')
plt.plot(pruned_levels_machine, aigc_technique6, marker='s', linestyle='-.', color='g', label='AIGC Technique 6 (Machine)')

# Human side plots
plt.plot(pruned_levels_human, wanda_perplexities, marker='o', linestyle=':', color='cyan', label='Wanda (Human)')
plt.plot(pruned_levels_human, sparsegpt_perplexities, marker='x', linestyle='-.', color='magenta', label='SparseGPT (Human)')
plt.plot(pruned_levels_human, magnitude_perplexities, marker='s', linestyle='--', color='yellow', label='Magnitude (Human)')
plt.plot(pruned_levels_human, movement_perplexities, marker='^', linestyle='-', color='black', label='Movement (Human)')

# TODO: Annotate each point

plt.yscale('log')  # Set y-axis to logarithmic scale for better visualization
plt.xlabel('Pruned Level')
plt.ylabel('Perplexity')
plt.title('Pruned Level vs. Perplexity for llama-v1-7B')
plt.grid(True)
plt.legend()
plt.show()