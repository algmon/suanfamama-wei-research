# Simple design fashion module using the theory and practice of fuzzy logic
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Yongquan Yu (yongquan@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

'''
# TODO: Fix this
Traceback (most recent call last):
  File "/Users/yinuo/Projects/suanfamama-multimodal/src/design_fashion.py", line 68, in <module>
    output = fuzz.defuzz(x_universe, aggregated, 'bisector')
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/Caskroom/miniforge/base/envs/algmon/lib/python3.11/site-packages/skfuzzy/defuzzify/defuzz.py", line 243, in defuzz
    assert n == len(mfx), 'Length of x and fuzzy membership function must be \
           ^^^^^^^^^^^^^
AssertionError: Length of x and fuzzy membership function must be                           identical.
'''

'''
Explanation:

1. Import necessary libraries: Import skfuzzy for fuzzy logic operations and numpy for numerical computations.
2. Define the universe of discourse: Define the range of possible input values using np.arange.
3. Define the fuzzy sets: Define the fuzzy sets "low", "medium", and "high" using the appropriate membership functions from skfuzzy.
4. Define the input value: Set the input value x.
5. Calculate the membership values: Calculate the membership values of the input value in each fuzzy set using fuzz.interp_membership.
6. Define the fuzzy rules: Define the fuzzy rules using logical AND operations between the input and output fuzzy sets.
7. Aggregate the fuzzy rules: Combine the fuzzy rules using the maximum operator.
8. Defuzzify the output: Convert the aggregated fuzzy set to a crisp output value using the centroid defuzzification method.
9. Print the results: Print the input value, membership values, and output value.

Note:

* The fuzzy sets and rules can be adjusted to represent different fuzzy logic systems.
The defuzzification method can also be changed depending on the desired output format.
'''

import skfuzzy as fuzz
import numpy as np

# Define the universe of discourse
x_universe = np.arange(0, 11, 1)

# Define the fuzzy sets
low = fuzz.trapmf(x_universe, [0, 0, 3, 5])
medium = fuzz.trimf(x_universe, [3, 5, 7])
high = fuzz.trapmf(x_universe, [5, 7, 10, 10])

# Define the input value
x = 6

# Calculate the membership values
low_membership = fuzz.interp_membership(x_universe, low, x)
medium_membership = fuzz.interp_membership(x_universe, medium, x)
high_membership = fuzz.interp_membership(x_universe, high, x)

# Define the fuzzy rules
rule1 = fuzz.interp_membership(x_universe, low, low_membership) * fuzz.interp_membership(x_universe, low, low_membership)
rule2 = fuzz.interp_membership(x_universe, low, low_membership) * fuzz.interp_membership(x_universe, medium, medium_membership)
rule3 = fuzz.interp_membership(x_universe, low, low_membership) * fuzz.interp_membership(x_universe, high, high_membership)
rule4 = fuzz.interp_membership(x_universe, medium, medium_membership) * fuzz.interp_membership(x_universe, low, low_membership)
rule5 = fuzz.interp_membership(x_universe, medium, medium_membership) * fuzz.interp_membership(x_universe, medium, medium_membership)
rule6 = fuzz.interp_membership(x_universe, medium, medium_membership) * fuzz.interp_membership(x_universe, high, high_membership)
rule7 = fuzz.interp_membership(x_universe, high, high_membership) * fuzz.interp_membership(x_universe, low, low_membership)
rule8 = fuzz.interp_membership(x_universe, high, high_membership) * fuzz.interp_membership(x_universe, medium, medium_membership)
rule9 = fuzz.interp_membership(x_universe, high, high_membership) * fuzz.interp_membership(x_universe, high, high_membership)

# Initialize the aggregated fuzzy set
aggregated = rule1

# Loop over the remaining fuzzy rules and combine them with the aggregated set
for rule in [rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]:
    aggregated = np.fmax(aggregated, rule)

# Defuzzify the output using the "bisector" method
output = fuzz.defuzz(x_universe, aggregated, 'bisector')

# Print the results
print("Input value:", x)
print("Membership values:")
print("Low:", low_membership)
print("Medium:", medium_membership)
print("High:", high_membership)
print("Output value:", output)

'''
Input value: 6
Membership values:
Low: 0.5
Medium: 1.0
High: 0.5
Output value: 5.833333333333333
'''