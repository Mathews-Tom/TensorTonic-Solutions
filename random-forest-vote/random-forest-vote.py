"""
## Random Forest Majority Vote

A Random Forest makes predictions by aggregating the outputs of multiple decision trees. For classification, each tree votes for a class and the final prediction is the class with the most votes (majority vote).

Given the predictions from T decision trees for N samples, compute the majority vote for each sample. Break ties by choosing the smallest class label.

### Algorithm

1. For each sample, count votes from all trees
2. Select the class with the highest vote count
3. If multiple classes are tied, pick the smallest class label

### Examples

Input:predictions = [[0, 1, 0], [0, 1, 1], [0, 0, 0]]
Output: [0, 1, 0]
Sample 0: votes {0:3} = 0. Sample 1: votes {1:2, 0:1} = 1. Sample 2: votes {0:2, 1:1} = 0.

Input: predictions = [[0, 1], [1, 0]]
Output: [0, 0]
Both samples have a 1-1 tie between classes 0 and 1. Ties are broken by choosing the smallest label, so both predict 0.

### Hints

Hint 1: Loop over each sample index i. For each i, count how many times each class appears across all trees using a dictionary. Then find the class with the highest count.

Hint 2: To break ties by smallest label, find the max count first, then use min() over all keys that have that count.

### Requirements

- predictions[t][i] is tree t's prediction for sample i
- Use NumPy for your implementation
- Count votes across all trees for each sample
- Return the class with the most votes (break ties by smallest label)
- Return a list of integers with length equal to the number of samples

### Constraints

- predictions has at least one tree and one sample
- All trees predict for the same number of samples
- Class labels are non-negative integers
- Return a list of integers
- Time limit: 300 ms
"""

import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    preds = np.asarray(predictions, dtype=int)  # shape (T, N)
    T, N = preds.shape

    result = []

    for i in range(N):
        votes = preds[:, i]
        counts = np.bincount(votes)   # index = class label, value = vote count
        winner = np.argmax(counts)    # smallest label chosen automatically on ties
        result.append(int(winner))

    return result