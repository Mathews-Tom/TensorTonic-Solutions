"""
## Precision and Recall at K

Precision@k and recall@k are the standard metrics for evaluating top-k recommendation lists. Precision@k measures what fraction of the recommended items are relevant, while recall@k measures what fraction of all relevant items were recommended. Together they capture the trade-off between recommendation quality and coverage.

Given a ranked list of recommended items, a set of relevant (ground truth) items, and a cutoff k, compute both precision@k and recall@k.

### Algorithm

$$\text{Precision@k}=\frac{|\text{top-k}\bigcap \text{relevant}|}{k}$$

$$\text{Recall@k}=\frac{|\text{top-k}\bigcap \text{relevant}|}{|\text{relevant}|}$$

### Examples

Input: recommended = [1, 3, 5, 7, 9], relevant = [1, 2, 3, 4, 5], k = 3
Output: [1.0, 0.6]
Top-3 = [1, 3, 5]. All 3 are relevant. Precision = 3/3 = 1.0. Recall = 3/5 = 0.6.

Input: recommended = [10, 20, 30], relevant = [1, 2, 3], k = 3
Output: [0.0, 0.0]
None of the recommended items are relevant. Both precision and recall are 0.

### Hints

Hint 1: Slice the recommended list to get the top k items: top_k = recommended[:k]. Convert the relevant items to a set for O(1) lookup. Count how many items in top_k appear in the relevant set.

Hint 2: Precision is hits/k, recall is hits/len(relevant). Return them as a two-element list [precision, recall].

### Requirements

- Consider only the first k items from the recommended list
- Count how many of those top-k items appear in the relevant set
- Precision@k = hits / k
- Recall@k = hits / number of relevant items
- Return [precision, recall] as a list of two floats

### Constraints

- recommended has at least k elements
- relevant has at least 1 element
- k >= 1
- Return a list of two floats [precision, recall]
- Time limit: 300 ms
"""

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    relevant_set = set(relevant)

    hits = sum(1 for item in top_k if item in relevant_set)

    precision = hits / k
    recall = hits / len(relevant_set)

    return [float(precision), float(recall)]