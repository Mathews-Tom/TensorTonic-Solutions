"""
## Shadow Deployment Evaluation

In shadow deployment (also called dark launching), a candidate model runs alongside the production model on identical inputs. Only the production model's predictions are served to users. The shadow model's outputs are logged for offline comparison.

Given prediction logs from both models and evaluation criteria, determine if the shadow model is ready to replace production.

### Metrics

$$\text{Accuracy}=\frac{\text{number of predictions matching actual}}{n}$$
 
$$\text{Accuracy Gain}=\text{Shadow Accuracy} - \text{Production Accuracy}$$

P95 latency uses the nearest-rank method:

$$index=\left \lceil 0.95\times n \right \rceil - 1$$

Sort shadow latencies ascending and select the element at this index.

$$\text{Agreement Rate}=\frac{\text{inputs where both models predict the same value}}{n}$$
 
### Promotion Decision

The shadow model is promoted only when all criteria are satisfied simultaneously:

1. accuracy_gain >= min_accuracy_gain
2. shadow_latency_p95 <= max_latency_p95
3. agreement_rate >= min_agreement_rate

### Examples

Input: production_log = [   {"input_id": 1, "prediction": 1, "actual": 1, "latency_ms": 15},   {"input_id": 2, "prediction": 0, "actual": 1, "latency_ms": 20},   {"input_id": 3, "prediction": 1, "actual": 1, "latency_ms": 18},   {"input_id": 4, "prediction": 0, "actual": 0, "latency_ms": 22}, ] shadow_log = [   {"input_id": 1, "prediction": 1, "actual": 1, "latency_ms": 10},   {"input_id": 2, "prediction": 1, "actual": 1, "latency_ms": 25},   {"input_id": 3, "prediction": 1, "actual": 1, "latency_ms": 20},   {"input_id": 4, "prediction": 0, "actual": 0, "latency_ms": 30}, ] criteria = {"min_accuracy_gain": 0.0, "max_latency_p95": 50.0, "min_agreement_rate": 0.5}
Output: {"promote": True, "metrics": {"shadow_accuracy": 1.0, "production_accuracy": 0.75, "accuracy_gain": 0.25, "shadow_latency_p95": 30, "agreement_rate": 0.75}}
Shadow is more accurate (1.0 vs 0.75), P95 latency is 30ms (within 50ms limit), and agreement rate is 0.75 (above 0.5 threshold).

Input: production_log = [   {"input_id": 1, "prediction": 1, "actual": 1, "latency_ms": 15},   {"input_id": 2, "prediction": 0, "actual": 1, "latency_ms": 20},   {"input_id": 3, "prediction": 1, "actual": 1, "latency_ms": 18},   {"input_id": 4, "prediction": 0, "actual": 0, "latency_ms": 22}, ] shadow_log = [   {"input_id": 1, "prediction": 1, "actual": 1, "latency_ms": 40},   {"input_id": 2, "prediction": 1, "actual": 1, "latency_ms": 45},   {"input_id": 3, "prediction": 1, "actual": 1, "latency_ms": 50},   {"input_id": 4, "prediction": 0, "actual": 0, "latency_ms": 200}, ] criteria = {"min_accuracy_gain": 0.0, "max_latency_p95": 100.0, "min_agreement_rate": 0.5}
Output: {"promote": False, "metrics": {"shadow_accuracy": 1.0, "production_accuracy": 0.75, "accuracy_gain": 0.25, "shadow_latency_p95": 200, "agreement_rate": 0.75}}
Shadow is more accurate and has high agreement, but P95 latency is 200ms which exceeds the 100ms limit. Promotion blocked.

### Hints

- Hint 1: For P95 with nearest-rank, sort the latencies ascending and pick the element at index ceil(0.95 * n) - 1.
- Hint 2: Agreement rate compares what the two models predicted, not whether they were correct.

### Requirements

- Compute accuracy as fraction of predictions matching the actual value
- Use nearest-rank method for P95 latency of the shadow model
- Agreement rate measures how often both models predict the same value
- Promotion requires all criteria to be satisfied simultaneously

### Constraints

- 1 <= len(production_log) == len(shadow_log) <= 10000
- Predictions and actuals are integers
- latency_ms > 0
- -1.0 <= min_accuracy_gain <= 1.0
- Time limit: 300 ms
"""

import math

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    n = len(production_log)

    # Accuracy
    production_correct = sum(
        1 for row in production_log
        if row["prediction"] == row["actual"]
    )
    shadow_correct = sum(
        1 for row in shadow_log
        if row["prediction"] == row["actual"]
    )

    production_accuracy = production_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - production_accuracy

    # Shadow P95 latency using nearest-rank method
    shadow_latencies = sorted(row["latency_ms"] for row in shadow_log)
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = shadow_latencies[p95_index]

    # Agreement rate
    agreement_count = sum(
        1 for p_row, s_row in zip(production_log, shadow_log)
        if p_row["prediction"] == s_row["prediction"]
    )
    agreement_rate = agreement_count / n

    metrics = {
        "shadow_accuracy": shadow_accuracy,
        "production_accuracy": production_accuracy,
        "accuracy_gain": accuracy_gain,
        "shadow_latency_p95": shadow_latency_p95,
        "agreement_rate": agreement_rate,
    }

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": metrics,
    }