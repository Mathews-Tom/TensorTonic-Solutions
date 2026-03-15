"""
## SARSA Update

SARSA is an on-policy temporal difference (TD) learning algorithm. The name comes from the quintuple (State, Action, Reward, next State, next Action) used in each update. It learns Q-values (action-value estimates) by bootstrapping from the next state-action pair that the agent actually takes.

Given a Q-table and a single transition (s, a, r, s', a'), perform one SARSA update and return the updated Q-table.

### Algorithm

1. Compute the TD error:

$$\delta =r+\gamma\cdot Q(s',a')-Q(s,a)$$

2. Update the Q-value for the current state-action pair:

$$Q(s,a)\leftarrow Q(s,a)+\alpha \cdot \gamma$$

### Examples

Input: q_table = [[0, 0], [0, 0]], state = 0, action = 1, reward = 1.0, next_state = 1, next_action = 0, alpha = 0.1, gamma = 0.9
Output: [[0.0, 0.1], [0.0, 0.0]]
TD error = 1.0 + 0.9 * 0 - 0 = 1.0. Update: Q(0,1) = 0 + 0.1 * 1.0 = 0.1. All other Q-values stay the same.

Input: q_table = [[1, 2], [3, 4]], state = 0, action = 0, reward = 5.0, next_state = 1, next_action = 1, alpha = 0.5, gamma = 0.9
Output: [[4.8, 2.0], [3.0, 4.0]]
TD error = 5.0 + 0.9 * 4 - 1 = 7.6. Update: Q(0,0) = 1 + 0.5 * 7.6 = 4.8.

### Hints

- Hint 1: First make a deep copy of the Q-table (copy each row). Then compute td = reward + gamma * q_table[next_state][next_action] - q_table[state][action]. Finally update the copy: new_q[state][action] += alpha * td.
- Hint 2: Make sure to use the original Q-table values when computing the TD error, not the copy. The copy is only for writing the update.

### Requirements

- Compute the TD error using the reward, discounted next Q-value, and current Q-value
- Update only Q(state, action) using the learning rate alpha
- Do not modify the original Q-table. Return a new 2D list
- All other Q-values remain unchanged

### Constraints

- q_table is a 2D list of floats with shape [num_states][num_actions]
- state, action, next_state, next_action are valid indices
- 0 <= alpha <= 1, 0 <= gamma <= 1
- Return a new 2D list (do not modify the input)
- Time limit: 300 ms
"""

def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Deep copy so the original q_table is not modified
    new_q = [row[:] for row in q_table]

    # TD error from original table
    td_error = reward + gamma * q_table[next_state][next_action] - q_table[state][action]

    # Update only the selected state-action entry
    new_q[state][action] += alpha * td_error

    return new_q