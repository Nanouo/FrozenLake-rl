# FrozenLake-rl

A tabular Q-learning agent that learns to cross Gymnasium's FrozenLake — written from scratch (no RL libraries) to understand reinforcement learning from the ground up.

## Results

| environment | success rate | notes |
|---|---|---|
| deterministic 4x4 | 100% | 10/10 independent training runs |
| deterministic 8x8 | 100% | ~94% of runs produce a working agent |
| **slippery 4x4** | **71.9%** | vs a computed optimum of **72.6%** |

On slippery ice the intended action only happens 1/3 of the time, so 100% is
impossible — a perfect policy still gets pushed into holes. The 72.6% ceiling
was computed with value iteration over the environment's transition model, so
the tuned agent runs at ~99% of theoretical maximum.

## Running it

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe main.py            # train
./.venv/Scripts/python.exe eval_harness.py    # train 10 agents, greedy-evaluate each
./.venv/Scripts/python.exe visualize.py       # policy arrows + animated rollout
```

## Files

- `main.py` — the `RLModel` class: Q-table, epsilon-greedy action selection, Bellman update, training loop
- `eval_harness.py` — trains N independent agents and evaluates each greedily (epsilon=0). Single runs are misleading; this is the honest test
- `visualize.py` — renders the learned policy as arrows and animates a greedy rollout

## Two bugs worth knowing about

**1. The all-zeros tie-break.** 4 of 10 identical runs learned nothing, with no
error and no crash. When every value in a Q-table row is 0, `np.argmax` returns
index 0 — which is LEFT — and LEFT into a wall is a no-op. So every
"exploitation" step walked into a wall until the step limit ran out. Fixed by
breaking ties randomly (`argmax_random`): reliability went 6/10 -> 10/10.

**2. A learning rate that was correct until the environment changed.**
`learning_rate = 0.8` is fine on deterministic ice, where one observation is the
complete truth about a state-action pair. On slippery ice a Q-value has to be an
*average* over three possible outcomes, and 0.8 makes it chase whichever outcome
happened most recently instead of converging. Dropping to 0.05 (plus more
episodes, since small updates need more samples) took success from an unstable
47% — swinging between 0% and 77% across runs — to a consistent 71.9%.

Neither bug raised an exception. Both were only visible by running training many
times and noticing the results were inconsistent.
