"""Run several independent training trials and greedily evaluate each.

Exposes the flakiness the cumulative-win-rate print hides: after training,
follow the learned policy deterministically (argmax, no exploration) and see
whether it actually reaches the goal.
"""
import contextlib
import io
import numpy as np
from main import RLModel


def greedy_eval(model, episodes=100, max_steps=100):
    """Success rate following the learned table with no exploration."""
    wins = 0
    for _ in range(episodes):
        state, _ = model.env.reset()
        for _ in range(max_steps):
            action = int(np.argmax(model.q_table[state]))
            state, reward, terminated, truncated, _ = model.env.step(action)
            if terminated or truncated:
                if reward == 1.0:
                    wins += 1
                break
    return wins / episodes


def main(trials=10):
    results = []
    for t in range(trials):
        model = RLModel()
        with contextlib.redirect_stdout(io.StringIO()):  # silence train() prints
            model.train()
        rate = greedy_eval(model)
        results.append(rate)
        print(f"Trial {t + 1:2d}: greedy success rate = {rate:.0%}")
    solved = sum(1 for r in results if r > 0.5)
    print(f"\nSolved (>50%): {solved}/{trials}")


if __name__ == "__main__":
    main()
