"""Terminal visualizations for the trained FrozenLake agent.

Two views:
  - policy map:  the greedy best action for every tile, as arrows
  - rollout:     animate the agent walking its greedy policy from S to G

Run:
  ./.venv/Scripts/python.exe visualize.py            # policy map + stacked rollout frames
  ./.venv/Scripts/python.exe visualize.py --animate  # live animation (clears screen each step)
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

from main import RLModel

# Windows terminals default to cp1252, which can't encode the arrow glyphs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Action index -> arrow. Matches gymnasium's FrozenLake: 0=Left 1=Down 2=Right 3=Up
ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
INVERT = "\033[7m"


def _grid_shape(model):
    """(rows, cols) from the env's map description."""
    desc = model.env.unwrapped.desc  # bytes array, e.g. b'S' b'F' b'H' b'G'
    return desc.shape


def _tile_kind(model, state):
    """'S' start, 'F' frozen, 'H' hole, 'G' goal for a state index."""
    rows, cols = _grid_shape(model)
    desc = model.env.unwrapped.desc
    return desc[state // cols, state % cols].decode()


def render_policy(model):
    """Print the greedy policy: one arrow per tile, holes/goal/start marked."""
    rows, cols = _grid_shape(model)
    print(f"{BOLD}Learned policy (greedy best action per tile){RESET}")
    for r in range(rows):
        cells = []
        for c in range(cols):
            state = r * cols + c
            kind = _tile_kind(model, state)
            if kind == "H":
                cells.append(f"{RED} H {RESET}")
            elif kind == "G":
                cells.append(f"{GREEN} G {RESET}")
            else:
                # best action from the learned table
                arrow = ARROWS[int(np.argmax(model.q_table[state]))]
                color = CYAN if kind == "S" else ""
                cells.append(f"{color} {arrow} {RESET}")
        print(" " + "|".join(cells))
    print()


def render_frame(model, state):
    """One grid frame with the agent highlighted at `state`."""
    rows, cols = _grid_shape(model)
    lines = []
    for r in range(rows):
        cells = []
        for c in range(cols):
            s = r * cols + c
            kind = _tile_kind(model, s)
            glyph = {"S": "S", "F": ".", "H": "H", "G": "G"}[kind]
            if s == state:
                cells.append(f"{INVERT}{BOLD}{YELLOW} A {RESET}")  # agent
            elif kind == "H":
                cells.append(f"{RED} {glyph} {RESET}")
            elif kind == "G":
                cells.append(f"{GREEN} {glyph} {RESET}")
            else:
                cells.append(f" {glyph} ")
        lines.append(" " + "|".join(cells))
    return "\n".join(lines)


def rollout(model, animate=False, max_steps=50, delay=0.4):
    """Walk the greedy policy from the start and show each step."""
    state, _ = model.env.reset()
    print(f"{BOLD}Greedy rollout{RESET}")
    for step in range(max_steps):
        if animate:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"{BOLD}Greedy rollout{RESET}")
        print(f"\nStep {step}:")
        print(render_frame(model, state))
        if animate:
            time.sleep(delay)
        action = int(np.argmax(model.q_table[state]))
        state, reward, terminated, truncated, _ = model.env.step(action)
        if terminated or truncated:
            if animate:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"{BOLD}Greedy rollout{RESET}")
            print(f"\nStep {step + 1}:")
            print(render_frame(model, state))
            outcome = f"{GREEN}reached the goal in {step + 1} steps{RESET}" if reward == 1.0 \
                else f"{RED}fell in a hole / timed out{RESET}"
            print(f"\nResult: {outcome}\n")
            return
    print(f"\nResult: {RED}did not finish in {max_steps} steps{RESET}\n")


def main():
    animate = "--animate" in sys.argv
    model = RLModel()
    print("Training...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):  # silence train() prints
        model.train()
    print("Done.\n")
    render_policy(model)
    rollout(model, animate=animate)


if __name__ == "__main__":
    main()
