# FrozenLake-rl — What This Project Is, In Plain English

Written so I can explain this later without re-deriving it, and so anyone
reading the code knows why it looks the way it does. No jargon assumed.

---

## What the project is

There's a classic practice problem in AI called **FrozenLake**. You have a small
grid of tiles — think a 4x4 checkerboard. The top-left tile is where you start.
The bottom-right is the goal. Some tiles in between are **holes in the ice**:
step on one and you fall in and the attempt is over.

The task: build a program that learns to walk from the start to the goal without
falling in.

The catch, and the entire point: **you don't tell the program where the holes
are.** You don't tell it the rules, or the layout, or which direction the goal
is. It has to figure all of that out by trying things and seeing what happens.
That's what makes it a *learning* problem instead of a maze-solving problem.

## How the agent remembers what it learns

The program keeps a **scorecard**. For every tile, it stores four numbers — one
for each direction it could move: left, down, right, up. Each number is its
current belief about *how good that move is from that tile*.

At the very start every number is zero, because it knows nothing. As it plays
thousands of attempts, the numbers fill in. Once they're accurate, the strategy
is trivial: stand on a tile, look at your four numbers, go whichever direction
has the highest one.

That scorecard is the entire "brain." It's a table of 16 tiles x 4 directions =
64 numbers. In the code it's `self.q_table`.

## How the numbers get filled in

The only feedback the environment ever gives is **+1 for reaching the goal**.
That's it. Falling in a hole gives you nothing — not even a penalty. Walking on
safe ice gives you nothing.

So the learning works backwards from that single point of success. When the
agent finally stumbles into the goal, the move that got it there gets credit.
Next time it's near the goal, *that* tile now looks valuable, so the move that
leads to *it* gets credit. Over many attempts the value seeps backward across
the board like a stain spreading, until even the starting tile knows which way
points home.

That backward-seeping update is `update_q_table`.

## The explore/exploit balance

Early on, the scorecard is worthless — it's all zeros. So the agent moves
**randomly** to gather information.

Later, once the numbers mean something, moving randomly is wasteful. It should
**use** what it knows.

So the program starts out ~100% random and gradually shifts toward following its
scorecard. This is the classic tension: try new things, or use what already
works? Too much exploring and you never commit; too little and you lock in a bad
habit before you know better. That's `choose_action` and the `epsilon` settings.

## Version 1: it worked

On a predictable board — press right, you move right — the agent learned a
perfect route. **100% success**, on both the small 4x4 board and a bigger 8x8
one.

## The first bug, which was invisible

Then came the most useful thing that happened in the project.

Running the exact same code several times gave: **success, success, total
failure, total failure.** Same code. No error message. No crash. Just an agent
that sometimes learned perfectly and sometimes learned *nothing at all*.

The cause turned out to be a detail nobody would think to look at. When all four
numbers on a tile are tied at zero, the code that picks "the highest one" has to
break the tie somehow — and it always picked **the first one in the list, which
was "left."**

At the start, *every* tile is a four-way tie. So whenever the agent wasn't
moving randomly, it moved left. And moving left into the wall does nothing at
all — the agent just stands there burning its turn limit against the edge of the
board.

The fix was one line: when the numbers are tied, **pick randomly among the tied
options** instead of always taking the first (`argmax_random`). Reliability went
from 6 out of 10 runs to **10 out of 10**.

The lesson is the part worth keeping: this bug produced no error and was
invisible in any single run. The only way to find it was to run training many
times and notice the results were *inconsistent*. **A program that runs without
crashing is not the same as a program that works.**

## Version 2: making the ice actually slippery

The board so far was predictable. The real version of this problem turns on
**slippery ice**: when you press "down," you only actually go down **1/3 of the
time**. The other 2/3 you slide sideways.

You still control your intentions, but not your outcomes. This is what makes it
a genuine AI problem rather than a puzzle — it's the difference between "find
the shortest route" and "make good decisions in a world that doesn't do what you
say."

Performance immediately collapsed from 100% to about 30%, with the same code.

## The second bug: a setting that had quietly become wrong

Nothing was broken. The algorithm was correct. The problem was a single
configuration number called the **learning rate** — how much the agent revises
its beliefs after each new experience.

It was set to **0.8**, meaning *"throw away 80% of what I believed and trust
what just happened."*

On predictable ice, that's perfectly sensible. If pressing right *always* moves
you right, then one observation is the complete truth, and believing it
immediately is the fastest way to learn.

On slippery ice, it's a disaster. Now the same move produces three different
outcomes depending on luck, so each number on the scorecard needs to be an
**average** across all of them. With the setting at 0.8, the agent was
effectively rewriting its beliefs based on the *last coin flip it saw*. One
lucky run and a move looked great; one unlucky run and it looked terrible. The
numbers never settled down.

Turning it down to **0.05** — *"nudge my belief 5% toward what just happened"* —
means that many small nudges gradually average out the randomness. That's what
averaging *is*, done incrementally.

Combined with more practice runs, success went from an unstable **47%**
(swinging between 0% and 77% depending on the run) to a steady **72%**.

## The last question: how do you know 72% is good?

On slippery ice, **100% is impossible**. Sometimes the ice simply pushes you
into a hole no matter how well you played. So what's a good score? 72%? 85%?
Without knowing, "72%" is a number with no meaning.

It turns out the answer is computable. The environment's internal rulebook — the
exact probability of every outcome — can be read directly by the program. Using
it, you can calculate the best score *any possible strategy* could achieve,
perfect play included. The technique is called **value iteration**: apply the
same update rule the agent uses, but across every tile at once and using the
true probabilities instead of sampled experience, over and over until the
numbers stop changing.

That number is **72.6%**.

The tuned agent scores **71.9%**. So it's operating at roughly **99% of the best
that is theoretically possible**, and the remaining gap isn't worth chasing —
it's the ice, not the agent.

This reframes the whole result. "72%" sounds mediocre. "72% against a hard
ceiling of 72.6%" is *solved*.

## What this project actually taught

1. **Correct code can still be broken.** Both real bugs threw no errors. One was
   a tie-breaking detail; the other was a setting that used to be right and
   silently stopped being right when the environment changed.
2. **Run it more than once.** Both bugs were only visible across repeated runs.
   Single-run testing would have declared victory twice, wrongly.
3. **Averages hide things.** A configuration averaging 47% turned out to swing
   between 0% and 77%. The average looked mediocre; the truth was "wildly
   unreliable," which is a completely different problem.
4. **A score needs a scale.** Any number is meaningless without knowing what
   perfect looks like.
5. **Settings aren't universal.** The exact same code, unchanged, was excellent
   in one environment and terrible in another. What broke was the assumptions
   the settings were tuned under.

## Measurements behind the claims

Slippery 4x4, 10 independent training runs per setting (mean / worst run).
Optimum is 72.6%.

| learning rate | 3,000 episodes | 10,000 episodes | 20,000 episodes |
|---|---|---|---|
| 0.8 | 47.1% / 0.0% | 60.6% / 48.4% | 48.9% / 11.2% |
| 0.5 | 65.2% / 39.2% | 68.7% / 49.8% | 64.7% / 48.0% |
| 0.2 | 61.8% / 30.4% | 57.1% / 23.0% | 67.8% / 51.6% |
| 0.1 | 59.3% / 38.6% | 67.8% / 48.6% | 63.2% / 49.6% |
| **0.05** | 69.0% / 47.6% | 69.7% / 50.8% | **71.9% / 69.4%** |
| 0.01 | 6.1% / 3.0% | 4.8% / 1.2% | 11.9% / 4.2% |

Two things worth noting. A learning rate of 0.01 collapses — too small to spread
the goal's value across the board within the practice budget — so lower is not
simply better; 0.05 is a sweet spot. And 0.8 is so unstable that its own average
is unreliable: it measured 47.1% in one batch and 29.5% in a smaller one, same
setting.

## Honest notes

- The shipped configuration (0.05, 20,000 episodes) takes its 71.9% figure from
  the sweep above rather than a separate end-to-end run of `main.py` at those
  exact settings. Same configuration, but not independently re-verified.
- I wrote the agent — the Q-learning implementation in `main.py`. I used Claude
  as a tutor for the concepts, and it wrote the measurement scaffolding (the
  hyperparameter sweep and the value-iteration calculation) that produced the
  numbers above.
