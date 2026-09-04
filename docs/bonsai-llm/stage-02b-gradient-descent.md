---
title: Teach a Line to Fit Itself
stage: "02b"
archetype: implement
minutes: 100
new_concepts: [seeding, MSE, gradient, learning rate]
new_tools: []
---

# Stage 02b — Teach a Line to Fit Itself

## 🔁 Last time

You cleaned real gasoline prices, split them honestly by time, and found the
baseline you have to beat. What you don't have yet is any way to produce a
prediction. This stage builds that machine — on data simple enough that you
already know the right answer.

## 🎯 What you're making

A `gradient_descent.py` that starts from the flattest possible guess and
walks its way to the exact line you used to generate the data — proving it
recovered numbers it was never told. About 100 minutes.

"Fit" is the word for that walk. A line has two knobs: $w$, the **weight**,
controlling steepness by multiplying $x$; and $b$, the **bias**, shifting the
line up or down without changing its steepness. This is $y = mx + c$ from
school, wearing different letters — $w$ is the slope, which is why it
controls steepness, and $b$ is the y-intercept. A bias does an intercept's
usual job: without one, the line is forced through the origin, and real data
almost never passes through $(0, 0)$.

Both are **parameters** — numbers the line owns and adjusts, unlike $x$,
which is handed to it. Fitting means turning those two knobs until the
predictions stop being wrong — not zero wrong, usually, as little wrong as
this line can be. Every model in this journey, all the way to BonsaiGPT, is
this same idea with more knobs — millions, by the end. Get it working with
two first.

---

## 🤔 Before you open your editor

Paper first.

1. You have a knob that controls how wrong your line is. Turning it one way
   makes things worse, the other way makes things better. With no calculus
   at all, how would you find out which way to turn it?
2. Once you know which way, how far do you turn it? What goes wrong if you
   turn it a tiny amount each time? What goes wrong if you turn it a huge
   amount each time?
3. How would you know the line is done improving, versus still improving
   slowly, versus actually getting worse?

---

## 🔨 Build 1 — Data with a known answer

Every other dataset here is real, so you never know the true relationship
you're recovering — only how well your model did. Honest, but a bad way to
debug a new algorithm: a bug and a hard dataset look identical from outside.
Build a dataset where you already know the answer, because you chose it.

```powershell
cd C:\dev\bonsai
mkdir 02b_gradient_descent
```

Create `02b_gradient_descent/gradient_descent.py`:

```python
import numpy as np


def generate_data(n=200, true_w=3.0, true_b=7.0, noise_std=0.5, seed=0):
    """Return (x, y): y = true_w * x + true_b + noise, noise ~ N(0, noise_std)."""
    raise NotImplementedError
```

Read the signature before writing anything. `n` is how many points to
generate. `true_w` and `true_b` are the weight and bias hidden inside the
data — 3.0 and 7.0 by default, the exact numbers your descent loop has to
rediscover later. `noise_std` sets how much random scatter sits around the
line, so the data looks like a measurement instead of a formula. `seed`
fixes the randomness, so the same call always returns the same points. Every
parameter already has a default, so a bare `generate_data()` reproduces the
identical experiment every time — useful, since you'll call this constantly
while debugging.

Write the body: build `n` values of $x$ spread across a range, compute
`y = true_w * x + true_b`, then add Gaussian noise scaled by `noise_std` so
it looks like a measurement rather than a formula. Get repeatable noise from
a seeded generator: `np.random.default_rng(seed)` is NumPy's modern way to
build one — pass it the same `seed` twice and it returns the same sequence
of "random" draws both times, which is what makes the whole function
reproducible.

That `seed` argument is worth attention despite being one line. Unseeded
"random" data is a different dataset every run — a bug you saw five minutes
ago might not reproduce now. Seeding makes it "looks random, but is exactly
repeatable" — you'll want that constantly from here on.

??? tip "Hint — open only when stuck"
    `rng = np.random.default_rng(seed)`. Build `x` with
    `rng.uniform(-5, 5, size=n)`, and the noise with
    `rng.normal(0, noise_std, size=n)`. Then `y = true_w * x + true_b +
    noise`. Return `x, y`.

Call it in `02b_gradient_descent/explore.py`:

```python
from gradient_descent import generate_data

x, y = generate_data()
print(x.shape, y.shape)
print(x[:5])
print(y[:5])
```

Add this to the same `02b_gradient_descent/explore.py`, right after the
prints above:

```python
import matplotlib.pyplot as plt

plt.scatter(x, y, s=10, alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Generated data: y = 3x + 7 + noise")
plt.savefig("data_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
```

Run it:

```powershell
cd 02b_gradient_descent
python explore.py
```

You should see a tilted band of points — clearly a line, crossing the
y-axis near 7 — but fuzzy rather than exact. That fuzziness is `noise_std`
at work, and it's why Build 4's recovered $w$ and $b$ land close to 3 and 7
rather than precisely on them.

Run it twice — same five numbers, the seed working. Change `seed` and run
again — different numbers, same shape.

Every Build below keeps adding to this same file, so `x` and `y` stay
loaded for the rest of the stage — no need to regenerate them.

---

## 🔨 Build 2 — Measure how wrong the line is

You cannot improve a line without a number that says how bad it currently is.
That number is called **loss**, and the specific one you'll use is **mean
squared error** — MSE.

For each point, the error is the gap between what happened and what your
line predicted: $y - \hat y$, where $y$ is the actual value and $\hat y$
("y-hat") is your line's prediction. Adding those directly is a trap — a
line too high by 5 and too low by 5 elsewhere scores a suspicious zero,
cancelling two real mistakes. Squaring fixes that: every error becomes
positive, and a miss twice as large counts four times as much.

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat y_i)^2
$$

Average, not total, so the loss doesn't grow just from feeding it more data.

Add to `02b_gradient_descent/gradient_descent.py`:

```python
def mse_loss(y_true, y_pred):
    """Mean squared error between two equal-length arrays."""
    raise NotImplementedError
```

You're writing this one yourself. `y_true` and `y_pred` are two arrays of
the same length — the actual values and your line's predictions for every
point at once. Return a single number: the formula above, applied across
the whole array in one shot rather than point by point.

??? tip "Hint — open only when stuck"
    `(y_true - y_pred) ** 2` squares every element at once — no loop.
    `np.mean(...)` on the result both sums and divides by $n$ in one call:
    `return np.mean((y_true - y_pred) ** 2)`.

Add to `02b_gradient_descent/explore.py` (update the import at the top to
`from gradient_descent import generate_data, mse_loss`):

```python
y_guess_flat = 0 * x + 0     # w=0, b=0: the flattest, laziest possible line
y_guess_close = 1 * x + 0    # w=1, b=0: pointed the right way, not there yet
y_guess_true = 3 * x + 7     # w=3, b=7: the actual line the data came from

print("loss, flat (w=0, b=0):     ", mse_loss(y, y_guess_flat))
print("loss, closer (w=1, b=0):   ", mse_loss(y, y_guess_close))
print("loss, true (w=3, b=7):     ", mse_loss(y, y_guess_true))
```

Run it. Loss should fall as the guess gets closer to the true line — roughly
148, then 96, then 0.26. That last number is your starting point for Build
4: everything from here tries to shrink toward it. Notice it isn't zero,
even with the exact $w$ and $b$ the data came from — that's `noise_std` from
Build 1, and no amount of fitting removes noise that's actually in the data.

---

## 🔨 Build 3 — Find which way to turn the knob

Loss is a function of $w$ and $b$. Calculus answers question 1: the
derivative of loss with respect to a parameter says which way it must move
to make loss go *up* — so you move the opposite way.

Start from $\hat y = wx + b$ and $L = \frac{1}{n}\sum(y_i - \hat y_i)^2$.
Differentiate $L$ with respect to $w$, applying the chain rule since $L$
depends on $w$ only through $\hat y$:

$$
\frac{\partial L}{\partial w} = \frac{2}{n}\sum_{i=1}^{n}(\hat y_i - y_i)\,x_i
$$

The $x_i$ is there because the weight multiplies $x$ — the chain rule brings
it down from $\hat y = wx+b$. Do the same for the bias, and the $x_i$
disappears, because a bias is added, not multiplied:

$$
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat y_i - y_i)
$$

You now have two partial derivatives — $\partial L/\partial w$ tells you how
loss changes as you nudge the weight alone, $\partial L/\partial b$ the same
for the bias alone. This pair, one per parameter, is called the
**gradient** — both point toward *worse* loss, which is why the descent
loop moves opposite them.

Add to `02b_gradient_descent/gradient_descent.py`:

```python
def compute_gradients(x, y, w, b):
    """Return (dw, db): partial derivatives of MSE loss w.r.t. w and b, at this w, b."""
    raise NotImplementedError
```

??? tip "Hint — open only when stuck"
    `n = len(x)`, then `y_pred = w * x + b` and `error = y_pred - y` (NumPy
    applies both to every element at once — no loop). `dw = (2 / n) *
    np.sum(error * x)`, `db = (2 / n) * np.sum(error)`. Return `dw, db`.

A hand-derived formula is worth nothing until checked against something that
didn't come from the same derivation. The independent check here is
**numerical differentiation** — the limit definition of a derivative,
evaluated directly instead of symbolically:

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
$$

for a small $h$, instead of taking the limit as $h \to 0$. No calculus in
that calculation at all — nudge $w$ up and down by a tiny $h$, see how much
the loss actually moved, divide by $2h$ — which is exactly what makes it a
fair, independent check on the calculus you did by hand above.

This test checks one point, $w=1, b=1$. It runs your `compute_gradients`
there, then re-derives the same two numbers the nudge-and-divide way:
`loss_at` recomputes loss for any $w$, $b$, and `numerical_dw`/
`numerical_db` apply the formula above directly. Those two `assert` lines
are your real check — if your analytic and numerical gradients agree within
`1e-3`, you derived the gradient correctly, not just confidently.

Save this in `02b_gradient_descent/test_gradient_descent.py` — Build 4 adds a
second test to the same file:

```python
def test_gradient_matches_numerical_check():
    w, b, h = 1.0, 1.0, 1e-5
    dw, db = compute_gradients(x, y, w, b)

    def loss_at(w, b):
        return np.mean((y - (w * x + b)) ** 2)

    numerical_dw = (loss_at(w + h, b) - loss_at(w - h, b)) / (2 * h)
    numerical_db = (loss_at(w, b + h) - loss_at(w, b - h)) / (2 * h)

    assert abs(dw - numerical_dw) < 1e-3
    assert abs(db - numerical_db) < 1e-3
```

??? info "🐍 Python syntax — a function defined inside a function"
    `def loss_at(w, b):` is defined *inside* `test_gradient_matches_numerical_check`,
    so it only exists for the duration of that test. Its own `w`, `b`
    parameters shadow the outer test's `w`, `b` while it runs — but `x` and
    `y` aren't parameters at all; `loss_at` reads them straight from the
    module scope above, the same way the test itself does. It exists here
    purely to avoid repeating the loss formula twice in four lines.

This tests your calculus against a method that never used calculus. If they
disagree, trust the numerical check.

---

## 🔨 Build 4 — The descent loop

Now automate what you'd do by hand: look at the gradient, step opposite it,
repeat.

$$
w \leftarrow w - \eta\,\frac{\partial L}{\partial w}, \qquad
b \leftarrow b - \eta\,\frac{\partial L}{\partial b}
$$

$\eta$ (eta) is the **learning rate** — how big a step you take each time.
It answers question 2: too small, and you crawl toward the answer, needing
thousands of steps; too large, and you overshoot the bottom entirely —
Build 5 shows you something worse than "slow."

Add to `02b_gradient_descent/gradient_descent.py`:

```python
def gradient_descent(x, y, w_init=0.0, b_init=0.0, lr=0.05, epochs=500):
    """Run gradient descent. Return (w, b, loss_history)."""
    raise NotImplementedError
```

One pass through the dataset, followed by one parameter update, is called an
**epoch**. Loop over epochs: predict, measure loss, compute gradients,
update, repeat.

??? tip "Hint — open only when stuck"
    `w, b = w_init, b_init` and `loss_history = []`. Loop `for _ in
    range(epochs):` — each pass, compute `y_pred = w * x + b`, append
    `mse_loss(y, y_pred)` to `loss_history`, *then* get `dw, db =
    compute_gradients(x, y, w, b)` and update `w -= lr * dw`, `b -= lr *
    db`. Record the loss before you update — do it after, and
    `loss_history[0]` becomes your first *updated* line instead of your
    starting guess. Return `w, b, loss_history` once the loop ends.

Call it in `02b_gradient_descent/explore.py` (update the import to
`from gradient_descent import generate_data, mse_loss, gradient_descent`):

```python
w, b, loss_history = gradient_descent(x, y)
print(f"recovered w={w:.4f}, b={b:.4f}")
print(f"loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
```

Run it. You generated this data from $w=3, b=7$ — Build 1's whole point. A
correct loop lands close to those two numbers, having never been told them.

Add this to the same `02b_gradient_descent/explore.py`, right below the
`print` lines above — it reuses the `loss_history` those lines just
produced, so plot it to see the descent happen, not just read the final
number:

```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Loss during training")
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

Run `python explore.py` again. This re-runs the whole file top to bottom —
data, descent loop, and now the plot, all in one pass — so the curve
reflects the exact run you printed above. It should fall fast, then level
off — that flattening answers question 3: the model isn't stuck, it's close
enough now that each step has less room left to improve.

Now finish `test_gradient_descent.py` with the test that checks the whole
loop, not just one piece of it:

```python
def test_gradient_descent_recovers_known_coefficients():
    w, b, loss_history = gradient_descent(x, y)
    assert abs(w - 3.0) < 0.1
    assert abs(b - 7.0) < 0.1
    assert loss_history[-1] < loss_history[0]
```

At the top of the same file, above both tests:

```python
import numpy as np

from gradient_descent import generate_data, compute_gradients, gradient_descent

x, y = generate_data()
```

Run it:

```powershell
pytest
```

Commit:

```powershell
cd C:\dev\bonsai
git add .
git commit -m "Stage 02b: gradient descent on synthetic data"
```

---

## 🔨 Build 5 — Break it on purpose

You have a working descent loop and a sensible learning rate. Add this
temporarily to the bottom of `02b_gradient_descent/explore.py` — you can
delete it once you've seen the numbers, it isn't part of what ships — and
set `lr=10` for 10 epochs:

```python
w, b, loss_history = gradient_descent(x, y, lr=10, epochs=10)
print([f"{v:.3g}" for v in loss_history])
```

??? info "🐍 Python syntax — list comprehension, `[expr for x in iterable]`"
    `[f"{v:.3g}" for v in loss_history]` builds a new list by running
    `f"{v:.3g}"` once for every `v` in `loss_history` — shorthand for a
    `for` loop that appends to an empty list, written as one expression.

Run it — you're back at the repo root after that commit, so `cd` into the
stage folder first:

```powershell
cd 02b_gradient_descent
python explore.py
```

Watch the numbers, not just whether it "worked." The loss goes from about
148 to about 3.5 million in a single step, then keeps multiplying. Each
update doesn't nudge $w$ and $b$ toward the bottom of the bowl — it launches
them past it, further out than where they started, so the *next* gradient is
even bigger. Not a bug: the same formula from Build 4, with a step too large
for this loss surface to survive. This is what "too large" in question 2
costs — not a slower answer, a runaway one.

Read the numbers, then log what you saw.

---

## 🔮 Check your prediction

**Q1 — which way, with no calculus.** Nudge it a little each way and keep
whichever direction lowered the loss — exactly what a derivative tells you,
without the nudging. `compute_gradients` is that instinct, formalised; Build
3's numerical check proved the formal version agrees with it.

**Q2 — how far.** Too small: `lr=0.05` needed 500 epochs to close most of the
gap. Too large: `lr=10` diverged explosively — millions, then billions,
within ten steps. There's a wide safe middle; no formula hands you its exact
edges, you find them by watching the loss curve.

**Q3 — how to know it's done.** The flattening in Build 4's plot. While the
curve drops fast, it's still learning something; once it's nearly flat,
further epochs return less and less — not zero, but small enough that you'd
rather spend the time elsewhere.

---

## ✅ Done when

- `generate_data`, `mse_loss`, `compute_gradients`, and `gradient_descent`
  are all implemented in `02b_gradient_descent/gradient_descent.py`
- `data_scatter.png` exists, and looks like a fuzzy line, not a formula
- `test_gradient_matches_numerical_check` passes — your calculus agrees with
  a method that never used calculus
- `test_gradient_descent_recovers_known_coefficients` passes — the loop
  recovers $w \approx 3$, $b \approx 7$ from data that never mentioned them
- `loss_curve.png` exists, and it falls then flattens
- You've run `lr=10` and watched it diverge, and can say in your own words
  why a bigger step made things worse instead of just slower
- Committed

---

## 🗣️ Tell someone

Explain to a friend, in four sentences, what "training" a model actually
means, using the image of turning a dial while someone tells you warmer or
colder — not the words gradient, loss, or optimisation.

Banned words: gradient, loss, weight, parameter, optimise.

---

## 📓 Log it

Three lines: what you built, what broke, what you would do differently. Add
your recovered $w$ and $b$, and the epoch at which the loss curve visibly
flattened.

---

## 💼 They will ask you this

1. Why square the error instead of just taking the absolute value?
2. What happens to training if the learning rate is too high? Too low?

---

## 🧭 The pattern

Predict, measure how wrong, find which way to move, take a step, repeat.
That loop — not the two-parameter line it's wrapped around here — is what
every model in this journey does to learn anything, including BonsaiGPT with
its millions of parameters. You built the smallest working version of it.

---

## ➡️ Next up

This worked because you cheated, on purpose: you knew the right answer in
advance, since you chose it yourself in Build 1. Real gasoline prices offer
no such guarantee — you don't know the true relationship, if a clean one
even exists, and "loss went down" stops being enough on its own.

Next you point this same machine at the real gasoline data you cleaned back
in Stage 02a, and find out whether "loss went down" was ever the right
thing to watch.
