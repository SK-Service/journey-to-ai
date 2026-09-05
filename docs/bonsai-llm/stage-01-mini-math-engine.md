---
title: Build Your Own Mini Math Engine
stage: "01"
archetype: implement
minutes: 110
new_concepts: [array shape as a contract]
new_tools: []
---

# Stage 01 — Build Your Own Mini Math Engine

## 🔁 Last time

You set up a workshop — Python, a virtual environment, a repo, and one arrow
plotted on screen. Nothing about that arrow was yours to compute; NumPy did
the one line of work involved. This stage is where you stop taking NumPy's
word for it.

## 🎯 What you're making

`math_engine.py` — your own dot product, matrix multiplication and vector
transformation, written from scratch and checked against NumPy. Stage 09 will
import this file. About 110 minutes.

You know this mathematics. This stage is about what a machine has to *do* to carry
it out, which turns out to be a different question with a more interesting
answer.

---

## 🤔 Before you open your editor

Get a piece of paper. No code yet.

1. To multiply $A$ ($2 \times 3$) by $B$ ($3 \times 4$), how many individual
   multiplications happen? Write the number down.
2. Now do the same for two $1000 \times 1000$ matrices. Write that number down
   too, then look at it for a moment.
3. $AB$ needs A's columns to match B's rows. What is the *cheapest* check a
   program could do before starting work, so it fails immediately instead of
   halfway through?
4. You are about to write a function that multiplies matrices. Before you write
   it — what should it do when given shapes that do not fit? Crash? Return
   nothing? Something else?

Question 2 is why GPUs exist. Question 4 is most of the difference between code
that works and code that is usable.

---

## 🔨 Build 1 — Shapes are a contract

Make the folder and the file:

```powershell
cd C:\dev\bonsai
.venv\Scripts\Activate.ps1
mkdir 01_math_engine
```

In `01_math_engine/math_engine.py`:

```python
import numpy as np

scalar = 7.0
vector = np.array([2.0, 5.0])
matrix = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

for name, thing in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"{name:7} {np.shape(thing)}")
```

??? info "🐍 Python syntax — `for x, y in [...]`, f-strings"
    `[("scalar", scalar), ("vector", vector), ("matrix", matrix)]` is a list
    of three pairs. `for name, thing in ...` unpacks each pair into two names
    at once, one iteration per pair — the same idea as `train, val = ...`
    later in this journey, just supplying many pairs instead of one.

    `f"{name:7} ..."` is an f-string: the `f` before the quote means anything
    in `{}` gets evaluated and spliced into the string. The `:7` after `name`
    is a formatting spec — "pad this to 7 characters" — used here so the
    three printed shapes line up in a column.

Run it. The scalar has shape `()`, the vector `(2,)`, the matrix `(2, 3)`.

That empty pair of brackets is worth a moment. A scalar is not a
one-element list — it has *no* dimensions, and a vector of length one has one
dimension of size one. NumPy will let you multiply things whose shapes almost
match, in ways that are convenient right up until they are the reason your model
silently trains on nonsense.

$$
\text{scalar} \to () \qquad
\text{vector} \to (n,) \qquad
\text{matrix} \to (m, n)
$$

From here on, when something breaks, print the shape first. It is the fastest
diagnostic you have and it stays true all the way to Stage 09.

---

## 🔨 Build 2 — The dot product, by hand

Start with the smallest thing that is still real. Add to `math_engine.py`:

```python
def dot(u, v):
    """Dot product of two 1-D arrays of the same length."""
    raise NotImplementedError
```

Write the body with a loop — `sum(...)` and NumPy's own `np.dot` are both off
limits here. Multiply matching entries, add them up, return the total. Raise
a `ValueError` if the lengths disagree.

Now the check. Create `01_math_engine/test_math_engine.py`:

```python
import numpy as np
import pytest
from math_engine import dot


def test_dot_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(1, 8)
        u, v = rng.normal(size=n), rng.normal(size=n)
        assert np.isclose(dot(u, v), np.dot(u, v))


def test_dot_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dot(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
```

??? info "🐍 Python syntax — `from ... import`, `range`, `with ... :`"
    `from math_engine import dot` grabs just the one name `dot` out of your
    `math_engine` module, instead of importing the whole module under a
    prefix. `for _ in range(20)` runs the loop body 20 times; `range(20)`
    counts `0` up to but not including `20`, and `_` is the conventional
    name for a loop variable the body never actually uses.

    `with pytest.raises(ValueError):` opens a block that expects the code
    inside it to raise a `ValueError` — the test passes only if that error
    actually happens, and fails if the code runs without error instead.

??? info "🔧 What this code does — testing against many random cases"
    `rng.integers(1, 8)` picks a random length between 1 and 7 each
    iteration; `rng.normal(size=n)` draws `n` values from a standard normal
    distribution, so `u` and `v` are two random vectors of that same
    random length. Twenty iterations, twenty different random lengths and
    values, and every single one has to match NumPy's `np.dot` — one lucky
    pass would prove nothing, twenty different ones start to. The second
    test checks the opposite case: `dot` must raise, not silently return
    something, when the two vectors don't match in length.

```powershell
cd 01_math_engine
pytest
```

Notice what that first test is doing. It does not contain a single expected
answer. It generates twenty random pairs and demands that your function agree
with NumPy on all of them. You are not checking whether you got one sum right —
you are checking whether your *definition* matches the accepted one.

This pattern is the backbone of the whole journey: **build it yourself, then
check it against something you did not write.**

---

## 🔨 Build 3 — Matrix multiplication, by hand

Same file, next function. Add to `math_engine.py`:

```python
def matmul(A, B):
    """Matrix product of A (m x n) and B (n x p). Returns (m x p)."""
    raise NotImplementedError
```

??? info "🐍 Python syntax — `A[i, k]`, `B[:, j]`"
    `u[i]` from Build 2 indexes a 1-D array with one number. A 2-D array
    needs two, separated by a comma: `A[i, k]` is the entry in row `i`,
    column `k`. A bare `:` in either slot means "every position there" —
    `B[:, j]` is every row, column `j`, i.e. the whole of column `j`. You'll
    write both forms below.

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

You have written that sum before. Turn it into three nested loops: one over
$i$, one over $j$, and the sum over $k$ on the inside. Check the shapes agree
before you start and raise a `ValueError` if they do not.

??? tip "Hint — open only when stuck"
    `np.zeros((m, p))` gives you an output array to fill in. `A.shape` returns
    a tuple, so `m, n = A.shape` unpacks it in one line. The innermost loop is
    exactly your `dot` from Build 2 — row $i$ of $A$ against column $j$ of $B$.
    `B[:, j]` is that column.

Add to the test file:

```python
from math_engine import matmul


def test_matmul_matches_numpy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m, n, p = rng.integers(1, 6, size=3)
        A, B = rng.normal(size=(m, n)), rng.normal(size=(n, p))
        assert np.allclose(matmul(A, B), A @ B)


def test_matmul_rejects_bad_shapes():
    with pytest.raises(ValueError):
        matmul(np.zeros((2, 3)), np.zeros((4, 5)))
```

??? info "🐍 Python syntax — `@`, matrix multiplication"
    `A @ B` is Python's dedicated operator for matrix multiplication —
    introduced specifically so NumPy doesn't have to overload `*`, which
    instead multiplies two arrays element by element. `A @ B` and `A * B`
    both run without erroring and produce completely different, unrelated
    results, so mixing them up is a silent-bug risk worth remembering.

??? info "🔧 What this code does — same test, random shapes this time"
    `rng.integers(1, 6, size=3)` draws three random integers at once — `m`,
    `n`, `p` — so every one of the 20 iterations tests a different pair of
    shapes, not just different values. `rng.normal(size=(m, n))` extends
    Build 2's `size=n` to two dimensions: `A` is `m` rows by `n` columns,
    `B` is `n` rows by `p` columns — shapes compatible with each other by
    construction, matching Build 1's shape discussion. `np.allclose` is
    `np.isclose` from Build 2, extended to compare every entry of two
    arrays at once instead of two single numbers.

Run it: `pytest`. Four tests, four passes.

Then measure what it cost you. Add to `math_engine.py`:

```python
import time

rng = np.random.default_rng(2)
A, B = rng.normal(size=(200, 200)), rng.normal(size=(200, 200))

t0 = time.perf_counter(); matmul(A, B); mine = time.perf_counter() - t0
t0 = time.perf_counter(); A @ B;       theirs = time.perf_counter() - t0

print(f"mine   {mine:.3f}s")
print(f"numpy  {theirs:.4f}s")
print(f"ratio  {mine / theirs:.0f}x")
```

??? info "🐍 Python syntax — `;` on one line"
    `t0 = time.perf_counter(); matmul(A, B); mine = time.perf_counter() - t0`
    is three statements written on one line, separated by `;` instead of
    three lines. It works anywhere, but is rarely used — here only to keep
    the "start the clock, do the work, read the clock" sequence visually
    together. `{mine:.3f}` is the same formatting spec as `{name:7}` earlier,
    this time meaning "3 digits after the decimal point."

??? info "🔧 What this code does — timing the two versions"
    `A @ B` reuses the same operator from the test above — this is NumPy
    doing the identical multiplication, timed the same way your `matmul`
    is. `time.perf_counter()` returns a high-precision clock reading;
    calling it before and after some work and subtracting gives you how
    long that work took.

    `.4f` and `.0f` extend the `.3f` format spec from the note above the
    same way — 4 digits after the decimal for `theirs` (NumPy is fast
    enough that you need the extra digit to see it isn't zero), and 0
    digits for the ratio (a big number, rounded to the nearest whole one).
    The `s` and `x` right after each closing `}` aren't part of the
    formatting at all — they're literal characters typed into the string,
    short for "seconds" and "times."

Write that ratio in your log. Your loops are correct — and expect the ratio
to be large. Two, three, even four orders of magnitude slower is normal for
this comparison, not a sign you did something wrong. NumPy hands the work to
compiled linear-algebra code that people have been optimising for decades;
your version pays Python's per-operation overhead eight million times over
for a 200×200 multiply, and that overhead dominates.

This is why you write things from scratch once, to understand them, and then use
the library. Both halves matter, and by Stage 11 you will be glad you did the
first half.

---

## 🔨 Build 4 — Watch a matrix move something

A matrix is not a grid of numbers. It is a thing that *does* something to space.
Add to `math_engine.py`:

```python
import matplotlib.pyplot as plt

square = np.array([[0.0, 1.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 1.0, 0.0]])

M = np.array([[2.0, 0.5],
              [0.0, 1.0]])

moved = matmul(M, square)

plt.plot(square[0], square[1], label="before")
plt.plot(moved[0], moved[1], label="after")
plt.axis("equal"); plt.grid(True); plt.legend()
plt.show()
```

??? info "🔧 What this code does — plotting before and after"
    `square` and `moved` are each a `(2, 5)` array — row 0 holds every
    x-coordinate, row 1 holds every y-coordinate, for the five corners
    that trace the square (the last point repeats the first, to close the
    shape). `plt.plot(square[0], square[1], label="before")` draws one
    line connecting those five points in order; the second `plt.plot`
    does the same for `moved`, the result of your own `matmul`.
    `plt.axis("equal")` forces both axes to use the same scale, so a right
    angle actually looks like one instead of being stretched; `plt.legend()`
    shows the `label=` text for each line so you can tell which is which.

Your own `matmul` drew that.

Now try three of your own: one that rotates, one that reflects, one that
squashes the square flat onto a line. Predict the picture before you run each
one, then run it.

For the flat one, compute `np.linalg.det(M)` and note what the determinant is.

You know algebraically that a zero determinant means `M` has no inverse.
Look at the picture and see what that means physically: your square
collapsed onto a line, so many different points in the original square now
land on the exact same point on that line. Given only where a point ended
up, you cannot tell which of those original points it came from — the
information about *which one* is gone. That is what "no inverse" means
here: no matrix could undo the collapse and hand back the original square,
because undoing it would mean deciding, for every point on the line, which
one of infinitely many original points to send it back to.

Commit your work:

```powershell
cd C:\dev\bonsai
git add .
git commit -m "Stage 01: dot, matmul and a transformation, checked against NumPy"
```

---

## 🔮 Check your prediction

Go back to the four questions you answered on paper.

**Q1 and Q2 — the multiplication counts.** For $A$ ($2\times3$) times $B$
($3\times4$): each of the $2\times4=8$ output entries costs 3 multiplications,
so $24$ total. For two $1000\times1000$ matrices: $1000^3 = 1{,}000{,}000{,}000$
— a billion. If your numbers matched, your mental model of the triple loop was
already correct before you wrote a line of it. If they did not, look at which
dimension you double-counted — it is almost always the shared inner one.

**Q3 — the cheapest check.** Compare `A.shape[1]` to `B.shape[0]` before
touching a single element. That is exactly what your `matmul`'s `ValueError`
does, and it is why `test_matmul_rejects_bad_shapes` exists — the cheapest
check and the correctness check are the same line of code.

**Q4 — what to do on a mismatch.** You wrote `raise ValueError`, not "return
nothing" or "crash with whatever Python gives you by default." A caller can
catch a `ValueError` and decide what to do. A silent wrong answer is far more
dangerous than a loud crash — it is the difference between a bug you find in
five minutes and one you find in production.

---

## 🔎 Go find out

**Question:** Your `matmul` and NumPy's compute the same thing. Where did the
speed difference actually come from?

**Search phrases:**

- `what is BLAS linear algebra`
- `why is numpy matrix multiplication fast`
- `site:numpy.org internals c api`

**Follow-ups to answer in your log:**

- What do the letters BLAS stand for, and roughly how old is it?
- Your loops run one multiplication at a time. What does it mean for a
  processor to do several at once, and what is that called?
- Naive matrix multiplication of two $n \times n$ matrices costs about $n^3$
  operations. Is there an algorithm that does it in fewer? What is it called,
  and why is it not used everywhere?
- Your laptop has no GPU. What would change if it did, and why does that
  hardware suit this particular operation so well?

**Time-box:** 25 minutes.

**Verification:** the third follow-up has a clean answer with a name attached.
If you found it, you found the right thread. If you did not, search for
`matrix multiplication faster than n cubed` and you will land on it directly.

---

## 🤖 Second opinion

> I wrote matrix multiplication with three nested loops and it was about 300
> times slower than NumPy's. Here is my explanation of why — don't tell me the
> answer, poke holes in it. What did I get wrong, and what am I not asking?

Check anything it claims about complexity against a second source. Models are
confidently wrong about exponents more often than you would expect, which is
itself worth knowing about the thing you are spending this year building.

---

## ✅ Done when

- `pytest` passes all four tests in `01_math_engine/`
- `dot` and `matmul` use loops, not NumPy shortcuts
- Both raise `ValueError` on shapes that do not fit
- Your speed ratio is in `LOG.md`
- You have plotted a transformation that flattens the square, and know its
  determinant
- Committed with a message naming what you built

---

## 🗣️ Tell someone

Explain to a sibling, in **five sentences**, what a matrix does — using the
squashed-square picture and nothing else.

Banned words: matrix, vector, linear, transformation, dimension.

---

## 📓 Log it

Three lines in `LOG.md`: what you built, what broke, what you would do
differently. Include the speed ratio.

---

## 💼 They will ask you this

Two questions from the end of the journey, tied to what you built here:

1. Your matrix multiplication was correct and dramatically slower than
   NumPy's — likely by several orders of magnitude. Where did the time go?
2. Your test compares against NumPy. What class of bug would that test never
   catch?

The second one is worth real thought. Write your answer down — you will want it
again in Stage 12.

---

## 🧭 The pattern

Build it yourself, then check it against something you did not write — a
NumPy call, a known synthetic answer, a numerical approximation. You will run
this exact pattern in Stage 04 against gradient checking, in Stage 08 against
the causal mask, and everywhere between. It is not specific to matrices; it is
how you verify anything you build from scratch for the rest of this journey.

---

## ➡️ Next up

You can transform vectors now. But every number in those matrices, you chose
yourself.

The whole point of machine learning is that the machine chooses them, by
looking at data and adjusting until it is less wrong. Next you get the
data — real gasoline prices, downloaded, cleaned and plotted. No models yet.
Just the honest and slightly tedious business of finding out what the numbers
actually look like, which is where most of the real work in this field turns
out to live.
