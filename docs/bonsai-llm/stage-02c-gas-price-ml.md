---
title: Point It at the Real World
stage: "02c"
archetype: data
minutes: 105
new_concepts: [overfitting, residual, regime change]
new_tools: []
---

# Stage 02c — Point It at the Real World

## 🔁 Last time

You proved your gradient descent loop by recovering numbers you invented
yourself — reassuring, but rigged, since you already knew the answer. This
time the same loop meets real gasoline prices, and nobody hands you an
answer key.

## 🎯 What you're making

A linear regression trained on real weekly gasoline prices, using the last
few weeks as features, with an honest verdict on whether it beats the
simplest possible guess. About 105 minutes.

Almost every function this stage needs already exists in your repo. The new
problem isn't writing code — it's that nobody hands you a known answer this
time, so you have to decide, honestly, whether what you built is any good.

---

## 🤔 Before you open your editor

Paper again. This time the questions are about the data, not the algorithm.

1. You'll feed your model the last few weeks of prices to predict next
   week's. Write down exactly which weeks that includes. Then check: on
   the day you'd actually be making that prediction, would any of those
   weeks still be unknown to you?
2. Before training anything: if your model beat "next week equals this
   week," would that surprise you? If it lost, would that surprise you?
3. If you gave the model twenty past weeks instead of three, would it get
   better at predicting weeks it has never seen, or worse? If your answer
   is "it depends," name what it depends on.
4. Your model will be badly wrong somewhere in the validation set. Before
   seeing a single number, name three fundamentally different reasons one
   prediction could miss.

---

## 🔨 Build 1 — Feature engineering: lagged prices

`load_prices` and `time_split` from 02a already take you from raw data to a
clean, honestly-split table. Reuse them instead of writing them again.

```powershell
cd C:\dev\bonsai
mkdir 02c_gas_price_ml
Copy-Item 02a_data\raw 02c_gas_price_ml\raw -Recurse
Copy-Item 02a_data\load.py 02c_gas_price_ml\load.py
```

??? info "💻 Command line — `Copy-Item`"
    `Copy-Item <source> <destination>` copies a file. A folder needs
    `-Recurse` too, so every file inside comes along — leave it off and
    PowerShell silently copies nothing, no error.

The shape of the problem:

$$
[P_t, P_{t-1}, P_{t-2}] \rightarrow P_{t+1}
$$

Three consecutive weekly prices go in, the following week's price comes
out. The numbers you feed a model are its **features** — here, the three
prices on the left. The number it's trying to predict is the **target** —
the price on the right. Every model in this journey, all the way to
BonsaiGPT, comes down to some choice of features predicting some target.

??? info "Design of make_features"
    The idea: turn the single `price` column into a table where each row
    already carries the three most recent prices *and* the future price
    you're trying to predict — so that training a model later is just
    reading straight across one row.

    Take a tiny made-up price series to see how, one row per week, already
    sorted oldest to newest:

    | week | price |
    |---|---|
    | 0 | 2.00 |
    | 1 | 2.10 |
    | 2 | 2.20 |
    | 3 | 2.15 |
    | 4 | 2.25 |
    | 5 | 2.30 |

    `df["price"].shift(i)` builds a new column aligned to the *same* row
    index as `price`, but reading `i` rows further back. Before dropping
    anything, row `i` ends up holding: `lag0` = price at week `i` (shift by
    `0` — no movement at all), `lag1` = price at week `i-1`, `lag2` = price
    at week `i-2`, `target` = price at week `i+1` (`shift(-1)` — one row
    into the future).

    So row 2 becomes `lag0=2.20, lag1=2.10, lag2=2.00, target=2.15`. The
    first two rows can't have a full `lag2` (there's no week `-1` or `-2`),
    and the last row can't have a `target` (there's no week after it) —
    that's exactly what `dropna` removes. What's left:

    | row (week) | lag0 | lag1 | lag2 | target |
    |---|---|---|---|---|
    | 2 | 2.20 | 2.10 | 2.00 | 2.15 |
    | 3 | 2.15 | 2.20 | 2.10 | 2.25 |
    | 4 | 2.25 | 2.15 | 2.20 | 2.30 |

    Each row is one week in time. Consecutive rows aren't independent
    examples — the window just slides forward by one week at a time, so
    row 3's `lag1` and `lag2` are literally row 2's `lag0` and `lag1`, one
    slot down. Look even closer: row 2's `target` (2.15) is exactly row
    3's `lag0` (2.15) — the same price shows up twice, once as "the
    answer" for one row and once as "this week's price" for the very next
    one, because next week's price *is* both things at once. That overlap
    is also why `lag0`, `lag1`, and `lag2` end up so correlated with each
    other once you run this on real data — worth remembering when Build 2
    talks about a slow, collinear descent, and when 🔮 asks whether twenty
    lag columns would actually help.

Create `02c_gas_price_ml/features.py`:

```python
def make_features(df, n_lags=3):
    """Add n_lags lagged price columns and a next-week target column. Drop rows with any missing value."""
    raise NotImplementedError
```

`df` is whatever `load_prices` returned, and `n_lags=3` sets how many
lagged columns to build — see the design box above for exactly how the
shifting and dropping works. Build the columns in a loop, name them
`lag0`, `lag1`, `lag2`, add `target` the same way, then drop any row
missing either.

??? tip "Hint — open only when stuck"
    The plan: loop `n_lags` times, each pass shifting `price` by one more
    step and stashing the new column's name in `cols`; add `target` as one
    more shift, in the opposite direction; then drop any row missing one
    of those columns and renumber what's left before returning.
    ```python
    cols = []
    for i in range(n_lags):
        col = f"lag{i}"
        df[col] = df["price"].shift(i)
        cols.append(col)
    df["target"] = df["price"].shift(-1)
    return df.dropna(subset=cols + ["target"]).reset_index(drop=True)
    ```
    `cols + ["target"]` joins your list of lag-column names onto one more
    name — Python's `+` concatenates two lists end to end, the same
    operator that adds numbers. `dropna(subset=...)` only checks the
    columns you name, not the whole row. `reset_index(drop=True)` renumbers
    the surviving rows from `0`, since `dropna` leaves gaps in the old
    numbering.

Call it in `02c_gas_price_ml/explore.py`:

```python
from load import load_prices, time_split
from features import make_features

df = load_prices("your_downloaded_filename.csv")
print(df.head())                    # first 5 rows of the raw prices

df = make_features(df, n_lags=3)    # reuse make_features to build the lag/target columns
print(df[["date", "lag0", "lag1", "lag2", "target"]].head())  # select these columns by name; first 5 rows of the feature table

train, val = time_split(df)         # val is entirely after train, chronologically

feature_cols = ["lag0", "lag1", "lag2"]    # the model's inputs: the three lag columns
X_train, y_train = train[feature_cols].values, train["target"].values  # training features, training targets
X_val, y_val = val[feature_cols].values, val["target"].values          # same split, for validation

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print(X_train[:3])
print(y_train[:3])
print(X_val[:3])
print(y_val[:3])
```

Pay attention to every print above — together they trace the data through
each stage: raw prices, then the feature table (the design box above, for
real, on your own data), then the plain NumPy arrays a model actually
trains on. `train[feature_cols]` selects the three lag columns at once;
`.values` peels the numbers out of the DataFrame into an array — the
gradient descent code you're about to reuse from 02b works on arrays, not
column names. Notice the shapes: `X_train`/`X_val` are 2-D — one row per
week, one column per feature — while `y_train`/`y_val` stay 1-D, one
number per week, since there's only ever one target.

Run it:

```powershell
cd 02c_gas_price_ml
python explore.py
```

Expect roughly 1,500 training rows and a few hundred validation rows — the
exact count depends on how many weeks were published when you downloaded
the file, and a handful vanish at each end since a two-week-back lag needs
two prior rows to exist.

Every Build below keeps adding to this same `explore.py`, so `X_train`,
`y_train`, `X_val`, and `y_val` stay loaded for the rest of the stage.

---

## 🔨 Build 2 — Train using 02b's descent loop

02b's math assumed one feature (just `x`, the single number each
generated point had): $\hat y = wx + b$. You now have three, so
the prediction is a weight *vector* dotted against each row of features:

$$
\hat y = Xw + b
$$

where $X$ is your `(rows, 3)` array and $w$ is a length-3 vector — Stage 1's
matrix multiplication, finally doing real work instead of proving itself
against NumPy. The loss is the same MSE formula as always, and the
gradients generalize the same way 02b's did, carried by a matrix instead of
a single column:

$$
\frac{\partial L}{\partial w} = \frac{2}{n} X^T(\hat y - y), \qquad
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(\hat y_i - y_i)
$$

$X^T$ (transpose) flips rows and columns so the shapes line up — it plays
the exact role `np.sum(error * x)` played in 02b, summing across all three
features into one number per feature at once. Same four steps as 02b:
predict, measure, gradient, step. Only the shapes changed.

Create `02c_gas_price_ml/model.py`:

```python
import numpy as np


def mse_loss(y_true, y_pred):
    """Mean squared error between two equal-length arrays."""
    return np.mean((y_true - y_pred) ** 2)
```

Identical to 02b's — the formula doesn't care whether $\hat y$ came from one
feature or three. Add to the same file:

```python
def compute_gradients(X, y, w, b):
    """Return (dw, db): partial derivatives of MSE loss w.r.t. w (a vector) and b."""
    raise NotImplementedError
```

??? tip "Hint — open only when stuck"
    Here's what each line does: `n = len(y)` counts the observations, since
    both gradient formulas divide by it. `y_pred = X @ w + b` computes
    $\hat y$ — the model's current prediction for every row at once, using
    today's `w` and `b`. `error = y_pred - y` computes $\hat y - y$ for
    every row simultaneously — how wrong each prediction is, and in which
    direction. `dw = (2 / n) * (X.T @ error)` is the $\partial L/\partial w$
    formula from above: `X.T @ error` is the matrix multiplication that
    sums each feature's contribution across every row at once, and
    `(2 / n)` finishes the formula. `db = (2 / n) * np.sum(error)` is
    $\partial L/\partial b$ the same way — except there's no feature to
    multiply against, just the plain sum of the errors. Return both, in
    the order the docstring promises.
    ```python
    n = len(y)
    y_pred = X @ w + b
    error = y_pred - y
    dw = (2 / n) * (X.T @ error)
    db = (2 / n) * np.sum(error)
    return dw, db
    ```

A formula is worth nothing until checked against numbers you worked out by
hand. Let's write that check. Save this in `02c_gas_price_ml/test_model.py`:

```python
def test_gradient_matches_hand_worked_example():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 11.0])
    w, b = np.array([1.0, 1.0]), 0.0

    dw, db = compute_gradients(X, y, w, b)

    assert np.allclose(dw, [-14.0, -20.0])
    assert db == -6.0
```

`X` has two rows and two columns — two separate observations, each carrying
two feature values. Row one, `[1.0, 2.0]`, is one made-up example; row two,
`[3.0, 4.0]`, is a second, independent one. Two rows is the minimum needed
to prove the formula sums correctly across more than a single observation,
and still small enough to do on paper. `y` holds the actual answer for
each: `5.0` for row one, `11.0` for row two.

`w, b = [1.0, 1.0], 0.0` isn't the trained answer — it's a point you're
choosing on purpose, picked because 1s and a 0 keep the arithmetic simple.
At that point, $\hat y = Xw + b$ gives row one $1(1) + 2(1) + 0 = 3$ and row
two $3(1) + 4(1) + 0 = 7$.

Compare those to the actual values: row one is off by $3 - 5 = -2$, row two
by $7 - 11 = -4$. Run those two errors through Build 2's gradient formulas
and you land on exactly `[-14.0, -20.0]` and `-6.0` — that's where the
`assert` values come from, worked out independently of the code you're
about to test, not guessed or copied from a run. If your code disagrees,
the bug is in `compute_gradients`, not in this test.

Add one more function to `02c_gas_price_ml/model.py`. Now build the
function that actually trains: predict, measure, find the gradient, take a
step — repeated until the weights settle. `X` and `y` are the features and
targets you already built. `w_init` and `b_init` are where the search
starts — both default to `0.0`, the flattest possible guess, same as 02b.
`lr` is the step size, and `epochs` is how many times the whole
predict-measure-step cycle repeats.

```python
def gradient_descent(X, y, w_init=0.0, b_init=0.0, lr=0.01, epochs=3000):
    """Run gradient descent. Return (w, b, loss_history)."""
    raise NotImplementedError
```

??? tip "Hint — open only when stuck"
    The plan: start from the initial guess, then repeat `epochs` times —
    predict with the current `w` and `b`, record how wrong that prediction
    was, compute the gradient, and nudge `w` and `b` a small step in the
    direction that reduces the loss. Once the loop ends, return the final
    weights along with everything recorded along the way.

    `np.zeros(X.shape[1]) + w_init` builds a starting weight vector the
    same length as your feature count — one entry per feature, since
    `X @ w` needs a weight for every column of `X` to line up. This is why
    `w_init` stays a plain number instead of a vector you'd have to size
    yourself: with three features, that's `np.zeros(3) + 0.0`, still the
    flattest possible starting line, just now in three dimensions instead
    of one.

    `loss_history` exists because a single loss number after training
    tells you almost nothing — you need to watch it change over time to
    know whether the loop is actually working. Every entry is the loss at
    one epoch, recorded before that epoch's update (same as 02b), so by
    the end you have the whole trajectory: where it started, how fast it
    fell, and where it leveled off. This is what "loss went down" actually
    looks like in code, and it's exactly the list Build 4 plots a few
    pages from now to check for overfitting.
    ```python
    w = np.zeros(X.shape[1]) + w_init
    b = b_init
    loss_history = []
    for _ in range(epochs):
        y_pred = X @ w + b
        loss_history.append(mse_loss(y, y_pred))
        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
    return w, b, loss_history
    ```

Call it in `02c_gas_price_ml/explore.py` (add the import
`from model import mse_loss, compute_gradients, gradient_descent`):

```python
w, b, loss_history = gradient_descent(X_train, y_train)
print("w:", w, "b:", round(b, 4))
print("loss:", round(loss_history[0], 4), "->", round(loss_history[-1], 4))
```

Run it. Loss should fall from around 5 to under 0.005 — three collinear lag
columns make a slow, narrow bowl to descend, which is why this run needs
more epochs than 02b's single-feature one did.

Now finish `02c_gas_price_ml/test_model.py` with a second test, checking
the whole loop rather than one gradient at a single point:

??? info "Idea: proving the whole loop recovers a known answer"
    The earlier test checked one calculation, by hand, at one point. This
    one checks the entire training loop, end to end — the same trick as
    02b's Build 1: generate data from a `w` and `b` you chose yourself, so
    there's a known right answer to check the trained result against.

    `rng = np.random.default_rng(0)` seeds the randomness, so this test
    produces the exact same data — and therefore the exact same result —
    every time it runs. `n = 500` gives the descent loop enough data to
    average out the noise you're about to add; too few points and one
    unlucky value could throw the fit off. `true_w, true_b = [2.0, -1.0,
    0.5], 3.0` are the answer you're hiding — three arbitrary numbers with
    nothing to do with gasoline prices, chosen only because the test needs
    *some* ground truth.

    `X = rng.uniform(-5, 5, size=(n, 3))` builds 500 rows of three random,
    independent features — unlike the real lagged prices, these three
    columns have nothing to do with each other, on purpose, so this test
    isn't fighting the same collinearity the real data has. `y = X @
    true_w + true_b + rng.normal(0, 0.5, size=n)` builds the targets by
    running the *true* formula forward — exactly what `gradient_descent`
    is trying to discover in reverse — then adds a little noise so it
    isn't a trivial, noise-free fit.

    `w, b, loss_history = gradient_descent(X, y, lr=0.01, epochs=3000)`
    then runs your actual training loop on this made-up data, knowing
    nothing about how it was generated. The three asserts check three
    different things: `np.allclose(w, true_w, atol=0.1)` and `abs(b -
    true_b) < 0.1` check that the recovered weights and bias land close to
    the numbers you hid — within `0.1`, not exact, since gradient descent
    approaches an answer rather than teleporting to it. `loss_history[-1]
    < loss_history[0]` is a weaker, independent check — did the loss
    actually fall at all — useful because it would still catch a broken
    loop even if the tolerance above ever needed loosening.

```python
def test_gradient_descent_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 500
    true_w, true_b = np.array([2.0, -1.0, 0.5]), 3.0
    X = rng.uniform(-5, 5, size=(n, 3))
    y = X @ true_w + true_b + rng.normal(0, 0.5, size=n)

    w, b, loss_history = gradient_descent(X, y, lr=0.01, epochs=3000)

    assert np.allclose(w, true_w, atol=0.1)
    assert abs(b - true_b) < 0.1
    assert loss_history[-1] < loss_history[0]
```

At the top of the same file, above both tests:

```python
import numpy as np

from model import compute_gradients, gradient_descent
```

Run it:

```powershell
pytest
```

---

## 🔨 Build 3 — Compare against the baseline

Before you run this: expect to lose. 02a already measured how little
gasoline prices move week to week, and "next week equals this week" is
exactly the **baseline** that measurement predicts will be hard to beat.
Losing here isn't a bug — it's the finding, worth stating honestly rather
than hiding.

Re-anchor first if you're returning in a fresh terminal:

```powershell
cd C:\dev\bonsai
cd 02c_gas_price_ml
```

Add to `02c_gas_price_ml/explore.py`:

??? info "Idea: a fair, apples-to-apples comparison"
    The whole point of this block is to score two different predictions
    against the *same* validation targets, with the *same* loss function,
    so the comparison actually means something — you can't tell whether a
    model is any good without something honest to measure it against.

    `baseline_pred = X_val[:, 0]` is the naive guess: column `0` of
    `X_val` is `lag0`, this week's price, used unmodified as the guess for
    next week — "next week equals this week," written as data instead of
    a sentence. `model_pred = X_val @ w + b` is the actual model's guess,
    using the `w` and `b` Build 2 already trained.

    `baseline_mse = mse_loss(y_val, baseline_pred)` and `model_mse =
    mse_loss(y_val, model_pred)` score both predictions the exact same
    way, against the exact same `y_val` — the only fair comparison, since
    a different loss or a different slice of data could make either one
    look artificially better. The three prints report the two numbers and
    then answer the actual question directly: `model_mse < baseline_mse`
    is `True` only if the model actually earned its keep.

```python
baseline_pred = X_val[:, 0]
model_pred = X_val @ w + b

baseline_mse = mse_loss(y_val, baseline_pred)
model_mse = mse_loss(y_val, model_pred)

print("baseline val MSE:", round(baseline_mse, 5))
print("model val MSE:   ", round(model_mse, 5))
print("model beat baseline?", model_mse < baseline_mse)
```

Run it:

```powershell
python explore.py
```

The model should lose, by roughly a factor of two. Train it for
dramatically longer and it eventually closes the gap, but that's not
today's point: at a reasonable training budget, "loss went down" was not
enough to beat a one-line rule — exactly what 02b's closing line warned you
to go check.

---

## 🔨 Build 4 — Train vs validation loss curves

**Overfitting** means training loss keeps falling while validation loss
rises — the model is memorizing quirks of the weeks it trained on instead
of learning something that transfers to weeks it hasn't seen. Watch for it
by tracking both curves at once, not just the one you're training against.

Add to `02c_gas_price_ml/explore.py` (add the import
`import numpy as np` too — this file hasn't needed it directly until now):

??? info "Idea: watching both curves at once"
    The overall idea: train the same way as always, but at every single
    epoch, measure the loss twice — once on the data being trained on,
    once on data the model has never seen — and keep both running lists
    so they can be drawn as two lines on one chart. A single final number
    can't show *when* a model starts memorizing instead of learning; a
    curve over time can.

    `w2, b2 = np.zeros(X_train.shape[1]), 0.0` and `train_hist, val_hist =
    [], []` set up a fresh starting point and two empty lists to fill, one
    per curve. Inside the loop, `train_hist.append(...)` and
    `val_hist.append(...)` record both losses *before* that epoch's
    update — same convention as `gradient_descent`'s own `loss_history` —
    using the current `w2`, `b2` against `X_train`/`y_train` and
    `X_val`/`y_val` respectively. `dw, db = compute_gradients(...)` and
    the two update lines are exactly `gradient_descent`'s own step,
    reused directly rather than reimplemented, since the only thing new
    here is the second measurement, not the training itself.

    The plotting builds up the way you'd describe it out loud. There's no
    explicit `plt.figure(...)` here, unlike Build 5's chart — matplotlib
    quietly opens a default canvas the moment the first `plt.plot(...)`
    runs, so skipping it isn't a mistake, just less control over size.
    `plt.plot(train_hist, label="train")` and `plt.plot(val_hist,
    label="validation")` draw the two curves onto that canvas, each with
    a name for the legend to use later. `plt.xlabel`/`plt.ylabel` label
    what each axis actually means. `plt.legend()` turns the two `label=`s
    into the on-chart key that tells the curves apart. `plt.title(...)`
    names the chart itself. `plt.savefig(...)` writes it to a file —
    `dpi=150` for sharpness, `bbox_inches="tight"` to trim the empty
    border — and `plt.show()` still opens it on-screen too.

```python
import numpy as np
import matplotlib.pyplot as plt

w2, b2 = np.zeros(X_train.shape[1]), 0.0
train_hist, val_hist = [], []
for _ in range(3000):
    train_hist.append(mse_loss(y_train, X_train @ w2 + b2))
    val_hist.append(mse_loss(y_val, X_val @ w2 + b2))
    dw, db = compute_gradients(X_train, y_train, w2, b2)
    w2 -= 0.01 * dw
    b2 -= 0.01 * db

plt.plot(train_hist, label="train")
plt.plot(val_hist, label="validation")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.title("Train vs validation loss")
plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
plt.show()
```

Run it:

```powershell
python explore.py
```

Look at the two curves. They should fall together, validation consistently
above train but never rising while train keeps dropping — not the
overfitting shape. Three weights and three heavily correlated inputs don't
give this model enough room to memorize training-set noise; there's nothing
to overfit *with*. That changes once a model gets far more knobs than this
one has — exactly what Stage 04 adds.

Commit:

```powershell
cd C:\dev\bonsai
git add .
git commit -m "Stage 02c: linear model on real gasoline prices, vs baseline"
```

---

## 🔨 Build 5 — Diagnose your worst miss

Average error hides the interesting cases. Find your single worst one
instead.

Re-anchor first if you're returning in a fresh terminal:

```powershell
cd C:\dev\bonsai
cd 02c_gas_price_ml
```

Add to `02c_gas_price_ml/explore.py`:

??? info "Idea: finding one bad prediction, not the average one"
    Build 3 already told you the model's *overall* score against the
    baseline — one number, averaged across every validation row. That
    number can't tell you where the model struggles; it just tells you
    whether it does, on average. The idea here is different: find the
    single row where the model was most wrong, and go find out why.

    `model_pred` is worth a reminder if you've lost track of it — it's
    from Build 3, `model_pred = X_val @ w + b`, the model's prediction for
    every validation row using the `w` and `b` Build 2 trained. It's
    still sitting in memory because `explore.py` keeps accumulating
    across every Build.

    `residuals = y_val - model_pred` computes actual minus predicted, for
    every row at once — the standard tool for this exact question. You'd
    reach for residuals any time an aggregate score isn't enough and you
    need to know *which* specific points a model handles well or badly,
    not just how well it does on average.

    `worst_idx = np.argmax(np.abs(residuals))` finds the position of the
    single largest miss. `np.abs(...)` comes first because size is what
    matters here, not direction — a huge overprediction and a huge
    underprediction are equally bad misses. `np.argmax` then returns
    *where* that biggest value sits, not the value itself, so that index
    can be used to look up everything else about that one row.

    The three prints turn that bare index into something you can actually
    investigate: the real calendar date the miss happened on (so you can
    go look up what was happening in the world that week), what actually
    happened versus what the model guessed, in real dollars — not an
    abstract loss number — and the size of the miss itself. This is what
    the rest of Build 5 is built on.

```python
residuals = y_val - model_pred
worst_idx = np.argmax(np.abs(residuals))

print("worst miss on:", val["date"].iloc[worst_idx].date())
print("actual:", y_val[worst_idx], "predicted:", round(model_pred[worst_idx], 3))
print("residual:", round(residuals[worst_idx], 3))
```

Run it. As of this writing you should land on a week in late February 2022,
missed by roughly $0.55 — a large gap for a series that usually moves a few
cents. If your download lands you on a different date instead, that's
fine — the worst week can shift slightly as new data is published each
week. Apply the same three-bucket logic to whatever date you actually get.

Look up what happened in the real world on that date. Then classify the
miss into exactly one of three buckets: a feature the model never had
access to; a **regime change**, where the underlying relationship itself
shifted, not just the data; or irreducible noise, given how close to a
random walk this series already is. Investigative only — write down your
conclusion, don't retrain anything in response.

??? tip "Hint — open only when stuck"
    Search the date plus "gasoline price" or "oil price." Late February
    2022 is when Russia's invasion of Ukraine sent crude oil above
    $100 a barrel in days, and EIA's own reporting names the resulting
    jump in retail gasoline prices as the largest month-over-month
    increase on record. Nothing in three columns of past prices could see
    a geopolitical shock coming — this lands squarely in the first bucket,
    a missing feature, not a regime change and not noise.

Add one more block to `02c_gas_price_ml/explore.py`, finishing with a chart
worth showing someone:

```python
plt.figure(figsize=(10, 4))
plt.plot(val["date"], y_val, label="actual", linewidth=0.9)
plt.plot(val["date"], model_pred, label="predicted", linewidth=0.9)
plt.scatter(val["date"].iloc[worst_idx], y_val[worst_idx], color="red", zorder=5, label="worst miss")
plt.xlabel("date")
plt.ylabel("price (USD/gallon)")
plt.legend()
plt.title("Validation: actual vs predicted, worst miss marked")
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
```

??? info "🔧 What this code does — actual vs predicted, worst miss marked"
    Planning a chart like this means answering three questions before
    writing any matplotlib: what belongs on the same axes so it's directly
    comparable (actual and predicted price, both against date), what needs
    to stand out from everything else on the chart (the one worst miss, in
    a different colour, drawn on top), and what a viewer needs in order to
    read it without you standing next to them (axis labels, a legend, a
    title).

    `plt.figure(figsize=(10, 4))` opens a wide canvas — a multi-year price
    series reads better wide than square. The two `plt.plot(...)` calls
    draw the actual and predicted lines on those same axes, each carrying
    a `label=` for the legend to pick up later. `plt.scatter(...)` adds one
    more point on top, for the worst miss alone — `color="red"` makes it
    visually distinct from both lines, and `zorder=5` (higher than the
    lines' default) makes sure it's drawn on top of them, not hidden
    underneath. `plt.xlabel`/`plt.ylabel`/`plt.title` label the axes and
    the chart; `plt.legend()` turns every `label=` above into the on-chart
    key. `plt.savefig(...)` writes the file to disk; `plt.show()` still
    opens it on-screen too.

Run `python explore.py` again, then commit:

```powershell
cd C:\dev\bonsai
git add .
git commit -m "Stage 02c: diagnose worst validation miss"
```

---

## 🔮 Check your prediction

**Q1 — which weeks are allowed.** All three lag columns are already in the
past relative to the week you're predicting — `time_split` from 02a is what
actually enforces this, not discipline you have to remember.

**Q2 — would winning or losing surprise you.** Build 3 answered this with a
number: the model lost by roughly 2×. If 02a's "prices barely move week to
week" finding stuck with you, neither outcome should have surprised you.

**Q3 — more lagged weeks.** Build 4 showed a 3-weight model that isn't
overfitting yet. Twenty lag columns wouldn't clearly help either — the
design box back in Build 1 showed why: each row's `target` is literally
the next row's `lag0`, so consecutive lag columns are near-copies of the
same underlying series by construction, not twenty independent signals.

**Q4 — three reasons for a bad miss.** Build 5's three buckets: a missing
feature, a regime change, or noise this series already carries. Your worst
miss landed in the first one.

---

## ✅ Done when

- `make_features`, `compute_gradients`, and `gradient_descent` are all
  implemented and imported into `02c_gas_price_ml/explore.py`
- `test_gradient_matches_hand_worked_example` and
  `test_gradient_descent_recovers_known_coefficients` both pass
- You can state the model's validation MSE and the baseline's, and which
  one won
- `loss_curves.png` exists, and both lines fall together without crossing
- You've named your single worst miss, its date, and which of the three
  buckets it belongs in
- `actual_vs_predicted.png` exists, with the worst miss marked
- Committed

---

## 🗣️ Tell someone

Explain to a colleague, in four sentences, why a model trained on years of
real prices can still lose to guessing "next week equals this week."

Banned words: model, baseline, loss, algorithm.

---

## 📓 Log it

Three lines: what you built, what broke, what you'd do differently. Add
the validation MSE ratio between your model and the baseline, and your
worst miss's date and bucket.

---

## 💼 They will ask you this

1. Your model lost to the naive baseline. Is that a failed project? What
   would you say if a hiring manager asked why you'd report that?
2. Why didn't your model overfit here, and what would have to change for it
   to start?

---

## 🧭 The pattern

Train, compare against a named baseline, then go look at your single worst
mistake instead of stopping at the average one. That three-step habit — not
the gasoline prices, not the linear model — is what tells you whether a
result means anything, on any dataset you point a model at.

---

## ➡️ Next up

Gradient descent has now given you an honest answer twice: it recovered a
known line in 02b, and it lost to a one-line rule here, after finding a real
geopolitical event it had no way to see coming. That's **Foundations**,
finished — a math engine built from scratch, real data cleaned and split
without cheating, and a gradient-descent machine proven on both invented
and real numbers. Everything from here builds on that, which is why the
next part is called **Learning Machines**.

Stage 03 keeps this exact dataset and these same lagged features — nothing
here gets thrown away. What changes is the question: instead of "what
price," it asks "up or down," trading a bare number for a probability
squeezed through a sigmoid, and a loss curve for a confusion matrix. Same
instinct for hunting your worst mistake. Different target, different shape.
