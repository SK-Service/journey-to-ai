---
title: Get Real Data and Look at It
stage: "02a"
archetype: data
minutes: 95
new_concepts: [leakage]
new_tools: [pandas]
---

# Stage 02a — Get Real Data and Look at It

## 🔁 Last time

You built matrix multiplication from scratch and proved it against NumPy on
numbers you generated yourself. Every one of those numbers was clean, and you
chose them. Real data is neither. This stage is where you find out what that
actually means.

## 🎯 What you're making

A cleaned dataset of real weekly gasoline prices, split honestly into training
and validation sets, and a chart you would be willing to show someone. About 95
minutes.

No machine learning here. Not one model. This is the part of the job that
most tutorials skip — and skipping it is how people end up with results that
look wonderful and mean nothing.

---

## 🤔 Before you open your editor

Paper again, and this time the questions are about the data rather than the
algorithm.

1. You want to predict next week's petrol price. You have a spreadsheet with
   one row per week going back twenty years. Which rows are you allowed to
   learn from, and which are you not?
2. Suppose you shuffle all the rows randomly and take 80% for training. What
   have you accidentally let your model see?
3. A week is missing from the middle of the file. Name three different things
   you could do about it, and one situation where each would be the wrong
   choice.
4. Before any model exists, what is the simplest possible prediction for next
   week's price? Write it down. You will need it later, and it is a tougher
   opponent than it looks.

Question 2 has a name — **leakage** — letting information your model would
not actually have at prediction time slip into training anyway. It is the
single most common way a result turns out to be worthless.

---

## 🔨 Build 1 — Get the data

```powershell
cd C:\dev\bonsai
.venv\Scripts\Activate.ps1
pip install pandas
pip freeze > requirements.txt
mkdir 02a_data\raw
```

That `pip freeze` line matters again here, not just the first time in Stage
00: it rewrites `requirements.txt` with everything currently installed, so
running it again after adding `pandas` folds the new dependency into the
same file — you never edit that file by hand.

**pandas** is this stage's one new tool. It handles tables: rows, columns, dates,
missing values. NumPy holds numbers; pandas holds *labelled* numbers, which is
what real data always is. Without labels, a plain NumPy array of this same
data is just numbers by position — `data[0, 1]` might be a price or a date,
and you'd have to remember which column means what by counting. A pandas
DataFrame replaces position with a name: `df["price"]` and `df["date"]` say
exactly what they hold, so code that reads this file six months from now
still makes sense without you memorising a column order.

The US Energy Information Administration publishes weekly retail gasoline prices
going back to 1990, free and without a login. Find their download page, get the
weekly all-grades all-formulations series as a CSV, and put it in
`02a_data/raw/`.

!!! note "Take the raw file and do not touch it"
    Never edit a downloaded file by hand. Every change you make to the data
    happens in code, so that six months from now you can see exactly what you
    did and why. `raw/` is read-only by convention.

In `02a_data/load.py`:

```python
from pathlib import Path
import pandas as pd

RAW = Path(__file__).parent / "raw"


def load_prices(filename):
    """Return a DataFrame with columns: date, price. Sorted, oldest first."""
    raise NotImplementedError
```

??? info "🐍 Python syntax — `Path(...) / ...`, `__file__`"
    `pathlib.Path` overloads the `/` operator to join path segments, so
    `Path(__file__).parent / "raw"` means "join," not divide — the result is
    a path, not a number. `__file__` is a variable Python fills in for you
    automatically: the path of the file currently running. `.parent` is the
    folder that contains it. Together, `Path(__file__).parent / "raw"` means
    "the `raw` folder next to this script," wherever this project ends up on
    disk.

`filename` is the CSV you downloaded above, sitting inside `RAW`. The
docstring is the contract this function has to satisfy: return a
DataFrame with exactly two columns, `date` and `price`, sorted oldest
first — whatever the raw file's own column names and row order turn out
to be.

Write the body. `pd.read_csv` will get you started; you will need to skip some
header rows, rename the columns, parse the date column with
`pd.to_datetime`, and sort. Getting this to work *is* the Build — real files
are never shaped the way you want them.

??? tip "Hint — open only when stuck"
    `pd.read_csv(RAW / filename, skiprows=4, names=["date", "price"])`
    skips the file's header rows and names the two columns yourself,
    rather than trusting whatever the raw file's own header row says.
    `df["date"] = pd.to_datetime(df["date"])` turns the date column from
    text into a real date, which every later Build depends on. Return
    `df.sort_values("date")`, sorted oldest first. (What to do about
    missing or duplicated rows is Build 2's decision, not this one —
    leave that for when you've actually seen what's wrong.)

Create `02a_data/explore.py` and call it there:

```python
from load import load_prices

df = load_prices("your_downloaded_filename.csv")

print(df.head())
print(df.tail())
print(df.shape)
print(df.dtypes)
```

??? info "🔧 What this code does — first look at a DataFrame"
    `.head()` and `.tail()` return the first and last 5 rows, so you can spot
    both a broken header and a broken final row in one glance. `.shape` is
    `(rows, columns)` — no `()` after it, because it's a stored property, not
    something being calculated. `.dtypes` lists the data type pandas picked
    for each column; a date column that came back as plain text instead of a
    real date is the most common surprise here, and the next Build depends on
    it being a real one.

Run it:

```powershell
cd 02a_data
python explore.py
```

Read all four before going on. Every Build below keeps adding to this same
file, so `df` stays loaded and defined for the rest of the stage.

---

## 🔨 Build 2 — Find out what is wrong with it

Every real dataset has something wrong with it. Find yours.

Add to `02a_data/explore.py`:

```python
print("rows:          ", len(df))
print("date range:    ", df["date"].min(), "to", df["date"].max())
print("missing prices:", df["price"].isna().sum())
print("duplicated dates:", df["date"].duplicated().sum())

gaps = df["date"].diff().value_counts()
print("\nspacing between rows:")
print(gaps.head())
```

??? info "🐍 Python syntax — chaining `.method().method()`"
    `df["price"].isna().sum()` runs three steps left to right, each on the
    result of the one before: select the `price` column, flag which entries
    are missing, then count how many flags are true. This chaining pattern —
    call, then call again on what came back — is not specific to pandas; it
    works on any object with methods, and you'll see it again throughout this
    journey.

??? info "🔧 What this code does — reading the diagnostic prints"
    `df["date"].min()`/`.max()` are the earliest and latest dates in the
    file. `.isna().sum()` counts missing prices; `.duplicated().sum()` counts
    dates that appear more than once. `df["date"].diff()` subtracts each date
    from the one before it, giving you a gap in days for every row;
    `.value_counts()` then tallies how often each gap size occurs, so
    `gaps.head()` shows you the most common gaps — which should be
    overwhelmingly "7 days" if the data is clean.

Run it: `python explore.py`.

That last block is the important one. You expect every gap to be exactly seven
days. Anything else means a week is missing, duplicated, or recorded on the
wrong day.

Decide what to do about each problem you find, and **write the reason in a
comment next to the code that does it**. In three months you will not remember
why you dropped those four rows, and the comment is the difference between a
defensible decision and a mystery.

??? tip "Hint — open only when stuck"
    `df.isna().sum()` counts missing values per column. `df.dropna()` removes
    rows, `df.ffill()` carries the previous value forward. For a price series
    that moves slowly, carrying forward is usually defensible — but say so in
    the comment, because it is a choice and not a fact.

---

## 🔨 Build 3 — Split it without cheating

Here is the rule, and it is the whole reason this Build exists:

!!! warning "Never let the model see the future"
    Your validation set must come entirely **after** your training set in time.
    A random 80/20 shuffle scatters rows from every year into both sets, so
    some training rows would sit *after* the validation weeks they are meant
    to help predict. The model would be tuned using information from the
    future relative to what it is scored on — and it would score brilliantly,
    and the score would be a lie.

Add to `02a_data/load.py`, next to `load_prices`:

```python
def time_split(df, train_fraction=0.8):
    """Split chronologically. Everything in train precedes everything in val."""
    raise NotImplementedError
```

??? info "🐍 Python syntax — `train_fraction=0.8`"
    `def time_split(df, train_fraction=0.8):` gives the second parameter a
    **default value**. Call it as `time_split(df)` and `train_fraction` is
    `0.8`; call it as `time_split(df, 0.9)` and it isn't. Defaults let a
    caller override only what they need to, and any parameter with one must
    come after every parameter without one.

??? tip "Hint — open only when stuck"
    Sort by date first if you haven't already — `df.sort_values("date")`.
    Then `cut = int(len(df) * train_fraction)` gives you a row count, and
    `df.iloc[:cut]` / `df.iloc[cut:]` split by position: the first `cut`
    rows, then everything after. `.iloc` is pandas' way of saying "by
    position," as opposed to selecting by a column's actual values. Return
    both as a pair: `return df.iloc[:cut], df.iloc[cut:]`.

Write it, then call it in `02a_data/explore.py` (update the import to
`from load import load_prices, time_split`):

```python
train, val = time_split(df)
print(len(train), len(val))
print(train["date"].max(), val["date"].min())
```

??? info "🔧 What this code does — checking the split"
    `train, val = time_split(df)` unpacks the two DataFrames your function
    returns into two names at once — the same pattern as the `for name,
    thing in [...]` unpacking from Stage 01, just returning two values
    instead of looping over a list of them. `len(train)`/`len(val)` should
    add up to `len(df)`. `train["date"].max()` and `val["date"].min()`
    print the boundary date on each side of the split — the first should
    land before the second, since `train` is everything *before* `val` in
    time.

Now prove it — as its own file. It won't see what `explore.py` loaded, so it
needs its own `df`. Create `02a_data/test_load.py`:

```python
from load import load_prices, time_split

df = load_prices("your_downloaded_filename.csv")


def test_split_is_chronological():
    train, val = time_split(df)
    assert train["date"].max() < val["date"].min()
    assert len(train) + len(val) == len(df)
```

??? info "🐍 Python syntax — reading `test_split_is_chronological`"
    The two lines above the test load `df` fresh, the same way `explore.py`
    did — a separate file, a separate run, nothing carries over between them.
    `def test_split_is_chronological():` takes no parameters — the empty
    `()` is legal, it just means this function needs nothing handed to it.

    Each `assert` line checks one condition and fails loudly, naming what
    broke, if that condition is false. Watch the second one specifically:
    `==` *compares* two values for equality and hands back `True` or
    `False`; a single `=` *assigns* a value instead. They look one character
    apart and do unrelated things — this is the one line on the page where
    mixing them up would silently test the wrong thing.

Run it — re-anchor to the repo root first, in case you're picking this back
up in a fresh terminal or still sitting in `02a_data` from Build 1:

```powershell
cd C:\dev\bonsai
cd 02a_data
pytest
```

That assertion is not a formality. It is the one line standing between you and
a result that fools you, and versions of this exact mistake reach production at
real companies more often than anyone likes to admit.

---

## 🔨 Build 4 — Draw it

Add to `02a_data/explore.py`:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train["date"], train["price"], label="train", linewidth=0.9)
ax.plot(val["date"], val["price"], label="validation", linewidth=0.9)
ax.set_xlabel("year")
ax.set_ylabel("price (USD/gallon)")
ax.set_title("US weekly retail gasoline price")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("price_history.png", dpi=150, bbox_inches="tight")
plt.show()
```

??? info "🔧 What this code does — fig, ax, and saving to a file"
    `plt.subplots(figsize=(11, 4))` returns two things at once — `fig` (the
    whole canvas) and `ax` (the one plot area on it) — unpacked together the
    same way `train, val = time_split(df)` unpacked two values in Build 3.
    Everything after is a method on `ax`: `ax.plot(...)` draws one line,
    called twice here for train and validation; `ax.set_xlabel`/`set_ylabel`/
    `set_title` label the axes and title; `ax.legend()` draws the box naming
    each line from its `label=`; `ax.grid(alpha=0.3)` draws faint gridlines.
    `fig.savefig(...)` writes the whole canvas to a file, in whatever folder
    you're running from — `dpi=150` sets how sharp the image is,
    `bbox_inches="tight"` trims empty border. `plt.show()` still opens the
    on-screen window separately; the file is saved either way.

Run it: `python explore.py`.

Look at it properly before moving on. Find 2008 — the financial crisis. Find
2020 — the pandemic collapse. Find 2022 — the spike after Russia's invasion of
Ukraine. You are looking at recognisable history in a column of numbers, and
that is worth thirty seconds of attention.

Then look at the right-hand edge of the chart: the weeks closest to today,
whatever today is when you are reading this. That is where your validation
set lives, and it is the region any prediction you build has to actually work
in — not 2008, not 2022, but now.

Then answer one question from the picture: **week to week, how much does this
line actually move?** Not year to year — week to week. Zoom into any twelve
months and look. Add to `02a_data/explore.py`:

```python
weekly_change = df["price"].diff()
print(weekly_change.describe())
```

Run it: `python explore.py`.

!!! note "Read this before you build any model"
    That number is small, and it is why prediction here is hard. Next week's
    price is very close to this week's price, and "tomorrow equals today" is a
    genuinely strong baseline — hard enough to beat that professional
    forecasters often do not.

    So when your model arrives in Stage 02c and loses to it, that is not you
    failing. That is the finding, and being able to demonstrate it honestly is
    worth more in an interview than a model that appears to win because
    something leaked.

Commit your work, `raw/` included — safe to do, since you never hand-edited it:

```powershell
cd C:\dev\bonsai
git add .
git commit -m "Stage 02a: clean and split gasoline price data"
```

---

## 🔮 Check your prediction

Go back to the four questions from before you opened your editor.

**Q1 — which rows you can learn from.** Only the ones before the date you are
predicting. Build 3's `test_split_is_chronological` is that answer, enforced
as code rather than left as an intention.

**Q2 — what a random shuffle would let the model see.** The future,
concretely: shuffle the rows and take a random 80% for training, and some
of those training rows land on dates *after* some of your validation
weeks. The model gets tuned partly on what prices did in weeks that come
later than the ones it's scored against — exactly backwards from the real
task of predicting a week you haven't seen yet. Build 3's whole reason for
existing is to make that impossible by construction, not by discipline you
have to remember to apply every time.

**Q3 — the missing week.** You picked one of drop, forward-fill, or
interpolate in Build 2, and wrote down why. There is no single correct choice
here — the wrong answer is not picking one, it is picking one silently. Check
that your comment still explains it.

**Q4 — the simplest possible prediction.** "Next week equals this week." Look
at the `weekly_change` you measured a moment ago — if it is small, that baseline is
strong, and you now know why by number rather than by guessing. Whatever you
wrote before opening your editor, compare it now. Most first guesses here are
more complicated than "last week's value," and simpler than that is hard to
beat.

---

## 🔎 Go find out

**Question:** Where does this number come from? Somebody has to measure the
price of petrol every week. Who, and how?

**Search phrases:**

- `EIA weekly retail gasoline price survey methodology`
- `how is average gas price calculated`
- `site:eia.gov gasoline price survey`

**Follow-ups to answer in your log:**

- How many filling stations are surveyed, and how are they chosen?
- Which day of the week is the price recorded, and why does that matter for a
  model that predicts "next week"?
- Is the published figure an average of prices, or an average weighted by how
  much fuel each station sells? What difference would that make?
- The series is revised occasionally. If a past value changes after you have
  trained on it, what breaks?

**Time-box:** 25 minutes.

**Verification:** you should end with a specific day of the week. If you have
one, you found the methodology page. That single fact will constrain how you
frame the prediction problem in Stage 02c.

---

## ✅ Done when

- `load_prices` returns a sorted DataFrame with parsed dates
- You can state how many rows were missing or duplicated, and what you did
- `test_split_is_chronological` passes
- `price_history.png` exists and you would show it to someone
- The typical weekly change is written in your log
- Committed, with `raw/` included so the work is reproducible

---

## 🗣️ Tell someone

Explain to a friend, in **four sentences**, why splitting data randomly would
be cheating here — using an example from school or exams, not from finance.

Banned words: leakage, training, validation, model, data.

---

## 📓 Log it

Three lines: what you built, what broke, what you would do differently. Add the
typical weekly change and your baseline from question 4.

---

## 💼 They will ask you this

1. How did you split your data, and why that way?
2. Your model beats the baseline on validation. Name three things that could be
   true other than "the model is good".

---

## 🧭 The pattern

Load, inspect, clean, split. That is not a gasoline-prices routine — it is the
skeleton underneath almost any real tabular-data problem you will meet, in
this journey or in a job. The dataset changes every time. This sequence does
not.

---

## ➡️ Next up

You have honest data and a baseline to beat. What you do not have is any way to
make a prediction at all.

Next you write gradient descent — the algorithm underneath essentially
every model you build from here on. You will run it on data you generate
yourself, from an equation you choose, so that you know the right answer in
advance and can tell whether your code found it.

Real data comes back the stage after. First you need to be certain the machine
works.
