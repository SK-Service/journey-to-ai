---
title: Set Up Your Workshop
stage: "00"
archetype: setup
minutes: 100
new_concepts: [virtual environment]
new_tools: [Python, pip, git, VS Code]
---

# Stage 00 — Set Up Your Workshop

## 🎯 What you're making

A working Python workshop on your laptop, ending with a plot on your screen, a
passing test, and your first commit pushed to GitHub. About 100 minutes.

Nothing here is mathematics. This session is about making the tools work, so that
each session after this one is about ideas instead of error messages.

---

## 🔨 Build 1 — Install Python and prove it

Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/)
and download the **second-most-recent stable release**, not the newest one on
the page — for example, if 3.14 is newest, get 3.13.

!!! warning "Take the older version on purpose"
    Libraries you will need later — PyTorch in particular — occasionally lag
    a few months behind a brand-new Python release. Installing the newest
    Python can get you a `No matching distribution found` error in Stage 11
    that looks like you broke something, when you only picked a number that
    is too new. This is exactly the same reasoning you will use for the
    PyTorch install itself in Stage 11 — check what is current, then pick a
    step behind it, rather than trusting a fixed number in a document that
    does not know today's date.

In the installer, **tick "Add python.exe to PATH"** before clicking Install.
It is a small checkbox at the bottom and it is the single most common reason
setup goes wrong.

Open PowerShell (Start menu, type `powershell`) and run:

```powershell
python --version
```

You should see `Python 3.x.something` — the version you installed.

??? tip "You saw nothing, or the Microsoft Store opened"
    Windows ships a placeholder that hijacks the word `python`. Go to
    Settings → Apps → Advanced app settings → App execution aliases, and turn
    **off** the entries for `python.exe` and `python3.exe`. Close PowerShell,
    open it again, and retry.

??? tip "'python' is not recognized"
    PATH did not get set. Re-run the installer, choose Modify, and make sure
    the PATH option is ticked. Then close and reopen PowerShell — an open
    window does not pick up a new PATH.

---

## 🔨 Build 2 — Make a home for the project

Two rules about where this folder lives.

**Keep it out of OneDrive.** OneDrive syncs every file change to the cloud.
Later you will write training checkpoints that change every few seconds, and
sync churn will corrupt runs and eat your disk quota.

**Keep it out of your other work folders.** This is a side project. It should
never be one wrong command away from your goal.

```powershell
mkdir C:\dev\bonsai
cd C:\dev\bonsai
```

??? info "💻 Command line — `mkdir`, `cd`"
    `mkdir C:\dev\bonsai` creates a folder at that path. `cd C:\dev\bonsai`
    moves your terminal's "current folder" into it, so every command you run
    after this — installing libraries, creating files, `git` commands —
    happens there instead of wherever the terminal happened to open.

Now create a **virtual environment** — a private copy of Python that belongs to
this project alone. Install a library into it and nothing else on your machine
changes.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

??? info "💻 Command line — `python -m venv`, `Activate.ps1`"
    `-m venv` tells Python to run its built-in `venv` module, which creates
    the `.venv` folder — a private copy of the Python interpreter, empty of
    libraries until you `pip install` into it. Nothing is copied out of your
    main Python install.

    `Activate.ps1` doesn't install anything. It's a script that edits *this
    terminal window's* `PATH` so `python` and `pip` point at `.venv`'s copies
    instead of your system-wide one — which is what makes `(.venv)` appear in
    your prompt. Close the terminal and that edit is gone; run the activate
    line again next time you open a new one.

Your prompt should now start with `(.venv)`.

??? tip "'running scripts is disabled on this system'"
    PowerShell blocks scripts by default. Allow signed local scripts for your
    own user account only:

    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

    Answer `Y`, then run the activate line again.

With `(.venv)` showing, install what you need:

```powershell
pip install numpy matplotlib pytest
pip freeze > requirements.txt
```

That last line records exactly what you installed. It is how you — or an
interviewer — rebuild this environment on another machine a year from now.

!!! note "Every session starts the same way"
    `cd C:\dev\bonsai` then `.venv\Scripts\Activate.ps1`. If a command fails
    with `ModuleNotFoundError`, check for `(.venv)` in your prompt first. It is
    the answer about half the time.

---

## 🔨 Build 3 — Put something on the screen

Install [VS Code](https://code.visualstudio.com/), open it, and use
**File → Open Folder** on `C:\dev\bonsai`. Install the Microsoft **Python**
extension when it offers.

You'll do two things in VS Code from here on: write Python files, and run
them. To create one, look at the **Explorer** panel on the left — hover
over the folder name and click the "New File" icon (or right-click it) —
and name the file with a `.py` extension. To run one, use VS Code's
**built-in terminal**, not a separate PowerShell window; you'll open it the
same way every time from here on.

Make a file called `hello_vector.py`:

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([3.0, 4.0])

print("vector:", v)
print("shape: ", v.shape)
print("length:", np.linalg.norm(v))

plt.figure(figsize=(4, 4))
plt.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1)
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid(True)
plt.title("my first vector")
plt.show()
```

??? info "🐍 Python syntax — `import numpy as np`, `v[0]`"
    `import` pulls a library's code into your file so you can use it.
    `import numpy as np` also gives it a short local name — `as np` — so the
    rest of the file writes `np.array(...)` instead of spelling out
    `numpy.array(...)` every time. The name after `as` is your choice, but
    `np` and `plt` are conventions strong enough that changing them would
    confuse anyone else reading the code.

    `v[0]` and `v[1]` pull out the first and second entries of `v`. Square
    brackets index into a sequence by position, and Python counts from `0` —
    so `v[0]` is the first entry, not the one "before" it. You'll write this
    constantly from here on.

??? info "🔧 What this code does — plotting the arrow"
    `plt.figure(figsize=(4, 4))` opens a blank 4x4-inch canvas. `plt.quiver(0,
    0, v[0], v[1], angles="xy", scale_units="xy", scale=1)` draws one arrow
    from the origin `(0, 0)` to the point `(v[0], v[1])` — `quiver` is
    matplotlib's name for "draw arrows," and those three keyword arguments
    tell it to draw a plain xy arrow rather than the scaled vector-field
    arrows it draws by default. `plt.xlim(-1, 5)`/`plt.ylim(-1, 5)` fix the
    visible axis range so the arrow doesn't float off-center; `plt.grid(True)`
    and `plt.title(...)` add gridlines and a title. Nothing appears on screen
    until `plt.show()` — every call before it only describes the picture.

That arrow runs 3 across and 4 up from the origin — a right triangle, with
the arrow itself as the hypotenuse. Before you run this, use the
Pythagorean theorem to work out what `length:` should print.

Open a terminal — **Terminal → New Terminal** in the top menu bar — and run
it there:

```powershell
python hello_vector.py
```

A window opens with an arrow in it, and the terminal prints a length of
`5.0` — matching what you worked out by hand above. That is the point — the
machine agrees with you, so the machine is working.

Close the plot window to end the program.

---

## 🔨 Build 4 — Write a test that fails, then make it pass

For the next six months, one question will keep coming back: *how do I know
this is right?* The answer is almost always a test — a small piece of code that
checks another piece of code and complains when it is wrong.

Create `test_setup.py`:

```python
import numpy as np


def vector_length(v):
    """Return the Euclidean length of v."""
    raise NotImplementedError


def test_vector_length():
    assert np.isclose(vector_length(np.array([3.0, 4.0])), 5.0)
    assert np.isclose(vector_length(np.array([0.0, 0.0])), 0.0)
    assert np.isclose(vector_length(np.array([1.0, 1.0])), np.sqrt(2))
```

??? info "🐍 Python syntax — `def`, `raise`, and `assert`"
    `def vector_length(v):` defines a function: a name, its parameters in
    `()`, a colon, then an indented body. The line right after `def` in
    quotes — `"""Return the Euclidean length of v."""` — is a docstring,
    a description of what the function does. `raise NotImplementedError` is
    a placeholder: it deliberately crashes with a clear message until you
    replace it with a real body that computes and returns a value.

    The second function, `def test_vector_length():`, is not called
    anywhere on this page — `pytest` finds every function named `test_...`
    automatically and runs it.

    `assert` is Python's built-in "this had better be true" check.
    `assert condition` does nothing at all if `condition` is `True`; if
    it's `False`, Python immediately raises an `AssertionError` and stops
    right there, naming the exact line that failed. A few examples outside
    any test:
    ```python
    assert 2 + 2 == 4                  # True - nothing happens
    assert len([1, 2, 3]) == 3         # True - nothing happens
    assert 1 == 2, "one is not two"    # False - raises: AssertionError: one is not two
    ```
    The optional text after a comma is a message shown only if the check
    fails, so you can say what went wrong in your own words instead of just
    seeing the failing expression. A test function is a normal function
    whose body is a string of these checks — `pytest` runs every
    `test_...` function it finds and reports every `assert` that failed,
    across every test, in one pass.

Run it:

```powershell
pytest
```

It fails. That is correct — you have not written `vector_length` yet.

Now replace the `raise NotImplementedError` line with a body that computes the
length, and run `pytest` again until it passes. You know the formula.

??? tip "Stuck on the syntax rather than the maths"
    A function returns a value with `return`. `np.sqrt` takes a square root,
    and `np.sum` adds up an array. Squaring is `**2`, and it applies to every
    element of a NumPy array at once.

Red, then green. That loop is how you will verify every stage from here on.

---

## 🔨 Build 5 — Put it under version control

```powershell
git --version
```

If that fails, install [Git for Windows](https://git-scm.com/download/win),
accepting the defaults, then reopen PowerShell.

Left alone, git tracks every file in the folder — including `.venv`, which
is large, machine-specific, and rebuildable, not something worth saving a
history of. A `.gitignore` file tells git which paths to leave alone, before
it ever gets the chance to track them.

Create a file called `.gitignore` containing these three lines:

```text
.venv/
__pycache__/
*.pyc
```

That keeps your virtual environment out of the repository — it is large,
machine-specific, and rebuildable from `requirements.txt`.

```powershell
git init -b main
git add .
git commit -m "Stage 0: workshop set up"
```

??? info "💻 Command line — `git init -b`, `git add .`, `git commit -m`"
    `git init` creates a new, empty repository in the current folder; `-b
    main` names its first branch `main` (git's older default was `master`).
    `git add .` stages every file in the current folder — `.` meaning "here"
    — for the next commit; "staged" means "marked as ready to save," not
    saved yet. `git commit -m "..."` saves everything staged as one point in
    history; `-m` supplies the message directly instead of opening an editor
    for it.

Now make it real. On [github.com](https://github.com), create a new **public**
repository. Do not add a README — you already have files. Then run the two
commands GitHub shows you under *push an existing repository*.

That commit is the first entry in a history you will still be adding to in six
months. The history itself becomes part of what you show people at the end.

---

## 🔎 Go find out

**Question:** Why does NumPy exist? Python already has lists, and you can
already do arithmetic on them.

**Search phrases:**

- `numpy vs python list performance`
- `why is numpy faster than pure python`
- `site:numpy.org what is numpy`

**Follow-ups to answer in your log:**

- What does *vectorised* mean, and what is it contrasted with?
- What is the language underneath NumPy actually written in?
- Why must every element of a NumPy array have the same type, when a Python
  list can hold anything?
- NumPy runs on your CPU. What kind of hardware is the alternative, and what
  is that library called?

**Time-box:** 25 minutes. Then stop, whether or not you feel finished.

**Verification:** write your answer in three sentences a friend would follow,
then read the front page of numpy.org and see where you were wrong. Being wrong
here costs nothing and the correction sticks.

---

## 🤖 Second opinion

Paste your three sentences into Claude or ChatGPT with this:

> I am learning this by building it from scratch. Here is my understanding of
> why NumPy exists and why plain Python lists are not enough — don't tell me
> the answer, poke holes in it. What did I get wrong, what did I leave out, and
> what question should I be asking that I am not?

Then check what it tells you against numpy.org. Note anything it got wrong in
your log. You are building a language model this year; learning when to trust
one is part of the job.

---

## ✅ Done when

- `python --version` prints the version you installed
- Your prompt shows `(.venv)` after activating
- `python hello_vector.py` draws an arrow and prints `5.0`
- `pytest` reports one passing test
- `requirements.txt` and `.gitignore` exist
- Your repository is on GitHub with one commit

---

## 🗣️ Tell someone

Explain to a sibling, in **four sentences**, what a virtual environment is
and why you would want one.

Banned words: environment, dependency, package, isolate. Find your own.

---

## 📓 Log it

Make a file called `LOG.md` and write three lines: what you built, what broke,
and what you would do differently. Commit it.

You will write three lines every session. Over time it becomes the most
convincing thing in your portfolio, because it is the one part nobody can fake.

---

## 🧭 The pattern

A private, disposable copy of a tool, rebuilt from a recorded list of exactly
what it needs — that is not a Python habit. It is how you will set up a new
laptop, a CI server, or a teammate's machine for the rest of your career.
`requirements.txt` is the recorded list; the venv is the disposable copy.

---

## ➡️ Next up

You have a workshop and one arrow on a screen. What you do not have is any way
to *do* anything with that arrow.

NumPy will multiply two matrices for you in a single call. Next session you
will write that multiplication yourself, by hand, and then check your answer
against NumPy's. You know the mathematics already. The interesting part is
finding out what the machine has to do to carry it out — and that turns out to
matter enormously once the matrices get big.
