---
title: Foundations Recap
---

# One Operation, Five Lenses

<style>
.foundations-figures svg{display:block; width:100%; height:auto;}
.foundations-figures svg rect{fill:none; stroke:currentColor; stroke-width:1.1;}
.foundations-figures svg rect.acc-soft{fill:var(--md-accent-fg-color); fill-opacity:0.18; stroke:currentColor;}
.foundations-figures svg line{stroke:currentColor; stroke-width:1.1;}
.foundations-figures svg line.acc-stroke{stroke:var(--md-accent-fg-color); stroke-width:1.3;}
.foundations-figures svg path{fill:none; stroke:currentColor; stroke-width:1.1;}
.foundations-figures svg path.acc-stroke{stroke:var(--md-accent-fg-color); fill:none; stroke-width:1.3;}
.foundations-figures svg path.link{stroke:currentColor; stroke-width:1; fill:none; opacity:0.55; stroke-dasharray:2.5 2.5;}
.foundations-figures svg circle{fill:currentColor;}
.foundations-figures svg circle.acc-fill{fill:var(--md-accent-fg-color);}
.foundations-figures svg text{fill:currentColor; font-family:var(--md-code-font, "Roboto Mono", monospace);}
.foundations-figures svg text.acc-fill{fill:var(--md-accent-fg-color);}
.foundations-figures svg .arrow-fill{fill:currentColor;}
.foundations-figures svg .arrow-fill.acc{fill:var(--md-accent-fg-color);}
.foundations-figures figure{margin:0.5em 0;}
.foundations-figures figcaption{font-size:0.8em; opacity:0.75; margin-top:0.4em;}
.foundations-figures .pipeline-scroll{overflow-x:auto;}
.foundations-figures .pipeline-scroll svg{min-width:640px;}
</style>

<div class="foundations-figures" markdown>

You've now looked at prediction from five different angles — starting with
matrix multiplication written by hand in Python, and ending with a model
judged against real gasoline prices, with no answer key to check yourself
against. This is the foundation the rest of the journey stands on. Stage 03
picks it up next, with a new question for it to answer.

## The thread

You started by setting up a Python environment from scratch, then moved
into the mathematics of machine learning with matrix multiplication, and
from there into the calculus behind gradient descent. Along the way, you
also learned to plot and look at data before trusting it.

<div class="grid cards" markdown>

-   **Stage 00 &middot; Set Up Your Workshop**

    <svg viewBox="0 0 240 170" role="img" aria-label="Python, git, and a virtual environment converge into one repository, which produces one plotted arrow.">
      <rect x="12" y="14" width="60" height="24"></rect>
      <text x="42" y="30" text-anchor="middle" font-size="9">python</text>
      <rect x="90" y="14" width="60" height="24"></rect>
      <text x="120" y="30" text-anchor="middle" font-size="9">git</text>
      <rect x="168" y="14" width="60" height="24"></rect>
      <text x="198" y="30" text-anchor="middle" font-size="9">.venv</text>
      <line x1="42" y1="38" x2="42" y2="56"></line>
      <line x1="120" y1="38" x2="120" y2="56"></line>
      <line x1="198" y1="38" x2="198" y2="56"></line>
      <line x1="42" y1="56" x2="198" y2="56"></line>
      <line x1="120" y1="56" x2="120" y2="66"></line>
      <rect x="88" y="66" width="64" height="26"></rect>
      <text x="120" y="83" text-anchor="middle" font-size="8.5">repo ready</text>
      <line x1="120" y1="92" x2="120" y2="110"></line>
      <line x1="80" y1="150" x2="160" y2="150"></line>
      <line x1="80" y1="112" x2="80" y2="150"></line>
      <path class="acc-stroke" d="M80,150 L138,122" marker-end="url(#fr-a00)"></path>
      <defs>
        <marker id="fr-a00" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
        </marker>
      </defs>
    </svg>

    *python, git and a virtual environment converge into one repo — the
    first arrow on screen, computed by a library, not by you.*

    You assembled an environment you can reproduce from scratch and watched
    one arrow plot. None of that arithmetic was yours yet.

    &mdash; environment, not equation yet &mdash;

-   **Stage 01 &middot; Build Your Own Mini Math Engine**

    <svg viewBox="0 0 240 170" role="img" aria-label="Matrix A's first row and matrix B's first column meet in matrix C's top-left cell, one dot product per cell.">
      <text x="100" y="82" text-anchor="middle" font-size="8" opacity="0.65">A</text>
      <rect x="70" y="90" width="30" height="30" class="acc-soft"></rect>
      <rect x="100" y="90" width="30" height="30"></rect>
      <rect x="70" y="120" width="30" height="30"></rect>
      <rect x="100" y="120" width="30" height="30"></rect>
      <text x="126" y="6" text-anchor="middle" font-size="8" opacity="0.65">B</text>
      <rect x="140" y="10" width="30" height="30" class="acc-soft"></rect>
      <rect x="170" y="10" width="30" height="30"></rect>
      <rect x="140" y="40" width="30" height="30"></rect>
      <rect x="170" y="40" width="30" height="30"></rect>
      <text x="135" y="108" text-anchor="middle" font-size="13">&times;</text>
      <rect x="140" y="90" width="30" height="30" stroke="currentColor" stroke-width="1.6"></rect>
      <text x="155" y="110" text-anchor="middle" font-size="12" class="acc-fill">&Sigma;</text>
      <rect x="170" y="90" width="30" height="30"></rect>
      <rect x="140" y="120" width="30" height="30"></rect>
      <rect x="170" y="120" width="30" height="30"></rect>
      <text x="165" y="163" text-anchor="middle" font-size="8" opacity="0.65">C</text>
      <path class="link" d="M70,105 L170,105"></path>
      <path class="link" d="M155,10 L155,150"></path>
    </svg>

    *A's top row and B's left column sweep into one cell of C — a dot
    product, repeated for every cell.*

    You wrote dot product and matrix multiplication yourself, checked
    against NumPy line by line. Stage 09 imports this file unmodified.

    $$c = \sum_i a_i b_i$$

-   **Stage 02a &middot; Get Real Data and Look at It**

    <svg viewBox="0 0 240 170" role="img" aria-label="A table of weekly prices becomes a scatter plot of price against time.">
      <text x="34" y="12" text-anchor="middle" font-size="8" opacity="0.65">week</text>
      <text x="77" y="12" text-anchor="middle" font-size="8" opacity="0.65">price</text>
      <rect x="14" y="18" width="84" height="98"></rect>
      <line x1="56" y1="18" x2="56" y2="116"></line>
      <line x1="14" y1="42" x2="98" y2="42"></line>
      <line x1="14" y1="66" x2="98" y2="66"></line>
      <line x1="14" y1="90" x2="98" y2="90"></line>
      <rect x="58" y="18" width="38" height="98" class="acc-soft"></rect>
      <path class="acc-stroke" d="M104,67 L124,67" marker-end="url(#fr-a02a)"></path>
      <defs>
        <marker id="fr-a02a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
        </marker>
      </defs>
      <line x1="140" y1="20" x2="140" y2="140"></line>
      <line x1="140" y1="140" x2="228" y2="140"></line>
      <text x="184" y="152" text-anchor="middle" font-size="8" opacity="0.65">week</text>
      <circle cx="150" cy="132" r="2.3"></circle>
      <circle cx="158" cy="120" r="2.3"></circle>
      <circle cx="166" cy="124" r="2.3"></circle>
      <circle cx="174" cy="108" r="2.3"></circle>
      <circle cx="182" cy="112" r="2.3"></circle>
      <circle cx="190" cy="95" r="2.3"></circle>
      <circle cx="198" cy="90" r="2.3"></circle>
      <circle cx="206" cy="78" r="2.3"></circle>
      <circle cx="214" cy="60" r="2.3"></circle>
      <circle cx="222" cy="50" r="2.3"></circle>
    </svg>

    *The price column becomes points you can actually judge — trend, noise
    and gaps, all visible at once.*

    You pulled real weekly gasoline prices into a table and looked before
    trusting — for gaps, for scale, for whether the story matches the
    numbers.

    $$y \text{ — observed}$$

-   **Stage 02b &middot; Teach a Line to Fit Itself**

    <svg viewBox="0 0 240 170" role="img" aria-label="A loss curve with a series of steps descending toward the minimum, each step shrinking.">
      <line x1="30" y1="20" x2="30" y2="150"></line>
      <line x1="30" y1="150" x2="220" y2="150"></line>
      <text x="14" y="26" font-size="8" opacity="0.65">loss</text>
      <text x="220" y="163" text-anchor="middle" font-size="8" opacity="0.65">w</text>
      <path d="M42,42 Q88,146 130,146 Q172,146 212,42"></path>
      <circle class="acc-fill" cx="55" cy="70" r="3"></circle>
      <path class="acc-stroke" d="M55,70 L75,104" marker-end="url(#fr-a02b)"></path>
      <circle class="acc-fill" cx="75" cy="104" r="3"></circle>
      <path class="acc-stroke" d="M75,104 L95,127" marker-end="url(#fr-a02b)"></path>
      <circle class="acc-fill" cx="95" cy="127" r="3"></circle>
      <path class="acc-stroke" d="M95,127 L112,138" marker-end="url(#fr-a02b)"></path>
      <circle class="acc-fill" cx="112" cy="138" r="3"></circle>
      <path class="acc-stroke" d="M112,138 L128,144" marker-end="url(#fr-a02b)"></path>
      <circle class="acc-fill" cx="128" cy="144" r="3"></circle>
      <defs>
        <marker id="fr-a02b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
          <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
        </marker>
      </defs>
      <text x="60" y="88" font-size="8" class="acc-fill">&minus;&nabla;L</text>
      <text x="128" y="160" text-anchor="middle" font-size="8" opacity="0.65">min</text>
    </svg>

    *Predict, measure loss, step against the gradient, repeat — each step
    shorter as the loop nears the minimum.*

    You wrote the loop that finds a line without being told the answer:
    predict, measure loss, compute the gradient, step, repeat.

    $$\hat{y} = wx + b, \quad \arg\min_{w,b} L$$

-   **Stage 02c &middot; Point It at the Real World**

    <svg viewBox="0 0 260 170" role="img" aria-label="Three weeks of lagged prices multiplied by a weight vector produce a predicted price, compared against the actual price.">
      <text x="18" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t-2</text>
      <text x="42" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t-1</text>
      <text x="66" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t</text>
      <rect x="6" y="16" width="24" height="24"></rect>
      <rect x="30" y="16" width="24" height="24"></rect>
      <rect x="54" y="16" width="24" height="24"></rect>
      <rect x="6" y="40" width="24" height="24"></rect>
      <rect x="30" y="40" width="24" height="24"></rect>
      <rect x="54" y="40" width="24" height="24"></rect>
      <rect x="6" y="64" width="24" height="24"></rect>
      <rect x="30" y="64" width="24" height="24"></rect>
      <rect x="54" y="64" width="24" height="24"></rect>
      <text x="42" y="102" text-anchor="middle" font-size="8" opacity="0.65">X</text>
      <text x="94" y="112" text-anchor="middle" font-size="13">&times;</text>
      <rect x="106" y="16" width="24" height="24"></rect>
      <rect x="106" y="40" width="24" height="24"></rect>
      <rect x="106" y="64" width="24" height="24"></rect>
      <text x="118" y="102" text-anchor="middle" font-size="8" opacity="0.65">w</text>
      <text x="144" y="52" text-anchor="middle" font-size="13">=</text>
      <rect x="156" y="16" width="24" height="24" class="acc-soft"></rect>
      <rect x="156" y="40" width="24" height="24" class="acc-soft"></rect>
      <rect x="156" y="64" width="24" height="24" class="acc-soft"></rect>
      <text x="168" y="102" text-anchor="middle" font-size="8" class="acc-fill">y&#770;</text>
      <line x1="184" y1="76" x2="200" y2="76" class="acc-stroke" marker-end="url(#fr-a02c)"></line>
      <defs>
        <marker id="fr-a02c" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
          <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
        </marker>
      </defs>
      <rect x="202" y="64" width="24" height="24"></rect>
      <text x="214" y="102" text-anchor="middle" font-size="8" opacity="0.65">y</text>
      <text x="214" y="128" text-anchor="middle" font-size="7" class="acc-fill">residual</text>
    </svg>

    *Three lagged prices, one weight vector, one predicted price — measured
    against what actually happened next.*

    You pointed the loop at real prices with no answer key, and had to
    decide honestly whether it beat the simplest possible guess.

    $$\hat{y} = Xw$$

</div>

## Zooming into one stage

In Stage 02c, you ran an end-to-end machine learning pipeline in miniature
— from collecting data all the way to calculating the residual.

<div class="pipeline-scroll">
<svg viewBox="0 0 900 180" role="img" aria-label="Pipeline: raw prices become lag features, get split chronologically into train and validation, gradient descent fits on train rows only, the model predicts on validation, and the result is compared against a naive baseline with no assumed winner.">

  <defs>
    <marker id="fr-pf" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
    </marker>
  </defs>

  <line class="acc-stroke" x1="140" y1="76" x2="160" y2="76" marker-end="url(#fr-pf)"></line>
  <line class="acc-stroke" x1="290" y1="76" x2="310" y2="76" marker-end="url(#fr-pf)"></line>
  <line class="acc-stroke" x1="440" y1="76" x2="460" y2="76" marker-end="url(#fr-pf)"></line>
  <line class="acc-stroke" x1="590" y1="76" x2="610" y2="76" marker-end="url(#fr-pf)"></line>
  <line class="acc-stroke" x1="740" y1="76" x2="760" y2="76" marker-end="url(#fr-pf)"></line>

  <rect x="10" y="20" width="130" height="112"></rect>
  <line x1="24" y1="34" x2="24" y2="120"></line>
  <line x1="24" y1="120" x2="116" y2="120"></line>
  <circle cx="34" cy="114" r="2"></circle>
  <circle cx="44" cy="106" r="2"></circle>
  <circle cx="54" cy="110" r="2"></circle>
  <circle cx="64" cy="98" r="2"></circle>
  <circle cx="74" cy="102" r="2"></circle>
  <circle cx="84" cy="88" r="2"></circle>
  <circle cx="94" cy="82" r="2"></circle>
  <circle cx="104" cy="70" r="2"></circle>
  <circle cx="114" cy="58" r="2"></circle>
  <text x="75" y="146" text-anchor="middle" font-size="9">raw prices</text>
  <text x="75" y="159" text-anchor="middle" font-size="8" opacity="0.65">load_prices</text>

  <rect x="160" y="20" width="130" height="112"></rect>
  <rect x="174" y="50" width="20" height="20"></rect>
  <rect x="196" y="50" width="20" height="20"></rect>
  <rect x="218" y="50" width="20" height="20"></rect>
  <rect x="240" y="50" width="20" height="20" class="acc-soft"></rect>
  <line x1="238" y1="46" x2="238" y2="94"></line>
  <text x="225" y="146" text-anchor="middle" font-size="9">lag features</text>
  <text x="225" y="159" text-anchor="middle" font-size="8" opacity="0.65">make_features(n=3)</text>

  <rect x="310" y="20" width="130" height="112"></rect>
  <rect x="324" y="66" width="68" height="28"></rect>
  <rect x="392" y="66" width="34" height="28" class="acc-soft"></rect>
  <text x="375" y="146" text-anchor="middle" font-size="9">train / validation</text>
  <text x="375" y="159" text-anchor="middle" font-size="8" opacity="0.65">time_split — no shuffling</text>

  <rect x="460" y="20" width="130" height="112"></rect>
  <path d="M478,50 Q501,110 524,110 Q547,110 570,50"></path>
  <circle class="acc-fill" cx="488" cy="68" r="2.5"></circle>
  <path class="acc-stroke" d="M488,68 L503,90" marker-end="url(#fr-pf)"></path>
  <circle class="acc-fill" cx="503" cy="90" r="2.5"></circle>
  <path class="acc-stroke" d="M503,90 L518,104" marker-end="url(#fr-pf)"></path>
  <circle class="acc-fill" cx="518" cy="104" r="2.5"></circle>
  <text x="525" y="146" text-anchor="middle" font-size="9">gradient descent</text>
  <text x="525" y="159" text-anchor="middle" font-size="8" opacity="0.65">on train rows only</text>

  <rect x="610" y="20" width="130" height="112"></rect>
  <text x="675" y="72" text-anchor="middle" font-size="10">X_val&middot;w+b</text>
  <line x1="675" y1="80" x2="675" y2="96" class="acc-stroke" marker-end="url(#fr-pf)"></line>
  <text x="675" y="112" text-anchor="middle" font-size="11" class="acc-fill">y&#770;_val</text>
  <text x="675" y="146" text-anchor="middle" font-size="9">predict on validation</text>
  <text x="675" y="159" text-anchor="middle" font-size="8" opacity="0.65">held-out rows only</text>

  <rect x="760" y="20" width="130" height="112"></rect>
  <text x="825" y="58" text-anchor="middle" font-size="14" class="acc-fill">?</text>
  <rect x="797" y="66" width="18" height="50"></rect>
  <rect x="829" y="66" width="18" height="50"></rect>
  <text x="806" y="127" text-anchor="middle" font-size="7" opacity="0.65">model</text>
  <text x="838" y="127" text-anchor="middle" font-size="7" opacity="0.65">naive</text>
  <text x="825" y="146" text-anchor="middle" font-size="9">compare vs baseline</text>
  <text x="825" y="159" text-anchor="middle" font-size="8" opacity="0.65">model MSE vs baseline MSE</text>

</svg>
</div>

Raw prices become features, features get split before any fitting happens,
gradient descent sees only the train rows, and the verdict — model against
the naive "next week equals this week" guess — is never assumed in
advance.

## The shape underneath

Look back at those five stages and one thing stands out: Stage 01 through
Stage 02c were never really five different problems. They were one
operation, done at a larger scale each time.

<svg viewBox="0 0 300 150" role="img" aria-label="Many weeks of lagged prices, arranged as matrix X, multiplied by weight vector w, equal predicted prices y-hat, compared against actual prices y with a residual.">
  <text x="28" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t-2</text>
  <text x="52" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t-1</text>
  <text x="76" y="10" text-anchor="middle" font-size="7.5" opacity="0.65">t</text>
  <rect x="16" y="18" width="24" height="24"></rect>
  <rect x="40" y="18" width="24" height="24"></rect>
  <rect x="64" y="18" width="24" height="24"></rect>
  <rect x="16" y="42" width="24" height="24"></rect>
  <rect x="40" y="42" width="24" height="24"></rect>
  <rect x="64" y="42" width="24" height="24"></rect>
  <rect x="16" y="66" width="24" height="24"></rect>
  <rect x="40" y="66" width="24" height="24"></rect>
  <rect x="64" y="66" width="24" height="24"></rect>
  <text x="28" y="82" text-anchor="middle" font-size="11">&#8942;</text>
  <text x="52" y="82" text-anchor="middle" font-size="11">&#8942;</text>
  <text x="76" y="82" text-anchor="middle" font-size="11">&#8942;</text>
  <rect x="16" y="90" width="24" height="24"></rect>
  <rect x="40" y="90" width="24" height="24"></rect>
  <rect x="64" y="90" width="24" height="24"></rect>
  <text x="52" y="130" text-anchor="middle" font-size="9" opacity="0.65">X &nbsp;&mdash;&nbsp; n weeks</text>
  <text x="98" y="66" text-anchor="middle" font-size="14">&times;</text>
  <rect x="108" y="30" width="24" height="24"></rect>
  <rect x="108" y="54" width="24" height="24"></rect>
  <rect x="108" y="78" width="24" height="24"></rect>
  <text x="120" y="24" text-anchor="middle" font-size="8" opacity="0.65">w</text>
  <text x="142" y="66" text-anchor="middle" font-size="14">=</text>
  <rect x="152" y="18" width="24" height="24" class="acc-soft"></rect>
  <rect x="152" y="42" width="24" height="24" class="acc-soft"></rect>
  <rect x="152" y="66" width="24" height="24" class="acc-soft"></rect>
  <text x="164" y="82" text-anchor="middle" font-size="11" class="acc-fill">&#8942;</text>
  <rect x="152" y="90" width="24" height="24" class="acc-soft"></rect>
  <text x="164" y="10" text-anchor="middle" font-size="8" class="acc-fill">y&#770;</text>
  <line x1="176" y1="102" x2="196" y2="102" class="acc-stroke" marker-start="url(#fr-syn1)" marker-end="url(#fr-syn2)"></line>
  <defs>
    <marker id="fr-syn1" viewBox="0 0 10 10" refX="2" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
      <path class="arrow-fill acc" d="M10,0 L0,5 L10,10 z"></path>
    </marker>
    <marker id="fr-syn2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
      <path class="arrow-fill acc" d="M0,0 L10,5 L0,10 z"></path>
    </marker>
  </defs>
  <rect x="198" y="18" width="24" height="24"></rect>
  <rect x="198" y="42" width="24" height="24"></rect>
  <rect x="198" y="66" width="24" height="24"></rect>
  <text x="210" y="82" text-anchor="middle" font-size="11">&#8942;</text>
  <rect x="198" y="90" width="24" height="24"></rect>
  <text x="210" y="10" text-anchor="middle" font-size="8" opacity="0.65">y</text>
  <text x="186" y="126" text-anchor="middle" font-size="8" class="acc-fill">residual</text>
</svg>

*Every week becomes a row; the weight vector never changes size. This is
the shape every later stage scales up.*

> Stage 03 keeps the same `X`. It asks a harder question of it — not a
> price, but a direction.

</div>
