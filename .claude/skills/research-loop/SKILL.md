---
name: research-loop
description: How to do research on Scacchi (Curiosity-Driven Tree Search) the way its author wants - read training curves and evals rather than final numbers, derive every change from the NLL and the probability model with no tricks, and revise the theory from what the curves show. Use this whenever you launch, monitor, compare or report on a training run or ablation; whenever you propose or derive a change to the loss, the search, the backup or the belief representation; whenever a run surprises you; and before editing math.md or index.html. Use it even if the request is only "run hex6" or "is X better than Y".
---

# Research loop for Scacchi

The project's contribution is a tree search that follows from a few simple
ideas. Its value is that simplicity and generality, not a win-rate number. The
blog (`index.html`) says it directly: "mathematical elegance over raw
performance: a principled foundation can always be optimized later". Work in
that spirit. Three habits follow.

## 1. Look at curves and evals, not scalars

A final number hides the things that matter here: a run that is still
climbing, one that learned late, a belief that is quietly collapsing. Two
examples from this repo: the maximum backup scored 1% on Hex6 at a short
budget and 51% at a longer one; under the exact KL the Beta concentration fell
from 3.1 to 0.09 in eight iterations while the loss looked fine for the first
five. Neither is visible in an end-of-run table.

So before saying anything about a run:

1. Draw the curves and open the image with the Read tool.

   ```bash
   .venv/bin/python .claude/skills/research-loop/scripts/curves.py \
       new=<history> reference=<history> --out outputs/<experiment>/curves.png
   ```

   A history is a local JSONL (`+logging.jsonl_path=...`) or
   `wandb:pal/scacchi-az/<run_id>`. Always put a reference run on the same
   axes; a curve alone says little. Add `--logx` for long runs (the blog plots
   against compute on a log axis) and `--metric <regex>` for anything else.
2. Read shapes, in this order:
   - **Eval win rate vs the solved baseline** over training: when does it leave
     zero, how steep is the transition, is it still rising at the cutoff? A
     rising curve means "train longer", never "this is the level".
   - **Search-target NLL for V and Q**: does it fall smoothly, plateau, or
     blow up? This is the quantity the theory is about.
   - **What the beliefs are doing**: policy target entropy (the blog's
     Figure 13), belief entropy/variance for binned heads, predicted vs
     search-target concentration for Beta heads. Predicted and target moving
     together in one direction is drift, not learning.
   - **Outcome NLL and the fraction of proven targets**, as context for the
     above.
3. Describe what the curves do, then what that implies. Quote a number only to
   locate a feature of a curve ("leaves zero around 300k frames").

Do not rank variants by end-of-run win rate, and do not answer a question
about mechanism with more seeds. The author's words: "seed variation can be a
thing, but that is not a serious experiment". If two curves overlap, say they
overlap.

Log to W&B by default (project `scacchi-az`; set `WANDB_RUN_GROUP` and
`WANDB_NAME` so runs are findable) and also write a local JSONL so curves can
be drawn without the network. Show the figure to the author, not a table of
endpoints.

## 2. Derive from the NLL, and keep it simple and general

Writing equations does not make something rigorous. A derivation counts only
if it starts from the probability model and each step is either an identity or
an assumption stated in words. The model is small:

- A game outcome is a draw from an unknown win probability (win/draw/loss in
  general). A belief is a distribution over that probability.
- The network is trained by the negative log-likelihood of what search
  concluded, `E_{q_search}[-log q_theta]` (the same gradient as
  `KL(q_search || q_theta)`).
- A move is chosen with the probability that it is the best one under the
  current beliefs (Thompson sampling).
- Search makes beliefs more precise by looking deeper and passing what it
  finds back up.

For any proposed piece, answer these before writing code:

- **What is the random variable, and what distribution does the model assign
  to it?** If you cannot say, there is nothing to take a likelihood of.
- **Is the loss the NLL of that variable?** If it is a surrogate with "the
  same optimum", it is a different objective; say so and keep it out of the
  default path.
- **Where did each constant come from?** A number that is not derived
  (a reference concentration, a prior strength, a count per simulation, a clip
  threshold) is a trick. Tricks have a short life in the literature and they
  hide what the theory is missing.
- **Does it hold for a general distribution and a general game?** Prefer
  statements that are true for any belief (bins, Beta, Dirichlet) over ones
  that lean on a parametric family or on Hex having no draws.
- **Would it fit in the blog as one equation and one sentence?** If it needs a
  page of caveats, it is not the right formulation yet.

Things already tried here that fail these tests, so do not reintroduce them
as if principled: treating simulations as Bernoulli observations to add
`n` to the concentration ("abusing math"); the dispersion surrogate loss; the
hand-set reference concentration for proven nodes; moment-matching a result
back into a Beta.

When the only way to make something work is a trick, do not ship the trick.
Report the gap: "the theory does not determine X; here is the smallest case
that shows it". An honest open question is worth more than a patch.

Name every assumption a derivation uses (for example "the examined child is
independent of the best of the others") and build the smallest exact case
where it holds and one where it fails. Exact toy cases with brute-force
enumeration are the first test of any update rule; training runs come after.

## 3. Let the curves revise the theory, tastefully

The loop:

1. **Predict.** Before a run, write down what the theory says the curves
   should look like and what would contradict it.
2. **Run and look** (section 1).
3. **Locate the disagreement.** Which statement in the model does the curve
   bear on? "Concentration falls every iteration" bears on how search targets
   are formed, not on the optimiser.
4. **Change the theory at its smallest point.** Prefer removing an assumption
   or a constant to adding a term. The result should be simpler or more
   general than what it replaces. If the change adds a knob, it is probably
   treating a symptom.
5. **Re-derive, test on exact toy cases, then go back to step 1.**

Taste here means: the fix explains the curve, makes a new prediction that can
be checked, and leaves the blog's story shorter rather than longer. A change
that only moves a win rate, without a reason the curve should move, is not a
theory update.

Keep experimental rules as non-default options until the author adopts them;
do not change defaults, `math.md` or `index.html` to match an idea that has
not been agreed.

## Reporting

Lead with what the curves showed and attach the figure. Then: which part of
the theory it bears on, the proposed change as one displayed equation, what
that change predicts, and what is still open. State assumptions and untested
claims plainly. In chat, write maths as display blocks with `$$ ... $$`;
inline maths does not render for the author.
