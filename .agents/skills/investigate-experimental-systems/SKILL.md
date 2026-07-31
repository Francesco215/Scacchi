---
name: investigate-experimental-systems
description: Investigate experimental algorithms and learning systems by reconstructing their information flow, verifying claims against primary evidence, locating contradictions, forming falsifiable hypotheses, and designing economical validation experiments. Use when analyzing research code, training behavior, evaluation results, failed experiments, competing explanations, or deciding what experiment to run next.
---

# Investigate Experimental Systems

Use evidence to determine where a system creates, transforms, loses, stores, or misuses information. Derive implementation-specific details from the current sources instead of preserving conclusions from an older experiment.

## Define the question

State the uncertainty or decision being investigated. Specify what evidence could change the answer and what outcome would count as sufficient resolution.

Do not begin with a preferred intervention. Begin with the behavior that needs explanation.

## Reconstruct the information flow

Instantiate this abstract loop for the current system:

```text
current system
  -> observations or generated data
  -> derived signal or target
  -> update
  -> new system
  -> external behavior
  -> future observations or data
```

For every arrow, ask:

- What information is available?
- How is it selected, transformed, or aggregated?
- What can be discarded, distorted, or amplified?
- How is the result consumed downstream?
- Does downstream behavior change the future data distribution?

Distinguish these claims:

1. Useful information exists.
2. The procedure exposes it.
3. The update captures it.
4. The system retains and generalizes it.
5. The stored change improves the real objective.

Do not infer a later claim solely from evidence for an earlier one.

## Find primary evidence

Inspect the most authoritative available sources:

1. Mathematical specification and design documents.
2. Executed implementation paths.
3. Tests and invariants.
4. Resolved configurations and checkpoints.
5. Raw logs and evaluation artifacts.
6. Historical summaries and commentary.

Trace reported metrics back to the code that computes them. Verify their population, weighting, denominator, timing, aggregation, and units. Treat names such as `optimal`, `information`, `confidence`, or `improvement` as claims requiring verification.

Prefer direct artifacts over recollections. Search broadly first, then follow the narrow execution path relevant to the question.

## Separate evidence from interpretation

Keep these categories explicit:

- **Observation:** directly measured or read from an authoritative source.
- **Deduction:** follows mathematically or logically from stated premises.
- **Assumption:** accepted temporarily but not established.
- **Hypothesis:** a falsifiable proposed explanation.
- **Conclusion:** supported within stated conditions and uncertainty.

State the conditions and data distribution under which each result holds. Do not equate displacement, correlation, lower loss, or internal consistency with useful information unless an appropriate external criterion supports that interpretation.

## Use contradictions to localize failures

Compare theory, implementation, internal metrics, and external behavior. When they disagree, locate the earliest arrow in the information flow where the evidence becomes inconsistent.

Generate alternative explanations that could produce all observations, including:

- a measurement artifact;
- a distribution shift;
- a confounded intervention;
- insufficient transfer or retention;
- downstream thresholding or compression;
- a correct local mechanism that does not improve the global objective.

Avoid fixating on one parameter when several subsystems remain plausible.

## Form a falsifiable hypothesis

Write the hypothesis before running the experiment:

```text
Observation:
Proposed mechanism:
Intervention:
Variables held fixed:
Predicted measurements:
Falsifying result:
Budget:
Stop or continue rule:
External confirmation:
```

Make predictions specific enough to fail. Separate mechanism predictions from objective-level predictions.

## Validate economically

Run the cheapest test capable of rejecting the hypothesis. Prefer read-only inspection, analytic checks, synthetic cases, frozen inputs, or offline comparisons before a costly closed-loop run when they answer the same question.

Use controls, paired inputs or common random coordinates when appropriate, repeated trials, explicit compute accounting, and uncertainty estimates. Freeze acceptance and checkpoint-selection rules before inspecting outcomes.

Verify that the intervention changes only the intended information-flow edge. If it also changes behavior, data collection, targets, compute, or evaluation, record those as part of the causal intervention.

## Interpret and update

Report:

- which predictions passed or failed;
- whether the proposed mechanism was supported;
- whether the real objective improved;
- important limitations and alternative explanations;
- the next uncertainty, if any.

Preserve failed predictions. Update confidence rather than rewriting the original hypothesis after seeing the result.

Periodically step back and reconsider the full information flow. Stop when the original question meets its declared resolution standard; record interesting follow-ups as separate work.

## Scacchi case-study sources

When working in Scacchi, consult [math.md](../../../math.md) and
[index.html](../../../index.html) as high-level descriptions of the method and
its intended information flow. Treat them as project sources to verify against
the current implementation, not as implementation-independent rules or
automatically current ground truth.
