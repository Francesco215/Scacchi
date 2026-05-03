# Dirichlet-Q AlphaZero

Dirichlet-Q AlphaZero is an experimental AlphaZero-style search and learning framework where value estimates are represented as **distributions over game outcomes** instead of scalar values.

For two-player zero-sum games, each state or action is modeled as a probability distribution over:

$$
\{L,D,W\}
$$

corresponding to loss, draw, and win from the current player's perspective.

The core idea is to replace scalar Q-values with **Dirichlet beliefs** over WDL outcome probabilities. This makes action-value uncertainty explicit and allows search to behave more like Bayesian posterior updating.

## Core idea

The network predicts three objects:

- a policy head $\pi_\theta(a \mid s)$,
- a state-value Dirichlet head $\alpha_\theta^V(s)$,
- an action-value Dirichlet-Q head $\alpha_\theta^Q(s,a)$.

The value head represents a belief over the WDL value of a state:

$$
\alpha_\theta^V(s) \Longrightarrow p(\phi_s^V \mid s).
$$

The Q head represents a belief over the WDL value of each action:

$$
\alpha_\theta^Q(s,a) \Longrightarrow p(\phi_{s,a}^Q \mid s,a).
$$

Search starts from the Q-head prior, evaluates actions, and updates action posteriors with WDL evidence:

$$
\alpha_a \leftarrow \alpha_a + c y_a.
$$

Here $y_a$ is a WDL target distribution and $c$ is an evidence-strength parameter.

## Search

At the root state, each legal action has a Dirichlet posterior:

$$
\phi_a \sim \operatorname{Dirichlet}(\alpha_a).
$$

Search samples from these posteriors to decide which action to explore. After evaluating an action, the resulting WDL evidence is backed up and used to update that action's posterior.

Terminal leaves return one-hot WDL evidence:

$$
y = e_z.
$$

Non-terminal leaves are evaluated using the value head:

$$
y = \bar{\phi}_\theta^V(s_\ell).
$$

This allows search to combine exact terminal outcomes with bootstrapped neural value estimates.

## Policy target

After search, the improved policy is not defined by visit counts. Instead, it is defined as the posterior probability that each action is optimal under utility:

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}\left(a = \arg\max_b U(\phi_b)\right).
$$

The utility of a WDL distribution is

$$
U(\phi) = p_W - p_L.
$$

The policy head is trained to predict this search-improved posterior-best distribution:

$$
\pi_\theta(a \mid s) \approx \pi_{\mathrm{search}}(a \mid s).
$$

## Training

The model can be trained with three main losses:

- policy loss, matching the posterior-best search policy;
- value loss, matching state-level WDL search targets;
- Q loss, matching action-level WDL search targets.

The simplest version trains only the WDL means with cross-entropy losses.

A more uncertainty-aware version trains the full Dirichlet distributions using KL divergence:

$$
D_{\mathrm{KL}}\left(
\operatorname{Dirichlet}(\beta)
\,\|\,
\operatorname{Dirichlet}(\alpha_\theta)
\right).
$$

Final self-play outcomes can also be used as grounded terminal targets for both the value head and the Q head.

## Why this is useful

Standard AlphaZero stores scalar action values and uses visit counts as policy targets. Dirichlet-Q AlphaZero instead keeps explicit posterior uncertainty over WDL outcomes.

This gives a more probabilistic interpretation of search:

- the Q head provides action-level priors;
- search gathers WDL evidence;
- root action posteriors are updated;
- the final policy is the probability that each move is optimal;
- the policy head amortizes this expensive Bayesian search procedure.

The goal is to make AlphaZero-style planning more uncertainty-aware, especially in settings where different actions have different levels of evidence.


