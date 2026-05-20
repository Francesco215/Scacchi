# Dirichlet-Q AlphaZero: Math Reference

## 1. Big picture

The network predicts three objects at each state:

1. a policy head $\pi_\theta(a \mid s)$,
2. a state-value Dirichlet head $\alpha_\theta^V(s)$,
3. an action-value Dirichlet-Q head $\alpha_\theta^Q(s,a)$.

The value head represents uncertainty over the WDL value of a state:

$$
\alpha_\theta^V(s) \Longrightarrow p(\phi_s^V \mid s).
$$

The Q head represents uncertainty over the WDL value of each action:

$$
\alpha_\theta^Q(s,a) \Longrightarrow p(\phi_{s,a}^Q \mid s,a).
$$

Search evaluates actions and accumulates WDL evidence by the first root action.
For policy improvement and Q-target construction, each action uses the
next-state value Dirichlet as its prior, aligned back to the root player's
perspective:

$$
\alpha_a^{\mathrm{child}}(s)
=
\operatorname{flip}
\left(
\alpha_{\theta^-}^V(s_a)
\right),
\qquad
s_a = \operatorname{Step}(s,a).
$$

The action posterior used for the search-improved policy and Q target is

$$
\alpha_a^{\mathrm{post}}(s)
=
\alpha_a^{\mathrm{child}}(s)
+
E_Q(s,a),
$$

The policy target after search is the posterior probability that each action is optimal:

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}
\left(
a = \arg\max_b U(\phi_b)
\right).
$$

The policy head distills this expensive Bayesian search target into a fast amortized prediction:

$$
\pi_\theta(a \mid s)
\approx
\pi_{\mathrm{search}}(a \mid s).
$$

The value head is used to evaluate non-terminal search leaves and to provide
the one-ply child priors for action-level posterior targets. The Q head is
trained to predict these action-level posteriors, but it is not used as the
base prior for the current policy or Q targets. Both Dirichlet heads are
trained by KL divergence toward posterior Dirichlet targets produced by
Bayesian-style evidence updates.

---

## 2. Outcome space, utility, and perspective

For a two-player zero-sum game, define the terminal outcome from the current player's perspective as

$$
z \in \mathcal{Z} = \{L,D,W\},
$$

where

$$
L = -1, \qquad D = 0, \qquad W = +1.
$$

A WDL distribution is

$$
\phi = (p_L,p_D,p_W) \in \Delta^2.
$$

Here $p_z$ is the probability of outcome $z$ from the perspective of the player to move.

Use the same utility everywhere:

$$
U(\phi) = p_W - p_L.
$$

This utility is used for search selection, posterior-best policy construction, final action selection, and policy training.

For any action-level WDL sample $\phi_a$, the preferred action is

$$
a^\star =
\arg\max_a U(\phi_a).
$$

WDL distributions are always represented from the perspective of the player to move. If

$$
d = (d_L,d_D,d_W),
$$

then the opponent-perspective WDL distribution is

$$
\operatorname{flip}(d) =
(d_W,d_D,d_L).
$$

The utility changes sign under this flip:

$$
U(\operatorname{flip}(d)) =
-U(d).
$$

The same flip applies to Dirichlet parameters:

$$
\operatorname{flip}(\alpha_L,\alpha_D,\alpha_W) =
(\alpha_W,\alpha_D,\alpha_L).
$$

During backup, apply this flip whenever the player-to-move perspective changes.

---

## 3. Network outputs

The network has three heads:

$$
\pi_\theta(a \mid s),
\qquad
\alpha_\theta^V(s),
\qquad
\alpha_\theta^Q(s,a).
$$

### 3.1 Policy head

The policy head is

$$
\pi_\theta(a \mid s) =
\operatorname{softmax}(\ell_\theta(s,a)).
$$

Its intended meaning is

$$
\pi_\theta(a \mid s)
\approx
\mathbb{P}
\left(
a =
\arg\max_b U(\phi_{s,b}^Q)
\right).
$$

That is, the policy head predicts the posterior probability that each move is optimal.

---

### 3.2 State-value Dirichlet head

The value head predicts Dirichlet parameters

$$
\alpha_\theta^V(s) =
\left(
\alpha_{\theta,L}^V(s),
\alpha_{\theta,D}^V(s),
\alpha_{\theta,W}^V(s)
\right),
\qquad
\alpha_{\theta,z}^V(s) > 0.
$$

The latent WDL value distribution is modeled as

$$
\phi_s^V
\sim
\operatorname{Dirichlet}
\left(
\alpha_\theta^V(s)
\right).
$$

The value-head mean is

$$
\bar{\phi}_{\theta,z}^V(s) =
\frac{\alpha_{\theta,z}^V(s)}
{\alpha_{\theta,0}^V(s)},
\qquad
\alpha_{\theta,0}^V(s) =
\sum_{z \in \{L,D,W\}}
\alpha_{\theta,z}^V(s).
$$

The value head answers:

$$
V(s) =
\text{WDL belief for state } s.
$$

---

### 3.3 Action Dirichlet-Q head

The Q head predicts Dirichlet parameters for each state-action pair:

$$
\alpha_\theta^Q(s,a) =
\left(
\alpha_{\theta,L}^Q(s,a),
\alpha_{\theta,D}^Q(s,a),
\alpha_{\theta,W}^Q(s,a)
\right),
\qquad
\alpha_{\theta,z}^Q(s,a) > 0.
$$

The latent WDL distribution for action $a$ is modeled as

$$
\phi_{s,a}^Q
\sim
\operatorname{Dirichlet}
\left(
\alpha_\theta^Q(s,a)
\right).
$$

The Q-head mean is

$$
\bar{\phi}_{\theta,z}^Q(s,a) =
\frac{\alpha_{\theta,z}^Q(s,a)}
{\alpha_{\theta,0}^Q(s,a)},
\qquad
\alpha_{\theta,0}^Q(s,a) =
\sum_{z \in \{L,D,W\}}
\alpha_{\theta,z}^Q(s,a).
$$

The Q head answers:

$$
Q(s,a) =
\text{WDL belief after choosing action } a \text{ in state } s.
$$

---

## 4. Stable Dirichlet parameterization

For both the value head and the Q head, use a mean-concentration parameterization.

The network predicts:

- WDL mean logits $r_\theta$,
- one scalar concentration logit $t_\theta$.

Define the predicted WDL mean as

$$
\bar{\phi}_\theta =
\operatorname{softmax}(r_\theta)
\in \Delta^2.
$$

Define the Dirichlet concentration as

$$
\alpha_{\theta,0} =
\operatorname{softplus}(t_\theta) > 0.
$$

Then define the Dirichlet parameters as

$$
\alpha_\theta =
\alpha_{\theta,0}\bar{\phi}_\theta.
$$

Equivalently, component-wise:

$$
\alpha_{\theta,z} =
\operatorname{softplus}(t_\theta)\bar{\phi}_{\theta,z},
\qquad
z \in \{L,D,W\}.
$$

There is no additive base Dirichlet prior in the network parameterization.

The mean is

$$
\mathbb{E}[\phi_z]
=
\frac{\alpha_{\theta,z}}{\alpha_{\theta,0}}
=
\bar{\phi}_{\theta,z}.
$$

The concentration is

$$
\alpha_{\theta,0}
=
\sum_z \alpha_{\theta,z}.
$$

Low concentration means broad epistemic uncertainty over the true WDL probabilities. High concentration means high confidence.

For the value head:

$$
\bar{\phi}_\theta^V(s) =
\operatorname{softmax}(r_\theta^V(s)),
\qquad
\alpha_{\theta,0}^V(s) =
\operatorname{softplus}(t_\theta^V(s)),
$$

and

$$
\alpha_\theta^V(s) =
\alpha_{\theta,0}^V(s)
\bar{\phi}_\theta^V(s).
$$

For the Q head:

$$
\bar{\phi}_\theta^Q(s,a) =
\operatorname{softmax}(r_\theta^Q(s,a)),
\qquad
\alpha_{\theta,0}^Q(s,a) =
\operatorname{softplus}(t_\theta^Q(s,a)),
$$

and

$$
\alpha_\theta^Q(s,a) =
\alpha_{\theta,0}^Q(s,a)
\bar{\phi}_\theta^Q(s,a).
$$

---

## 5. Evidence and Dirichlet updates

Let

$$
\phi
\sim
\operatorname{Dirichlet}(\alpha)
$$

be a prior over WDL outcome probabilities.

Suppose we receive WDL evidence

$$
d =
(d_L,d_D,d_W)
\in \Delta^2
$$

with evidence strength

$$
\lambda > 0.
$$

Then the posterior is

$$
p(\phi \mid d,\lambda) =
\operatorname{Dirichlet}
\left(
\alpha + \lambda d
\right).
$$

Equivalently, the updated Dirichlet parameters are

$$
\alpha'_z =
\alpha_z + \lambda d_z,
\qquad
z \in \{L,D,W\}.
$$

For a terminal outcome $z^\star$, the WDL evidence is one-hot:

$$
d =
e_{z^\star}.
$$

Then the posterior is

$$
p(\phi \mid z^\star,\lambda) =
\operatorname{Dirichlet}
\left(
\alpha + \lambda e_{z^\star}
\right).
$$

If $\lambda = 1$, this is the standard Dirichlet update from one observed categorical sample.

For multiple independent categorical observations with counts

$$
n =
(n_L,n_D,n_W),
$$

the exact posterior is

$$
p(\phi \mid n) =
\operatorname{Dirichlet}
\left(
\alpha + n
\right).
$$

The soft-evidence version is recovered by setting

$$
n =
\lambda d.
$$

In neural search, $\lambda d$ should be interpreted as calibrated pseudo-evidence unless it comes from actual independent terminal samples.

The important point is that $d$ can be one-hot because the posterior target is not $\lambda d$ alone. The posterior target is

$$
\alpha + \lambda d.
$$

Since the prior $\alpha$ is strictly positive, the posterior is always a valid Dirichlet distribution.

---

## 6. Leaf evaluation and backup

Search may stop at either terminal or non-terminal leaves.

The role of leaf evaluation is to return WDL evidence:

$$
\operatorname{Eval}(s_\ell) =
(d_\ell,\lambda_\ell),
\qquad
d_\ell \in \Delta^2,
\qquad
\lambda_\ell > 0.
$$

---

### 6.1 Terminal leaf

If the leaf is terminal with outcome $z$, return the one-hot WDL target:

$$
d_\ell =
e_z.
$$

Use terminal evidence strength:

$$
\lambda_\ell =
c_{\mathrm{terminal}}.
$$

Therefore, the evidence contribution is

$$
\lambda_\ell d_\ell =
c_{\mathrm{terminal}} e_z.
$$

---

### 6.2 Non-terminal leaf

If the leaf is non-terminal, use the value head.

The value-head mean is

$$
d_\ell =
\bar{\phi}_\theta^V(s_\ell).
$$

Using the stable parameterization, this is

$$
d_\ell =
\operatorname{softmax}
\left(
r_\theta^V(s_\ell)
\right).
$$

Use neural leaf evidence strength:

$$
\lambda_\ell =
c_{\mathrm{leaf}}.
$$

Therefore, the evidence contribution is

$$
\lambda_\ell d_\ell =
c_{\mathrm{leaf}}
\bar{\phi}_\theta^V(s_\ell).
$$

Usually,

$$
c_{\mathrm{terminal}} > c_{\mathrm{leaf}}.
$$

This encodes the fact that terminal outcomes are more reliable than bootstrapped neural evaluations.

---

### 6.3 Backup with perspective changes

During backup, return the evidence to the state or action being trained.

Whenever the player-to-move perspective changes, flip the WDL vector:

$$
d \leftarrow \operatorname{flip}(d).
$$

The backed-up evaluation for a root action $a$ has the form

$$
\operatorname{Eval}(s,a) =
(d_a,\lambda_a),
\qquad
d_a \in \Delta^2,
\qquad
\lambda_a > 0.
$$

The root action posterior can then be updated with

$$
\alpha_a
\leftarrow
\alpha_a + \lambda_a d_a.
$$

---

## 7. Root posterior search

At root state $s$, define the one-ply child state for each legal action:

$$
s_a = \operatorname{Step}(s,a).
$$

The value head on $s_a$ predicts the value from the next player-to-move
perspective. Align it back to the root player's perspective:

$$
\alpha_a^{(0)} =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
=
\operatorname{flip}
\left(
\alpha_{\theta^-}^V(s_a)
\right),
\qquad
a \in \mathcal{A}(s).
$$

This action prior is a value-head Dirichlet over the next state, reinterpreted
as the prior for the root action's outcome. The Q head still predicts
$\alpha_\theta^Q(s,a)$, but in this target construction it is trained toward
the posterior generated from the child value prior plus search evidence.

Using the stable value parameterization:

$$
\alpha_a^{(0)} =
\operatorname{flip}
\left[
\operatorname{softplus}(t_{\theta^-}^V(s_a))
\operatorname{softmax}
\left(
r_{\theta^-}^V(s_a)
\right)
\right]
$$

At simulation $t$, sample one WDL distribution per legal action:

$$
\phi_a^{(t)}
\sim
\operatorname{Dirichlet}
\left(
\alpha_a^{(t)}
\right),
\qquad
a \in \mathcal{A}(s).
$$

Select the action with highest sampled utility:

$$
a_t =
\arg\max_{a \in \mathcal{A}(s)}
U(\phi_a^{(t)}).
$$

After evaluating $a_t$ by search, suppose the backed-up evaluation gives

$$
\operatorname{Eval}(s,a_t) =
\left(
d_{a_t}^{(t)},
\lambda_{a_t}^{(t)}
\right),
\qquad
d_{a_t}^{(t)} \in \Delta^2,
\qquad
\lambda_{a_t}^{(t)} > 0.
$$

Update the selected action posterior as

$$
\alpha_{a_t}^{(t+1)} =
\alpha_{a_t}^{(t)}
+
\lambda_{a_t}^{(t)}
d_{a_t}^{(t)}.
$$

For actions not selected at simulation $t$:

$$
\alpha_a^{(t+1)} =
\alpha_a^{(t)},
\qquad
a \neq a_t.
$$

Equivalently:

$$
p
\left(
\phi_{a_t}
\mid
d_{a_t}^{(t)},
\lambda_{a_t}^{(t)}
\right)
=
\operatorname{Dirichlet}
\left(
\alpha_{a_t}^{(t)}
+
\lambda_{a_t}^{(t)}
d_{a_t}^{(t)}
\right).
$$

This is exact Bayesian updating if the evidence corresponds to independent categorical outcome evidence. In neural search, it is calibrated pseudo-evidence.

After $T$ root simulations, the final root posterior is

$$
\alpha_a^{(T)},
\qquad
a \in \mathcal{A}(s).
$$

In the current MCTX implementation, search traversal itself may use MCTX's
root selection rule, but the posterior used for policy targets is reconstructed
from the accumulated root-action evidence and this child value prior.

---

## 8. Search-improved policy target

After search, define the search-improved policy as the posterior probability that each action is optimal under utility.

The posterior used here is the child-value-prior posterior:

$$
\alpha_b^{(T)}
=
\alpha_{\theta^-}^{V \rightarrow Q}(s,b)
+
E_Q(s,b),
$$

where $E_Q(s,b)$ is the accumulated root-player-perspective evidence for
action $b$.

For each legal action $a$:

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}
\left(
a =
\arg\max_{b \in \mathcal{A}(s)}
U(\phi_b)
\right),
\qquad
\phi_b
\sim
\operatorname{Dirichlet}
\left(
\alpha_b^{(T)}
\right).
$$

Estimate this probability by Monte Carlo.

For $m = 1,\dots,M$, sample

$$
\phi_b^{(m)}
\sim
\operatorname{Dirichlet}
\left(
\alpha_b^{(T)}
\right),
\qquad
b \in \mathcal{A}(s).
$$

Then compute

$$
a_m^\star =
\arg\max_{b \in \mathcal{A}(s)}
U(\phi_b^{(m)}).
$$

The Monte Carlo estimator is

$$
\hat{\pi}_{\mathrm{search}}(a \mid s) =
\frac{1}{M}
\sum_{m=1}^M
\mathbf{1}
\left[
a_m^\star = a
\right].
$$

This is the final policy target.

Unlike standard AlphaZero, the policy target is not just visit count. It is the posterior probability that each move is optimal after search.

---

## 9. Policy loss

Train the policy head toward the search-improved posterior-best policy:

$$
\mathcal{L}_\pi(s) =
-\sum_{a \in \mathcal{A}(s)}
\operatorname{stopgrad}
\left(
\hat{\pi}_{\mathrm{search}}(a \mid s)
\right)
\log \pi_\theta(a \mid s).
$$

Equivalently:

$$
\mathcal{L}_\pi(s) =
D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}
\left(
\hat{\pi}_{\mathrm{search}}(\cdot \mid s)
\right)
\,\|\,
\pi_\theta(\cdot \mid s)
\right)
$$

up to an additive constant independent of $\theta$.

---

## 10. Dirichlet training targets

The value and Q heads are trained by KL divergence toward posterior Dirichlet targets.

The key rule is:

$$
\beta =
\alpha_{\mathrm{prior}} + \lambda d.
$$

Here:

- $d \in \Delta^2$ is WDL evidence,
- $\lambda > 0$ is evidence strength,
- $\alpha_{\mathrm{prior}}$ is the positive Dirichlet prior used when the evidence was generated,
- $\beta$ is the posterior Dirichlet target.

There is no target of the form

$$
\beta =
\kappa d.
$$

That form would be invalid for one-hot $d$, but it is not the target used here.

The actual target is

$$
\beta_z =
\alpha_{\mathrm{prior},z}
+
\lambda d_z.
$$

Because

$$
\alpha_{\mathrm{prior},z} > 0
$$

for every $z$, the target satisfies

$$
\beta_z > 0
$$

for every $z$, even when $d$ is one-hot.

Therefore no smoothing is needed.

---

## 11. Search-time target storage

During self-play or search, the network snapshot used to create the search priors should be treated as fixed. Denote that snapshot by $\theta^-$.

For a state $s$, the value prior at data-generation time is

$$
\alpha_{\mathrm{prior}}^V(s) =
\alpha_{\theta^-}^V(s).
$$

For a state-action pair $(s,a)$, the action-level prior used for policy and Q
targets is the child value prior aligned to the root perspective:

$$
\alpha_{\mathrm{prior}}^{V \rightarrow Q}(s,a) =
\operatorname{flip}
\left(
\alpha_{\theta^-}^V(\operatorname{Step}(s,a))
\right).
$$

After evidence is collected, construct posterior targets:

$$
\beta_V(s) =
\alpha_{\mathrm{prior}}^V(s)
+
\lambda_V d_V(s),
$$

and

$$
\beta_Q(s,a) =
\alpha_{\mathrm{prior}}^{V \rightarrow Q}(s,a)
+
\lambda_Q d_Q(s,a).
$$

These $\beta$ targets should be stored or treated as stop-gradient targets during training.

Do not construct moving targets from the current trainable $\alpha_\theta$ inside the same gradient step. The target should represent the posterior produced by search, not a target that chases the current prediction.

---

## 12. Value target and value loss

For a state $s$, suppose search or self-play produces WDL evidence

$$
d_V(s) \in \Delta^2
$$

with evidence strength

$$
\lambda_V > 0.
$$

The value prior from the data-generation network is

$$
\alpha_{\mathrm{prior}}^V(s) =
\alpha_{\theta^-}^V(s).
$$

The posterior value target is

$$
\beta_V(s) =
\alpha_{\mathrm{prior}}^V(s)
+
\lambda_V d_V(s).
$$

Then train the current value head by KL:

$$
\mathcal{L}_V(s) =
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}
\left(
\operatorname{stopgrad}
\left(
\beta_V(s)
\right)
\right)
\,\|\,
\operatorname{Dirichlet}
\left(
\alpha_\theta^V(s)
\right)
\right).
$$

If the value evidence comes from a terminal outcome $z$, then

$$
d_V(s) =
e_z.
$$

The target becomes

$$
\beta_V(s) =
\alpha_{\theta^-}^V(s)
+
\lambda_V e_z.
$$

This is valid because $\alpha_{\theta^-}^V(s)$ is strictly positive.

If the value evidence comes from a non-terminal leaf, then

$$
d_V(s) =
\operatorname{stopgrad}
\left(
\bar{\phi}_{\theta^-}^V(s_\ell)
\right),
$$

after applying the correct perspective flip.

Then

$$
\beta_V(s) =
\alpha_{\theta^-}^V(s)
+
\lambda_V
\operatorname{stopgrad}
\left(
\bar{\phi}_{\theta^-}^V(s_\ell)
\right).
$$

---

## 13. Q target and Q loss

For a searched action $(s,a)$, suppose search produces WDL evidence

$$
d_Q(s,a) \in \Delta^2
$$

with evidence strength

$$
\lambda_Q > 0.
$$

The prior for the Q target is not the old Q-head prediction. It is the
one-ply child value Dirichlet from the data-generation network, aligned to the
root player's perspective:

$$
\alpha_{\mathrm{prior}}^{V \rightarrow Q}(s,a) =
\operatorname{flip}
\left(
\alpha_{\theta^-}^V(s_a)
\right),
\qquad
s_a = \operatorname{Step}(s,a).
$$

The posterior Q target is

$$
\beta_Q(s,a) =
\alpha_{\mathrm{prior}}^{V \rightarrow Q}(s,a)
+
\lambda_Q d_Q(s,a).
$$

Then train the current Q head by KL:

$$
\mathcal{L}_Q(s,a) =
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}
\left(
\operatorname{stopgrad}
\left(
\beta_Q(s,a)
\right)
\right)
\,\|\,
\operatorname{Dirichlet}
\left(
\alpha_\theta^Q(s,a)
\right)
\right).
$$

Average this only over explored actions, i.e. actions with positive
accumulated tree evidence:

$$
\mathcal{L}_Q(s) =
\operatorname{mean}_{a \in \mathcal{A}_{\mathrm{explored}}(s)}
\mathcal{L}_Q(s,a).
$$

where $\mathcal{A}_{\mathrm{explored}}(s)$ is

$$
\mathcal{A}_{\mathrm{explored}}(s)
=
\left\{
a \in \mathcal{A}(s)
:
\sum_{i \in \mathcal{I}(a)}
\lambda_i
> 0
\right\}.
$$

Unexplored legal actions are excluded from the Q KL. They are not trained
toward an unchanged prior.

---

## 14. Accumulated Q targets from root search

During root search, action $a$ may receive multiple pieces of evidence.

Let

$$
\mathcal{I}(a)
$$

be the set of simulations in which action $a$ was evaluated.

Each evaluation gives

$$
(d_i,\lambda_i),
\qquad
i \in \mathcal{I}(a).
$$

The initial action prior is the child value prior:

$$
\alpha_a^{(0)} =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
=
\operatorname{flip}
\left(
\alpha_{\theta^-}^V(\operatorname{Step}(s,a))
\right).
$$

The final posterior after all evidence for action $a$ is

$$
\alpha_a^{(T)} =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
+
\sum_{i \in \mathcal{I}(a)}
\lambda_i d_i.
$$

Therefore the Q training target for action $a$ can be

$$
\beta_Q(s,a) =
\alpha_a^{(T)}.
$$

Equivalently:

$$
\beta_Q(s,a) =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
+
\sum_{i \in \mathcal{I}(a)}
\lambda_i d_i.
$$

Then the Q loss is

$$
\mathcal{L}_Q(s,a) =
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}
\left(
\operatorname{stopgrad}
\left(
\alpha_a^{(T)}
\right)
\right)
\,\|\,
\operatorname{Dirichlet}
\left(
\alpha_\theta^Q(s,a)
\right)
\right).
$$

This trains the Q head to predict the posterior that search produced from the
child value prior plus evidence. The loss is applied only when
$\mathcal{I}(a)$ is non-empty, equivalently when the accumulated evidence mass
for action $a$ is positive.

---

## 15. Total loss

The full training objective is

$$
\mathcal{L} =
\lambda_\pi \mathcal{L}_\pi
+
\lambda_V \mathcal{L}_V
+
\lambda_Q \mathcal{L}_Q.
$$

The policy loss is

$$
\mathcal{L}_\pi(s) =
-\sum_{a \in \mathcal{A}(s)}
\operatorname{stopgrad}
\left(
\hat{\pi}_{\mathrm{search}}(a \mid s)
\right)
\log \pi_\theta(a \mid s).
$$

The value loss is

$$
\mathcal{L}_V(s) =
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}
\left(
\operatorname{stopgrad}
\left(
\beta_V(s)
\right)
\right)
\,\|\,
\operatorname{Dirichlet}
\left(
\alpha_\theta^V(s)
\right)
\right).
$$

The Q loss is

$$
\mathcal{L}_Q(s) =
\operatorname{mean}_{a \in \mathcal{A}_{\mathrm{explored}}(s)}
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}
\left(
\operatorname{stopgrad}
\left(
\beta_Q(s,a)
\right)
\right)
\,\|\,
\operatorname{Dirichlet}
\left(
\alpha_\theta^Q(s,a)
\right)
\right).
$$

The targets are posterior Dirichlets:

$$
\beta_V(s) =
\alpha_{\theta^-}^V(s)
+
\lambda_V d_V(s),
$$

and

$$
\beta_Q(s,a) =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
+
\sum_{i \in \mathcal{I}(a)}
\lambda_i d_i.
$$

No separate mean loss is required. The Dirichlet KL trains both the WDL mean and the concentration.

---

## 16. Dirichlet KL formula

For

$$
\operatorname{Dirichlet}(\alpha_1)
\quad\text{and}\quad
\operatorname{Dirichlet}(\alpha_2),
$$

with

$$
\alpha_{10} =
\sum_{i=1}^{k}
\alpha_{1i},
\qquad
\alpha_{20} =
\sum_{i=1}^{k}
\alpha_{2i},
$$

the KL divergence is

$$
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}(\alpha_1)
\,\|\,
\operatorname{Dirichlet}(\alpha_2)
\right)
=
\log
\frac{\Gamma(\alpha_{10})}
{\Gamma(\alpha_{20})}
+
\sum_{i=1}^{k}
\log
\frac{\Gamma(\alpha_{2i})}
{\Gamma(\alpha_{1i})}
+
\sum_{i=1}^{k}
(\alpha_{1i}-\alpha_{2i})
\left[
\psi(\alpha_{1i})
-
\psi(\alpha_{10})
\right].
$$

For training, use

$$
\alpha_1 =
\operatorname{stopgrad}(\beta),
\qquad
\alpha_2 =
\alpha_\theta.
$$

Here $\Gamma$ is the gamma function and $\psi$ is the digamma function.

---

## 17. Core algorithm summary

At a root state $s$, step each legal action once and evaluate the value head at
the child state. This gives one positive Dirichlet prior per legal action:

$$
\alpha_a^{(0)} =
\alpha_\theta^{V \rightarrow Q}(s,a)
=
\operatorname{flip}
\left(
\alpha_\theta^V(\operatorname{Step}(s,a))
\right).
$$

Using the stable value-head parameterization:

$$
\alpha_\theta^{V \rightarrow Q}(s,a) =
\operatorname{flip}
\left[
\operatorname{softplus}(t_\theta^V(\operatorname{Step}(s,a)))
\operatorname{softmax}
\left(
r_\theta^V(\operatorname{Step}(s,a))
\right)
\right]
$$

Search repeatedly samples from these posteriors:

$$
\phi_a^{(t)}
\sim
\operatorname{Dirichlet}
\left(
\alpha_a^{(t)}
\right).
$$

It chooses an action by utility:

$$
a_t =
\arg\max_a U(\phi_a^{(t)}).
$$

It evaluates the selected action and receives WDL evidence:

$$
(d_{a_t}^{(t)},\lambda_{a_t}^{(t)}).
$$

It updates the selected action posterior:

$$
\alpha_{a_t}^{(t+1)} =
\alpha_{a_t}^{(t)}
+
\lambda_{a_t}^{(t)}
d_{a_t}^{(t)}.
$$

Terminal leaves return one-hot evidence:

$$
d =
e_z.
$$

Non-terminal leaves return value-head mean evidence:

$$
d =
\bar{\phi}_\theta^V(s_\ell).
$$

After search, the policy target is the posterior probability that each action is optimal:

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}
\left(
a =
\arg\max_b U(\phi_b)
\right),
\qquad
\phi_b
\sim
\operatorname{Dirichlet}
\left(
\alpha_b^{(T)}
\right).
$$

The policy head learns this posterior-best distribution:

$$
\pi_\theta(a \mid s)
\approx
\pi_{\mathrm{search}}(a \mid s).
$$

The value head learns posterior state-level WDL beliefs:

$$
\beta_V(s) =
\alpha_{\theta^-}^V(s)
+
\lambda_V d_V(s).
$$

The Q head learns posterior action-level WDL beliefs:

$$
\beta_Q(s,a) =
\alpha_{\theta^-}^{V \rightarrow Q}(s,a)
+
\sum_{i \in \mathcal{I}(a)}
\lambda_i d_i.
$$

The Q KL is averaged only over explored actions with positive accumulated
evidence mass.

The central search primitive is always the same Dirichlet evidence update:

$$
p(\phi \mid d,\lambda) =
\operatorname{Dirichlet}
\left(
\alpha + \lambda d
\right).
$$

The central network parameterization is always the same mean-concentration form:

$$
\alpha_\theta =
\operatorname{softplus}(t_\theta)
\operatorname{softmax}(r_\theta).
$$

There is no additive $\alpha_{\mathrm{base}}$ and no smoothing term.
