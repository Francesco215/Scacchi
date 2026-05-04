
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

Search starts from the Q-head prior, evaluates actions, and updates root action posteriors with WDL evidence:

$$
\alpha_a \longrightarrow \alpha_a + c y_a.
$$

After search, the improved policy target is the posterior probability that each action is optimal:

$$
\pi_{\mathrm{search}}(a \mid s) = \mathbb{P}\left(a = \arg\max_b U(\phi_b)\right).
$$

The policy head distills this expensive Bayesian search target into a fast amortized prediction:

$$
\pi_\theta(a \mid s) \approx \pi_{\mathrm{search}}(a \mid s).
$$

The value head is used to evaluate non-terminal search leaves. The Q head is used to initialize and train action-level posterior beliefs.

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
\phi = (\phi_L,\phi_D,\phi_W) \in \Delta^2.
$$

Equivalently, write

$$
\phi = (p_L,p_D,p_W).
$$

Here $\phi_z$ is the probability of outcome $z$ from the perspective of the player to move.

Define the value-style utility as

$$
U(\phi) = p_W - p_L.
$$

Alternatively, define the expected-score utility as

$$
S(\phi) = p_W + \frac{1}{2}p_D.
$$

These are affine-equivalent:

$$
S(\phi) = \frac{1}{2}\left(1 + U(\phi)\right).
$$

Therefore $U$ and $S$ induce the same action ranking.

For final action selection and policy training, use utility:

$$
a^\star = \arg\max_a U(\phi_a).
$$

For exploratory search, one may instead use win probability:

$$
G(\phi) = p_W.
$$

This gives the design distinction:

$$
\text{use } G(\phi) = p_W \text{ for exploration, and use } U(\phi) = p_W - p_L \text{ for playing and training.}
$$

This distinction only matters when draws are possible. In binary win/loss games,

$$
p_L = 1 - p_W,
$$

so

$$
U(\phi) = 2p_W - 1.
$$

Therefore maximizing $p_W$ and maximizing $U(\phi)$ are equivalent in binary win/loss games.

WDL distributions are always represented from the perspective of the player to move. If

$$
y = (y_L,y_D,y_W),
$$

then the opponent-perspective WDL distribution is

$$
\textrm{flip}(y) = (y_W,y_D,y_L).
$$

The utility changes sign under this flip:

$$
U(\textrm{flip}(y)) = -U(y).
$$

The same flip applies to Dirichlet parameters:

$$
\textrm{flip}(\alpha_L,\alpha_D,\alpha_W) = (\alpha_W,\alpha_D,\alpha_L).
$$

During backup, apply this flip whenever the perspective changes between players.

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
\pi_\theta(a \mid s) = \textrm{softmax}(\ell_\theta(s,a)).
$$

Its intended meaning is

$$
\pi_\theta(a \mid s) \approx \mathbb{P}\left(a = \arg\max_b U(\phi_{s,b}^Q)\right).
$$

That is, the policy head predicts the posterior probability that each move is optimal for playing.

### 3.2 State-value Dirichlet head

The value head predicts Dirichlet parameters

$$
\alpha_\theta^V(s) = \left(\alpha_{\theta,L}^V(s),\alpha_{\theta,D}^V(s),\alpha_{\theta,W}^V(s)\right),
\qquad
\alpha_{\theta,z}^V(s) > 0.
$$

The latent WDL value distribution is modeled as

$$
\phi_s^V \sim \textrm{Dirichlet}(\alpha_\theta^V(s)).
$$

The value-head mean is

$$
\bar{\phi}_{\theta,z}^V(s) = \frac{\alpha_{\theta,z}^V(s)}{\alpha_{\theta,0}^V(s)},
\qquad
\alpha_{\theta,0}^V(s) = \sum_{z \in \{L,D,W\}} \alpha_{\theta,z}^V(s).
$$

The value head answers:

$$
V(s) = \text{WDL belief for state } s.
$$

### 3.3 Action Dirichlet-Q head

The Q head predicts Dirichlet parameters for each state-action pair:

$$
\alpha_\theta^Q(s,a) = \left(\alpha_{\theta,L}^Q(s,a),\alpha_{\theta,D}^Q(s,a),\alpha_{\theta,W}^Q(s,a)\right),
\qquad
\alpha_{\theta,z}^Q(s,a) > 0.
$$

The latent WDL distribution for action $a$ is modeled as

$$
\phi_{s,a}^Q \sim \textrm{Dirichlet}(\alpha_\theta^Q(s,a)).
$$

The Q-head mean is

$$
\bar{\phi}_{\theta,z}^Q(s,a) = \frac{\alpha_{\theta,z}^Q(s,a)}{\alpha_{\theta,0}^Q(s,a)},
\qquad
\alpha_{\theta,0}^Q(s,a) = \sum_{z \in \{L,D,W\}} \alpha_{\theta,z}^Q(s,a).
$$

The Q head answers:

$$
Q(s,a) = \text{WDL belief after choosing action } a \text{ in state } s.
$$

### 3.4 Stable Dirichlet parameterization

For either the value head or the Q head, use a mean-concentration parameterization.

Let

$$
m_\theta = \textrm{softmax}(r_\theta) \in \Delta^2.
$$

Let

$$
e_\theta = \textrm{softplus}(c_\theta) > 0.
$$

Then define

$$
\alpha_\theta = \alpha_{\mathrm{base}} + e_\theta m_\theta.
$$

Usually,

$$
\alpha_{\mathrm{base}} = (1,1,1).
$$

The concentration is

$$
\alpha_0 = \sum_z \alpha_z.
$$

Low concentration means broad epistemic uncertainty over the true WDL probabilities. High concentration means high confidence.

---

## 4. Evidence and Dirichlet updates

Let

$$
\phi \sim \textrm{Dirichlet}(\alpha)
$$

be a prior over WDL outcome probabilities.

Suppose we receive WDL evidence

$$
y = (y_L,y_D,y_W) \in \Delta^2
$$

with evidence strength

$$
c > 0.
$$

Then the posterior is

$$
p(\phi \mid y,c) = \textrm{Dirichlet}(\alpha + c y).
$$

Equivalently, the updated Dirichlet parameters are

$$
\alpha'_z = \alpha_z + c y_z,
\qquad
z \in \{L,D,W\}.
$$

For a terminal outcome $z^\star$, the target is one-hot:

$$
y = e_{z^\star}.
$$

Then the posterior is

$$
p(\phi \mid z^\star,c) = \textrm{Dirichlet}(\alpha + c e_{z^\star}).
$$

If $c = 1$, this is the standard Dirichlet-categorical Bayesian update from one observed categorical sample.

For multiple independent categorical observations with counts

$$
n = (n_L,n_D,n_W),
$$

the exact posterior is

$$
p(\phi \mid n) = \textrm{Dirichlet}(\alpha + n).
$$

The soft-evidence version is recovered by setting

$$
n = c y.
$$

In neural search, $c y$ should be interpreted as calibrated pseudo-evidence unless it comes from actual independent terminal samples.

---

## 5. Leaf evaluation and backup

Search may stop at either terminal or non-terminal leaves.

The role of leaf evaluation is to return WDL evidence:

$$
\textrm{Eval}(s_\ell) = (y_\ell,c_\ell),
\qquad
y_\ell \in \Delta^2,
\qquad
c_\ell > 0.
$$

### 5.1 Terminal leaf

If the leaf is terminal with outcome $z$, return the one-hot WDL target:

$$
y_\ell = e_z.
$$

Use terminal evidence strength:

$$
c_\ell = c_{\mathrm{terminal}}.
$$

Therefore, the evidence contribution is

$$
c_\ell y_\ell = c_{\mathrm{terminal}} e_z.
$$

### 5.2 Non-terminal leaf

If the leaf is non-terminal, use the value head.

The value-head mean is

$$
y_\ell = \bar{\phi}_\theta^V(s_\ell) = \frac{\alpha_\theta^V(s_\ell)}{\alpha_{\theta,0}^V(s_\ell)}.
$$

Use neural leaf evidence strength:

$$
c_\ell = c_{\mathrm{leaf}}.
$$

Therefore, the evidence contribution is

$$
c_\ell y_\ell = c_{\mathrm{leaf}} \bar{\phi}_\theta^V(s_\ell).
$$

Usually,

$$
c_{\mathrm{terminal}} > c_{\mathrm{leaf}}.
$$

This encodes the fact that terminal outcomes are more reliable than bootstrapped neural evaluations.

### 5.3 Backup with perspective changes

During backup, return the evidence to the root action being evaluated.

Whenever the player-to-move perspective changes, flip the WDL vector:

$$
y \leftarrow \textrm{flip}(y).
$$

The backed-up evaluation for a root action $a$ has the form

$$
\textrm{Eval}(s,a) = (y_a,c_a),
\qquad
y_a \in \Delta^2,
\qquad
c_a > 0.
$$

The root action posterior can then be updated with the evidence update:

$$
\alpha_a \leftarrow \alpha_a + c_a y_a.
$$

---

## 6. Root posterior search

At root state $s$, initialize the per-action posterior from the Q head:

$$
\alpha_a^{(0)} = \alpha_\theta^Q(s,a),
\qquad
a \in \mathcal{A}(s).
$$

At simulation $t$, sample one WDL distribution per legal action:

$$
\phi_a^{(t)} \sim \textrm{Dirichlet}(\alpha_a^{(t)}),
\qquad
a \in \mathcal{A}(s).
$$

For exploration, one possible rule is to select the action with highest sampled win probability:

$$
a_t = \arg\max_{a \in \mathcal{A}(s)} \phi_{a,W}^{(t)}.
$$

Alternatively, one may select by sampled utility:

$$
a_t = \arg\max_{a \in \mathcal{A}(s)} U(\phi_a^{(t)}).
$$

The important distinction is:

$$
\text{search exploration may use } p_W, \text{ but final policy improvement should use } U.
$$

After evaluating $a_t$ by search, suppose the backed-up evaluation gives

$$
\textrm{Eval}(s,a_t) = \left(y_{a_t}^{(t)},c_{a_t}^{(t)}\right),
\qquad
y_{a_t}^{(t)} \in \Delta^2,
\qquad
c_{a_t}^{(t)} > 0.
$$

Update the selected action posterior as

$$
\alpha_{a_t}^{(t+1)} = \alpha_{a_t}^{(t)} + c_{a_t}^{(t)} y_{a_t}^{(t)}.
$$

For actions not selected at simulation $t$,

$$
\alpha_a^{(t+1)} = \alpha_a^{(t)},
\qquad
a \neq a_t.
$$

Equivalently,

$$
p(\phi_{a_t} \mid y_{a_t}^{(t)},c_{a_t}^{(t)}) = \textrm{Dirichlet}\left(\alpha_{a_t}^{(t)} + c_{a_t}^{(t)} y_{a_t}^{(t)}\right).
$$

This is exact Bayesian updating if the evidence corresponds to independent categorical outcome evidence. In neural search, it is calibrated pseudo-evidence.

After $T$ root simulations, the final root posterior is

$$
\alpha_a^{(T)},
\qquad
a \in \mathcal{A}(s).
$$

---

## 7. Search-improved policy target and policy loss

After search, define the search-improved policy as the posterior probability that each action is optimal under utility.

For each legal action $a$,

$$
\pi_{\mathrm{search}}(a \mid s) = \mathbb{P}\left(a = \arg\max_{b \in \mathcal{A}(s)} U(\phi_b)\right),
\qquad
\phi_b \sim \textrm{Dirichlet}(\alpha_b^{(T)}).
$$

Estimate this probability by Monte Carlo.

For $m = 1,\dots,M$, sample

$$
\phi_b^{(m)} \sim \textrm{Dirichlet}(\alpha_b^{(T)}),
\qquad
b \in \mathcal{A}(s).
$$

Then compute

$$
a_m^\star = \arg\max_{b \in \mathcal{A}(s)} U(\phi_b^{(m)}).
$$

The Monte Carlo estimator is

$$
\hat{\pi}_{\mathrm{search}}(a \mid s) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}\left[a_m^\star = a\right].
$$

This is the final policy target.

Unlike standard AlphaZero, the policy target is not just visit count. It is the posterior probability that each move is optimal after search.

Train the policy head toward the search-improved posterior-best policy:

$$
\mathcal{L}_\pi(s) = -\sum_{a \in \mathcal{A}(s)} \textrm{stopgrad}\left(\hat{\pi}_{\mathrm{search}}(a \mid s)\right) \log \pi_\theta(a \mid s).
$$

Equivalently,

$$
\mathcal{L}_\pi(s) = D_{\mathrm{KL}}\left(\textrm{stopgrad}\left(\hat{\pi}_{\mathrm{search}}(\cdot \mid s)\right) \,\|\, \pi_\theta(\cdot \mid s)\right)
$$

up to an additive constant independent of $\theta$.

---

## 8. Value and Q training losses

The training targets produced by search can be represented either as WDL mean targets or as Dirichlet targets.

Given a WDL target

$$
y \in \Delta^2
$$

and evidence strength

$$
c > 0,
$$

define the Dirichlet target

$$
\beta = \alpha_{\mathrm{base}} + c y.
$$

There are two useful losses.

### 8.1 Mean cross-entropy loss

Given predicted Dirichlet parameters $\alpha_\theta$, define the predicted mean

$$
\bar{\phi}_{\theta,z} = \frac{\alpha_{\theta,z}}{\alpha_{\theta,0}},
\qquad
\alpha_{\theta,0} = \sum_z \alpha_{\theta,z}.
$$

The mean loss is

$$
\mathcal{L}_{\mathrm{mean}} = -\sum_{z \in \{L,D,W\}} y_z \log \bar{\phi}_{\theta,z}.
$$

This trains the predicted WDL mean, but it does not directly train the concentration.

### 8.2 Dirichlet KL loss

The Dirichlet KL loss is

$$
\mathcal{L}_{\mathrm{Dir}} = D_{\mathrm{KL}}\left(\textrm{Dirichlet}\left(\textrm{stopgrad}(\beta)\right) \,\|\, \textrm{Dirichlet}(\alpha_\theta)\right).
$$

This trains both the predicted WDL mean and the predicted concentration.

### 8.3 Value loss

For each searched state $s$, suppose search produces a value target

$$
y_V^{\mathrm{search}}(s) \in \Delta^2.
$$

The value-head predicted mean is

$$
\bar{\phi}_{\theta,z}^V(s) = \frac{\alpha_{\theta,z}^V(s)}{\alpha_{\theta,0}^V(s)}.
$$

The value mean loss is

$$
\mathcal{L}_{V,\mathrm{mean}}(s) = -\sum_{z \in \{L,D,W\}} y_{V,z}^{\mathrm{search}}(s) \log \bar{\phi}_{\theta,z}^V(s).
$$

If search also provides an evidence strength $c_V^{\mathrm{search}}(s)$, define

$$
\beta_V^{\mathrm{search}}(s) = \alpha_{\mathrm{base}} + c_V^{\mathrm{search}}(s) y_V^{\mathrm{search}}(s).
$$

Then the value Dirichlet loss is

$$
\mathcal{L}_{V,\mathrm{Dir}}(s) = D_{\mathrm{KL}}\left(\textrm{Dirichlet}\left(\textrm{stopgrad}\left(\beta_V^{\mathrm{search}}(s)\right)\right) \,\|\, \textrm{Dirichlet}\left(\alpha_\theta^V(s)\right)\right).
$$

A practical early version can use only

$$
\mathcal{L}_V(s) = \mathcal{L}_{V,\mathrm{mean}}(s).
$$

A more uncertainty-aware version can use

$$
\mathcal{L}_V(s) = \mathcal{L}_{V,\mathrm{Dir}}(s).
$$

### 8.4 Q loss

For searched root actions, suppose search produces an action-level WDL target

$$
y_a^{\mathrm{search}} = \left(y_{a,L}^{\mathrm{search}},y_{a,D}^{\mathrm{search}},y_{a,W}^{\mathrm{search}}\right) \in \Delta^2.
$$

The Q-head predicted mean is

$$
\bar{\phi}_{\theta,z}^Q(s,a) = \frac{\alpha_{\theta,z}^Q(s,a)}{\alpha_{\theta,0}^Q(s,a)}.
$$

The Q mean loss is

$$
\mathcal{L}_{Q,\mathrm{mean}}(s) = -\sum_{a \in \mathcal{A}_{\mathrm{visited}}(s)} w_a \sum_{z \in \{L,D,W\}} y_{a,z}^{\mathrm{search}} \log \bar{\phi}_{\theta,z}^Q(s,a).
$$

To train both the Q mean and Q concentration, construct a Dirichlet target:

$$
\beta_a^{\mathrm{search}} = \alpha_{\mathrm{base}} + c_a^{\mathrm{search}} y_a^{\mathrm{search}}.
$$

Then use the Q Dirichlet KL loss:

$$
\mathcal{L}_{Q,\mathrm{Dir}}(s) = \sum_{a \in \mathcal{A}_{\mathrm{visited}}(s)} w_a D_{\mathrm{KL}}\left(\textrm{Dirichlet}\left(\textrm{stopgrad}\left(\beta_a^{\mathrm{search}}\right)\right) \,\|\, \textrm{Dirichlet}\left(\alpha_\theta^Q(s,a)\right)\right).
$$

The total Q loss can combine both terms:

$$
\mathcal{L}_Q(s) = \lambda_{Q,\mathrm{mean}}\mathcal{L}_{Q,\mathrm{mean}}(s) + \lambda_{Q,\mathrm{Dir}}\mathcal{L}_{Q,\mathrm{Dir}}(s).
$$

The weights $w_a$ may be uniform, proportional to visit count, or proportional to posterior-best probability.

A practical early version can use only

$$
\mathcal{L}_Q(s) = \mathcal{L}_{Q,\mathrm{mean}}(s).
$$

A more uncertainty-aware version can use

$$
\mathcal{L}_Q(s) = \mathcal{L}_{Q,\mathrm{Dir}}(s).
$$

### 8.5 Outcome loss from self-play

For the actually played action $a_t$ in state $s_t$, the final game outcome gives a one-hot target

$$
e_{z_t} \in \{0,1\}^3.
$$

This target is grounded in a real terminal game outcome, unlike bootstrapped search targets.

The value-head outcome loss is

$$
\mathcal{L}_{V,\mathrm{outcome}}(s_t) = -\sum_{z \in \{L,D,W\}} e_{z_t,z} \log \frac{\alpha_{\theta,z}^V(s_t)}{\alpha_{\theta,0}^V(s_t)}.
$$

The Q-head outcome loss for the played action is

$$
\mathcal{L}_{Q,\mathrm{outcome}}(s_t,a_t) = -\sum_{z \in \{L,D,W\}} e_{z_t,z} \log \frac{\alpha_{\theta,z}^Q(s_t,a_t)}{\alpha_{\theta,0}^Q(s_t,a_t)}.
$$

---

## 9. Total loss

A minimal first version can use

$$
\mathcal{L} = \lambda_\pi \mathcal{L}_\pi + \lambda_V \mathcal{L}_{V,\mathrm{mean}} + \lambda_Q \mathcal{L}_{Q,\mathrm{mean}}.
$$

A more uncertainty-aware version can use

$$
\mathcal{L} = \lambda_\pi \mathcal{L}_\pi + \lambda_V \mathcal{L}_{V,\mathrm{Dir}} + \lambda_Q \mathcal{L}_{Q,\mathrm{Dir}}.
$$

A hybrid version can use both mean and Dirichlet losses:

$$
\mathcal{L} = \lambda_\pi \mathcal{L}_\pi + \lambda_{V,\mathrm{mean}}\mathcal{L}_{V,\mathrm{mean}} + \lambda_{V,\mathrm{Dir}}\mathcal{L}_{V,\mathrm{Dir}} + \lambda_{Q,\mathrm{mean}}\mathcal{L}_{Q,\mathrm{mean}} + \lambda_{Q,\mathrm{Dir}}\mathcal{L}_{Q,\mathrm{Dir}}.
$$

One may also add grounded terminal outcome losses:

$$
\mathcal{L} = \lambda_\pi \mathcal{L}_\pi + \lambda_V \mathcal{L}_V + \lambda_Q \mathcal{L}_Q + \lambda_{V,\mathrm{outcome}}\mathcal{L}_{V,\mathrm{outcome}} + \lambda_{Q,\mathrm{outcome}}\mathcal{L}_{Q,\mathrm{outcome}} + \lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}.
$$

A practical training progression is:

1. start with mean losses,
2. add outcome losses from completed self-play games,
3. add Dirichlet KL losses once evidence strengths are calibrated.

---

## 10. Core algorithm summary

At a root state $s$, the Q head gives one Dirichlet prior per legal action:

$$
\alpha_a^{(0)} = \alpha_\theta^Q(s,a).
$$

Search repeatedly samples from these posteriors, chooses an action, evaluates it, and updates its posterior:

$$
\alpha_{a_t}^{(t+1)} = \alpha_{a_t}^{(t)} + c_{a_t}^{(t)} y_{a_t}^{(t)}.
$$

Terminal leaves return one-hot WDL evidence:

$$
y = e_z.
$$

Non-terminal leaves return value-head WDL evidence:

$$
y = \bar{\phi}_\theta^V(s_\ell).
$$

After search, the policy target is the posterior probability that each action is optimal:

$$
\pi_{\mathrm{search}}(a \mid s) = \mathbb{P}\left(a = \arg\max_b U(\phi_b)\right),
\qquad
\phi_b \sim \textrm{Dirichlet}(\alpha_b^{(T)}).
$$

The policy head learns this posterior-best distribution:

$$
\pi_\theta(a \mid s) \approx \pi_{\mathrm{search}}(a \mid s).
$$

The value head learns state-level WDL targets:

$$
\alpha_\theta^V(s) \approx \beta_V^{\mathrm{search}}(s).
$$

The Q head learns action-level WDL targets:

$$
\alpha_\theta^Q(s,a) \approx \beta_a^{\mathrm{search}}.
$$

The central primitive is always the same Dirichlet evidence update:

$$
p(\phi \mid y,c) = \textrm{Dirichlet}(\alpha + c y).
$$

---

## Appendix A. Dirichlet KL formula

The KL divergence between two Dirichlet distributions is

$$
D_{\mathrm{KL}}\left(\textrm{Dir}(\beta) \,\|\, \textrm{Dir}(\alpha)\right) = \log\Gamma(\beta_0) - \sum_z \log\Gamma(\beta_z) - \log\Gamma(\alpha_0) + \sum_z \log\Gamma(\alpha_z) + \sum_z (\beta_z - \alpha_z)\left[\psi(\beta_z) - \psi(\beta_0)\right].
$$

where

$$
\beta_0 = \sum_z \beta_z,
\qquad
\alpha_0 = \sum_z \alpha_z.
$$

Here $\Gamma$ is the gamma function and $\psi$ is the digamma function
