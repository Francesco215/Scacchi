
# Dirichlet-Q AlphaZero: Math Reference

## 1. Outcome space

For a two-player zero-sum game, define the terminal outcome from the current player's perspective as

$$
z \in \mathcal{Z} = \{L, D, W\}
$$

where

$$
L = -1, \qquad D = 0, \qquad W = +1.
$$

For each WDL distribution, write

$$
\phi = (\phi_L, \phi_D, \phi_W) \in \Delta^2.
$$

Here $\phi_z$ is the probability of outcome $z$ from the perspective of the player to move.

---

## 2. Action Dirichlet-Q head

For each state-action pair $(s,a)$, the network predicts Dirichlet parameters

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
\phi_{s,a}^Q \sim \textrm{Dirichlet}\left(\alpha_\theta^Q(s,a)\right).
$$

The posterior mean WDL distribution is

$$
\bar{\phi}_{\theta,z}^Q(s,a) =
\frac{\alpha_{\theta,z}^Q(s,a)}
{\alpha_{\theta,0}^Q(s,a)},
\qquad
\alpha_{\theta,0}^Q(s,a) =
\sum_{z \in \{L,D,W\}} \alpha_{\theta,z}^Q(s,a).
$$

The concentration

$$
\alpha_{\theta,0}^Q(s,a)
$$

controls epistemic uncertainty. Low concentration means broad uncertainty over the true WDL probabilities; high concentration means high confidence.

A stable parameterization is

$$
m_\theta^Q(s,a) =
\textrm{softmax}(r_\theta^Q(s,a))
\in \Delta^2
$$

and

$$
e_\theta^Q(s,a) =
\textrm{softplus}(c_\theta^Q(s,a)) > 0.
$$

Then

$$
\alpha_\theta^Q(s,a) =
\alpha_{\mathrm{base}} + e_\theta^Q(s,a)m_\theta^Q(s,a),
$$

where usually

$$
\alpha_{\mathrm{base}} = (1,1,1).
$$

---

## 3. State Dirichlet value head

The network also predicts a state-value Dirichlet distribution

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

The latent WDL value distribution of the state is

$$
\phi_s^V \sim \textrm{Dirichlet}\left(\alpha_\theta^V(s)\right).
$$

The value-head mean is

$$
\bar{\phi}_{\theta,z}^V(s) =
\frac{\alpha_{\theta,z}^V(s)}
{\alpha_{\theta,0}^V(s)},
\qquad
\alpha_{\theta,0}^V(s) =
\sum_{z \in \{L,D,W\}} \alpha_{\theta,z}^V(s).
$$

The value head is used to evaluate non-terminal search leaves.

The Q head answers:

$$
Q(s,a) =
\text{WDL belief after choosing action } a \text{ in state } s.
$$

The value head answers:

$$
V(s) =
\text{WDL belief for state } s.
$$

---

## 4. Utility of a WDL distribution

Given a WDL vector

$$
\phi = (p_L, p_D, p_W),
$$

define the value-style utility

$$
U(\phi) = p_W - p_L.
$$

Alternatively, define the expected-score utility

$$
S(\phi) = p_W + \frac{1}{2}p_D.
$$

These are affine-equivalent because

$$
S(\phi) = \frac{1}{2}\left(1 + U(\phi)\right).
$$

Therefore, either utility induces the same action ranking.

---

## 5. Bayesian optimal-action policy

At state $s$, each legal action $a$ has an uncertain WDL distribution

$$
\phi_{s,a}^Q \sim
\textrm{Dirichlet}\left(\alpha_\theta^Q(s,a)\right).
$$

The sampled best action is

$$
A^\star =
\arg\max_{a \in \mathcal{A}(s)}
U(\phi_{s,a}^Q).
$$

The Bayesian policy is the posterior probability that each action is optimal:

$$
\pi_{\mathrm{Bayes}}(a \mid s) =
\mathbb{P}\left(A^\star = a \mid s\right).
$$

Equivalently,

$$
\pi_{\mathrm{Bayes}}(a \mid s) =
\mathbb{P}\left(
a =
\arg\max_{b \in \mathcal{A}(s)}
U(\phi_{s,b}^Q)
\right),
\qquad
\phi_{s,b}^Q \sim
\textrm{Dirichlet}\left(\alpha_\theta^Q(s,b)\right).
$$

This is the direct generalization of posterior probability matching / Thompson sampling from Bayesian bandits.

---

## 6. Monte Carlo estimator of the Bayesian policy

For $m = 1,\dots,M$, sample

$$
\phi_{s,a}^{Q,(m)} \sim
\textrm{Dirichlet}\left(\alpha_\theta^Q(s,a)\right)
$$

for every legal action $a$, then compute

$$
a_m^\star =
\arg\max_{a \in \mathcal{A}(s)}
U\left(\phi_{s,a}^{Q,(m)}\right).
$$

The Monte Carlo posterior-best policy is

$$
\hat{\pi}_{\mathrm{Bayes}}(a \mid s) =
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}\left[a_m^\star = a\right].
$$

This distribution can be used as a policy target, or as a prior before deeper search.

---

## 7. Policy head as amortized posterior-best predictor

The network also outputs a policy head

$$
\pi_\theta(a \mid s) =
\textrm{softmax}(\ell_\theta(s,a)).
$$

The intended meaning is

$$
\pi_\theta(a \mid s)
\approx
\mathbb{P}\left(
a =
\arg\max_b U(\phi_{s,b}^Q)
\right).
$$

That is:

> The policy head predicts the posterior probability that each move is optimal.

---

## 8. Leaf evaluation

Search may stop at either terminal or non-terminal leaves.

### Terminal leaf

If the leaf is terminal, the returned WDL target is one-hot:

$$
y_{\mathrm{terminal}} = e_z \in \{(1,0,0),(0,1,0),(0,0,1)\}.
$$

A terminal Dirichlet target can be written as

$$
\beta_{\mathrm{terminal}} =
\alpha_{\mathrm{base}}
+
c_{\mathrm{terminal}} e_z.
$$

Here $c_{\mathrm{terminal}} > 0$ is the effective evidence strength of a terminal outcome.

### Non-terminal leaf
> Double check this!
> 
If the leaf is non-terminal, use the value head.

The value-head mean is

$$
y_{\mathrm{leaf}} =
\bar{\phi}_\theta^V(s_\ell) =
\frac{\alpha_\theta^V(s_\ell)}
{\alpha_{\theta,0}^V(s_\ell)}.
$$

A calibrated non-terminal Dirichlet target is

$$
\beta_{\mathrm{leaf}} =
\alpha_{\mathrm{base}}
+
c_{\mathrm{leaf}} y_{\mathrm{leaf}}.
$$

Here $c_{\mathrm{leaf}} > 0$ is the effective evidence strength of a neural leaf evaluation.

Usually,

$$
c_{\mathrm{terminal}} > c_{\mathrm{leaf}}.
$$

This encodes the fact that terminal outcomes are more reliable than bootstrapped neural estimates.

One may also directly use

$$
\beta_{\mathrm{leaf}} = \alpha_\theta^V(s_\ell),
$$

but early in training it is often safer to use the value-head mean with a fixed calibrated evidence strength $c_{\mathrm{leaf}}$.

---

## 9. Perspective flip in two-player games

WDL distributions are always represented from the perspective of the player to move.

If

$$
y = (y_L, y_D, y_W),
$$

then the opponent-perspective WDL is

$$
\textrm{flip}(y) =
(y_W, y_D, y_L).
$$

The value-style utility changes sign under this flip:

$$
U(\textrm{flip}(y)) = -U(y).
$$

For Dirichlet parameters, the same flip applies:

$$
\textrm{flip}(\beta_L,\beta_D,\beta_W) =
(\beta_W,\beta_D,\beta_L).
$$

During backup, apply this flip whenever the perspective changes between players.

---

## 10. Root posterior during search

At the root state $s$, initialize the per-action posterior as

$$
\alpha_a^{(0)} =
\alpha_\theta^Q(s,a).
$$

A root simulation chooses an action using Thompson-style posterior sampling:

$$
\phi_a^{(t)} \sim
\textrm{Dirichlet}\left(\alpha_a^{(t)}\right)
$$

for every legal action $a$, then

$$
a_t =
\arg\max_{a \in \mathcal{A}(s)}
U(\phi_a^{(t)}).
$$

After evaluating action $a_t$ by search, suppose the backed-up leaf evaluation gives a WDL mean and evidence strength

$$
\textrm{Eval}(s,a_t) =
\left(y_{a_t}^{(t)}, c_{a_t}^{(t)}\right),
\qquad
y_{a_t}^{(t)} \in \Delta^2,
\qquad
c_{a_t}^{(t)} > 0.
$$

The corresponding evaluated Dirichlet target is

$$
\beta_{a_t}^{(t)} =
\alpha_{\mathrm{base}}
+
c_{a_t}^{(t)} y_{a_t}^{(t)}.
$$

The effective evidence part is

$$
e_{a_t}^{(t)} =
\beta_{a_t}^{(t)}
-\alpha_{\mathrm{base}}=c_{a_t}^{(t)} y_{a_t}^{(t)}.
$$

A simple pseudo-Bayesian update is

$$
\alpha_{a_t}^{(t+1)} =
\alpha_{a_t}^{(t)}
+
\rho e_{a_t}^{(t)}=
\alpha_{a_t}^{(t)}+
\rho c_{a_t}^{(t)} y_{a_t}^{(t)}.
$$

For actions not selected at simulation $t$,

$$
\alpha_a^{(t+1)} =
\alpha_a^{(t)},
\qquad
a \neq a_t.
$$

Here $\rho > 0$ is a calibration hyperparameter.

This is exact Bayesian updating only if the search evaluation corresponds to independent categorical outcome evidence. In neural search, it should be interpreted as calibrated pseudo-evidence.

---

## 11. Sublinear evidence version

Because neural search evaluations are correlated and bootstrapped, linear evidence growth can become overconfident.

A safer version maintains a running WDL average for each root action.

Let $N(a)$ be the number of times root action $a$ has been evaluated, and let $\bar{y}_a$ be its running WDL average.

After receiving a new evaluation $y_a^{(t)}$, update

$$
\bar{y}_a^{(t+1)} =
\frac{
N(a)\bar{y}_a^{(t)}
+
y_a^{(t)}
}
{N(a)+1}.
$$

Then choose an effective search concentration such as

$$
c_a^{\mathrm{search}} =
\rho \sqrt{N(a)+1}.
$$

The root posterior is then

$$
\alpha_a^{(t+1)} =
\alpha_\theta^Q(s,a)
+
c_a^{\mathrm{search}}\bar{y}_a^{(t+1)}.
$$

Other possible evidence schedules include

$$
c_a^{\mathrm{search}} =
\rho \log(1+N(a))
$$

or

$$
c_a^{\mathrm{search}} =
\rho N(a).
$$

The linear version is closest to Bayesian counting. The sublinear versions are usually safer when the evidence comes from neural search rather than independent terminal samples.

---

## 12. Search-improved posterior-best policy

After $T$ root simulations, the root posterior is

$$
\alpha_a^{(T)}.
$$

The search-improved Bayesian policy is

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}\left(
a =
\arg\max_{b \in \mathcal{A}(s)}
U(\phi_b)
\right),
\qquad
\phi_b \sim
\textrm{Dirichlet}\left(\alpha_b^{(T)}\right).
$$

Estimate it by Monte Carlo:

$$
\hat{\pi}_{\mathrm{search}}(a \mid s) =
\frac{1}{M}
\sum_{m=1}^{M}
\mathbf{1}\left[a_m^\star = a\right],
$$

where

$$
a_m^\star =
\arg\max_{b \in \mathcal{A}(s)}
U\left(\phi_b^{(m)}\right),
\qquad
\phi_b^{(m)} \sim
\textrm{Dirichlet}\left(\alpha_b^{(T)}\right).
$$

This is the final policy target.

Unlike standard AlphaZero, the policy target is not just visit count. It is the posterior probability that each move is optimal after search.

---

## 13. Policy loss

Train the policy head toward the search-improved posterior-best policy:

$$
\mathcal{L}_{\pi}(s) =
-\sum_{a \in \mathcal{A}(s)}
\textrm{stopgrad}
\left(
\hat{\pi}_{\mathrm{search}}(a \mid s)
\right)
\log \pi_\theta(a \mid s).
$$

Equivalently,

$$
\mathcal{L}_{\pi}(s) =
D_{\mathrm{KL}}
\left(
\textrm{stopgrad}
\left(
\hat{\pi}_{\mathrm{search}}(\cdot \mid s)
\right)
\,\|\,
\pi_\theta(\cdot \mid s)
\right)
$$

up to an additive constant independent of $\theta$.

---

## 14. Value loss

For each searched state $s$, search produces a value target.

The mean target is

$$
y_V^{\mathrm{search}}(s) \in \Delta^2.
$$

The value-head predicted WDL mean is

$$
\bar{\phi}_\theta^V(s) =
\frac{\alpha_\theta^V(s)}
{\alpha_{\theta,0}^V(s)}.
$$

A simple value mean loss is

$$
\mathcal{L}_{V,\mathrm{mean}}(s) =
-\sum_{z \in \{L,D,W\}}
y_{V,z}^{\mathrm{search}}(s)
\log
\bar{\phi}_{\theta,z}^V(s).
$$

If the search also produces a Dirichlet value target

$$
\beta_V^{\mathrm{search}}(s) =
\alpha_{\mathrm{base}}
+
c_V^{\mathrm{search}}(s)
y_V^{\mathrm{search}}(s),
$$

then train the value head with a Dirichlet KL loss:

$$
\mathcal{L}_{V,\mathrm{Dir}}(s) =
D_{\mathrm{KL}}
\left(
\textrm{Dirichlet}
\left(
\textrm{stopgrad}
\left(
\beta_V^{\mathrm{search}}(s)
\right)
\right)
\,\|\,
\textrm{Dirichlet}
\left(
\alpha_\theta^V(s)
\right)
\right).
$$

The mean loss trains the value mean. The Dirichlet KL loss trains both the value mean and the value concentration.

A practical early version can use

$$
\mathcal{L}_V(s) =
\mathcal{L}_{V,\mathrm{mean}}(s).
$$

A more uncertainty-aware version can use

$$
\mathcal{L}_V(s) =
\mathcal{L}_{V,\mathrm{Dir}}(s).
$$

---

## 15. Q loss with Dirichlet KL term

For searched root actions, suppose search produces an action-level WDL target

$$
y_a^{\mathrm{search}} =
\left(
y_{a,L}^{\mathrm{search}},
y_{a,D}^{\mathrm{search}},
y_{a,W}^{\mathrm{search}}
\right)
\in \Delta^2.
$$

The Q-head predicted WDL mean is

$$
\bar{\phi}_{\theta,z}^Q(s,a) =
\frac{\alpha_{\theta,z}^Q(s,a)}
{\alpha_{\theta,0}^Q(s,a)}.
$$

A Q mean loss is

$$
\mathcal{L}_{Q,\mathrm{mean}}(s) =
-\sum_{a \in \mathcal{A}_{\mathrm{visited}}(s)}
w_a
\sum_{z \in \{L,D,W\}}
y_{a,z}^{\mathrm{search}}
\log
\bar{\phi}_{\theta,z}^Q(s,a).
$$

To train both the Q mean and Q concentration, construct a Dirichlet target

$$
\beta_a^{\mathrm{search}} =
\alpha_{\mathrm{base}}
+
c_a^{\mathrm{search}}
y_a^{\mathrm{search}}.
$$

Then use the Q Dirichlet KL loss

$$
\mathcal{L}_{Q,\mathrm{Dir}}(s) =
\sum_{a \in \mathcal{A}_{\mathrm{visited}}(s)}
w_a
D_{\mathrm{KL}}
\left(
\textrm{Dirichlet}
\left(
\textrm{stopgrad}
\left(
\beta_a^{\mathrm{search}}
\right)
\right)
\,\|\,
\textrm{Dirichlet}
\left(
\alpha_\theta^Q(s,a)
\right)
\right).
$$

The total Q loss can combine both terms:

$$
\mathcal{L}_Q(s) =
\lambda_{Q,\mathrm{mean}}
\mathcal{L}_{Q,\mathrm{mean}}(s)
+
\lambda_{Q,\mathrm{Dir}}
\mathcal{L}_{Q,\mathrm{Dir}}(s).
$$

The weights $w_a$ may be uniform, proportional to visit count, or proportional to posterior-best probability.

A minimal first version may set

$$
\lambda_{Q,\mathrm{Dir}} = 0.
$$

Then later, once the targets are better calibrated, increase $\lambda_{Q,\mathrm{Dir}}$.

---

## 16. Dirichlet KL formula

The KL divergence between two Dirichlet distributions is

$$
D_{\mathrm{KL}}
\left(
\textrm{Dir}(\beta)
\,\|\,
\textrm{Dir}(\alpha)
\right)=\log\Gamma(\beta_0)
-\sum_z \log\Gamma(\beta_z)
-\log\Gamma(\alpha_0)
+\sum_z \log\Gamma(\alpha_z)
+\sum_z
(\beta_z - \alpha_z)
\left[
\psi(\beta_z) - \psi(\beta_0)
\right],
$$

where

$$
\beta_0 =
\sum_z \beta_z,
\qquad
\alpha_0 =
\sum_z \alpha_z.
$$

Here $\Gamma$ is the gamma function and $\psi$ is the digamma function.

---

## 17. Outcome loss from self-play

For the actually played action $a_t$ in state $s_t$, the final game outcome gives a one-hot target

$$
e_{z_t} \in \{0,1\}^3.
$$

The value-head outcome loss is

$$
\mathcal{L}_{V,\mathrm{outcome}}(s_t) =
-\sum_{z \in \{L,D,W\}}
e_{z_t,z}
\log
\frac{
\alpha_{\theta,z}^V(s_t)
}
{
\alpha_{\theta,0}^V(s_t)
}.
$$

The Q-head outcome loss for the played action is

$$
\mathcal{L}_{Q,\mathrm{outcome}}(s_t,a_t) =
-\sum_{z \in \{L,D,W\}}
e_{z_t,z}
\log
\frac{
\alpha_{\theta,z}^Q(s_t,a_t)
}
{
\alpha_{\theta,0}^Q(s_t,a_t)
}.
$$

These are grounded in real terminal game outcomes, unlike bootstrapped leaf targets.

---

## 18. Total loss

A full training loss may be

$$
\mathcal{L} =
\lambda_\pi \mathcal{L}_\pi
+
\lambda_V \mathcal{L}_V
+
\lambda_Q \mathcal{L}_Q
+
\lambda_{V,\mathrm{outcome}}
\mathcal{L}_{V,\mathrm{outcome}}
+
\lambda_{Q,\mathrm{outcome}}
\mathcal{L}_{Q,\mathrm{outcome}}
+
\lambda_{\mathrm{reg}}
\mathcal{L}_{\mathrm{reg}}.
$$

A practical first version can use

$$
\mathcal{L} =
\lambda_\pi \mathcal{L}_\pi
+
\lambda_V \mathcal{L}_{V,\mathrm{mean}}
+
\lambda_Q \mathcal{L}_{Q,\mathrm{mean}}.
$$

A more uncertainty-aware version can use

$$
\mathcal{L} =
\lambda_\pi \mathcal{L}_\pi
+
\lambda_V \mathcal{L}_{V,\mathrm{Dir}}
+
\lambda_Q \mathcal{L}_{Q,\mathrm{Dir}}.
$$

A hybrid version can use both mean and Dirichlet losses:

$$
\mathcal{L} =
\lambda_\pi \mathcal{L}_\pi
+
\lambda_{V,\mathrm{mean}}
\mathcal{L}_{V,\mathrm{mean}}
+
\lambda_{V,\mathrm{Dir}}
\mathcal{L}_{V,\mathrm{Dir}}
+
\lambda_{Q,\mathrm{mean}}
\mathcal{L}_{Q,\mathrm{mean}}
+
\lambda_{Q,\mathrm{Dir}}
\mathcal{L}_{Q,\mathrm{Dir}}.
$$

---

## 19. Core interpretation

The value head represents a Bayesian belief over the WDL value of a state:

$$
\alpha_\theta^V(s)
\quad
\Longrightarrow
\quad
p(\phi_s^V \mid s).
$$

The Q head represents a Bayesian belief over the WDL value of each action:

$$
\alpha_\theta^Q(s,a)
\quad
\Longrightarrow
\quad
p(\phi_{s,a}^Q \mid s,a).
$$

The policy head predicts the posterior probability that an action is optimal:

$$
\pi_\theta(a \mid s)
\approx
\mathbb{P}
\left(
a =
\arg\max_b U(\phi_{s,b}^Q)
\right).
$$

Search improves the root action posteriors:

$$
\alpha_\theta^Q(s,a)
\rightarrow
\alpha_a^{(T)}.
$$

The search-improved policy is

$$
\pi_{\mathrm{search}}(a \mid s) =
\mathbb{P}
\left(
a =
\arg\max_b U(\phi_b)
\right),
\qquad
\phi_b \sim
\textrm{Dirichlet}
\left(
\alpha_b^{(T)}
\right).
$$

The policy head distills this expensive Bayesian search target into a fast amortized prediction.

The value head makes non-terminal leaf evaluation simple:

$$
s_\ell
\rightarrow
\alpha_\theta^V(s_\ell)
\rightarrow
y_{\mathrm{leaf}}.
$$

The Q head makes action-level posterior uncertainty explicit:

$$
(s,a)
\rightarrow
\alpha_\theta^Q(s,a)
\rightarrow
\mathbb{P}(a \text{ is optimal}).
$$
