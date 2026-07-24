# Dirichlet Thompson Tree Search: Math and Implementation Reference

This document describes the active `scacchi.dirichlet_mctx` algorithm. It is
the JAX, fixed-capacity sister of MCTX used by `dirichlet_thompson` search. Its
posterior repair follows `tictactoe-demo/app.js`: search stores replacement
messages on edges, recomputes a posterior cache at every node on the selected
path, and uses Thompson sampling both for traversal and for policy readout.

The most important correction to the previous specification is:

> Search does **not** collect additive evidence by first root action. An edge
> message is replaced by the latest leaf or repaired-child Dirichlet, while a
> separate structural count controls how strongly a node trusts its repaired
> descendants.

That correction describes unresolved, uncertain values. Exact terminal and
solved values use a second, native representation: a categorical outcome and a
certified distance to terminal. Each certificate is stored by changing an
outcome tag and reinterpreting an existing integer payload, rather than by
allocating a parallel posterior object or encoding the result as a large or
sharply peaked Dirichlet. Certificates are absorbing and authoritative. Model
alphas remain the learned representation for unresolved leaves and caches; a
solved result is never reconstructed from their shape or concentration.

---

## 1. Objects predicted by the network

At a state $s$, the network returns

$$
\ell_\theta(s,a),
\qquad
V_s = \alpha_\theta^V(s),
\qquad
Q_{s,a} = \alpha_\theta^Q(s,a).
$$

Here $\ell_\theta$ are policy logits, $V_s$ is the state-value Dirichlet, and
$Q_{s,a}$ is the action-value Dirichlet. The two Dirichlet heads contain full
positive parameter vectors, not scalar values and not only their means.

For an ordered outcome space

$$
\mathcal Z = \{L,W\}
\quad\text{or}\quad
\mathcal Z = \{L,D,W\},
$$

write

$$
\phi \sim \operatorname{Dirichlet}(\alpha),
\qquad
\mu(\alpha)=\frac{\alpha}{\sum_z \alpha_z}.
$$

The scalar utility used everywhere is

$$
U(\phi)=\phi_W-\phi_L.
$$

For an exact categorical outcome, use the matching deterministic utility

$$
u_{\mathrm{cat}}(L)=-1,
\qquad
u_{\mathrm{cat}}(D)=0,
\qquad
u_{\mathrm{cat}}(W)=1.
$$

Thus uncertain edges draw $\phi$ and use $U(\phi)$, whereas categorical edges
use $u_{\mathrm{cat}}$ without sampling.

The policy logits are trained to imitate the search policy. Native Thompson
traversal does not add them to uncertain edge scores; it uses exact categorical
utilities or sampled action Dirichlets plus the legal-action mask. Logits are
not persistent tree state and categorical ties never request them. Search
samples uniformly among tied certified actions using its existing random key,
without reevaluating the stored embedding.

### 1.1 Perspective alignment

Every value is stored from the perspective of the player to move at its node.
For a source player $p$ and target player $q$, define

$$
\operatorname{Align}(x;p\to q)=
\begin{cases}
x, & p=q,\\
\operatorname{rev}(x), & p\ne q,
\end{cases}
$$

where reversal swaps loss and win and leaves draw fixed when it exists:

$$
\operatorname{rev}(x_L,x_D,x_W)=(x_W,x_D,x_L).
$$

The same operation applies to outcome probabilities and Dirichlet parameters.
It satisfies

$$
U(\operatorname{rev}(\phi))=-U(\phi).
$$

It also applies to categorical indices: loss and win are exchanged, draw is
fixed, and certified distance is unchanged.

---

## 2. Dirichlet parameterization

Each Dirichlet head predicts mean logits $r$ and a concentration logit $t$:

$$
\mu=\operatorname{softmax}(r),
\qquad
\alpha=c\mu.
$$

The modern `boardlaw_dirichlet` head uses a configurable concentration floor
$c_{\min}$ and optional ceiling $c_{\max}$:

$$
c=
\begin{cases}
c_{\min}+\operatorname{softplus}(t),
& c_{\max}\text{ is absent},\\[4pt]
c_{\min}+(c_{\max}-c_{\min})\operatorname{sigmoid}(t),
& c_{\max}\text{ is present}.
\end{cases}
$$

The optional unfloored head parameterization is

$$
c=\operatorname{softplus}(t)^2,
$$

optionally clipped above. In every case, search receives the resulting positive
vector $\alpha$ and does not reinterpret its learned concentration.

When categorical density supervision is active, the concentration must have a
finite ceiling or an explicit regularizer. Moving the target point into the
simplex interior prevents $\log 0$, but by itself does not stop point-density
likelihood from favoring unbounded concentration. Scacchi therefore requires a
finite `dirichlet_concentration_clip` for an active categorical head loss.

---

## 3. MCTX-shaped public API

The root contract parallels MCTX, with full Dirichlet values added:

```python
import functools

root = dirichlet_mctx.RootFnOutput(
    prior_logits=prediction.logits,
    value=prediction.alpha_v,
    action_values=prediction.alpha_q,
    embedding=env_state,
    terminal_outcome=root_terminal_outcome,
    to_play=env_state.current_player,
)

policy_output = dirichlet_mctx.dirichlet_thompson_policy(
    params=(),
    rng_key=rng_key,
    root=root,
    recurrent_fn=expand_fn,
    num_simulations=num_simulations,
    invalid_actions=~env_state.legal_action_mask,
    posterior_update=functools.partial(
        dirichlet_mctx.update_posterior,
        kappa=kappa,
    ),
    max_depth=max_depth,
    policy_samples=policy_samples,
)
```

Categorical tie-breaking needs no extra evaluation callback or configuration.
The search uses its existing random key to sample uniformly among equally good
certified actions. For a draw, every certified draw edge is tied; losing edges
are never candidates.
The sole scalar in the default repair rule is $\kappa>0$, appearing only in
$\gamma=n/(\kappa+n)$. It is node-prior strength, not leaf or terminal
evidence.

The MCTX-compatible argument remains named `recurrent_fn`. In Scacchi the
callable is named `expand_fn`, because one invocation does exactly

$$
(s,a)\longmapsto
\operatorname{Step}(s,a)
+\operatorname{Evaluate}(s').
$$

It returns the child embedding, $V_{s'}$, all $Q_{s',b}$, the child's legal
action mask, player to move, and one exact `terminal_outcome` tag. The tag is
an outcome index from the child's player perspective when the child is
terminal, and `NO_OUTCOME` otherwise. It is the complete terminal payload:
search uses it to publish the native categorical certificate without
constructing a terminal Dirichlet.

### 3.1 Native leaf result

The semantic leaf result is a tagged object:

$$
Y_{s'}=
\begin{cases}
\operatorname{Dir}(V_{s'}),&s'\text{ is non-terminal},\\
\operatorname{Cat}(z_{s'},0),&s'\text{ is terminal}.
\end{cases}
$$

The zero is the terminal node's certified distance. Its parent edge stores the
aligned outcome and one additional ply:

$$
z_{s,a}=\operatorname{Align}(z_{s'};p_{s'}\to p_s),
\qquad
\tau_{s,a}=1+\tau_{s'}=1.
$$

`RecurrentFnOutput.value` and `action_values` always contain the child's model
alphas. For a non-terminal frontier, `value` is the complete unresolved leaf
message $V_{s'}$. For a terminal frontier, `terminal_outcome` immediately owns
the semantics; the model alpha is not written as a terminal edge message and
is not used for selection, proof propagation, or a categorical training
target. There is no leaf-strength or terminal-strength constant.

---

## 4. Lightweight tree state

The tree is flat: it has no persistent `NodePosterior`. For each node it stores
fixed $V_s$, mutable $C_s$, topology, player, legality, embedding, an `int8`
outcome tag $z_s$, and one `int32` payload $d_s$. For each edge it stores one
alpha slot $H_{s,a}$, an `int8` outcome tag $z_{s,a}$, and one `int32` payload
$d_{s,a}$. Payload meanings depend on the outcome tags:

$$
d_{s,a}=\begin{cases}
R_{s,a},&z_{s,a}=\bot,\\
\tau_{s,a},&z_{s,a}\ne\bot,
\end{cases}
\qquad
d_s=\begin{cases}
n_s,&z_s=\bot,\\
\tau_s,&z_s\ne\bot.
\end{cases}
$$

These are the actual storage dtypes: outcome tags remain `int8`, while count,
support, and distance arithmetic remains `int32`. Reusing a payload slot never
requires a cast or aliases differently typed arrays. Alpha slots retain their
network floating dtype.

An unresolved node and a terminal node can both have payload zero. Therefore
terminal state is exactly $z_s\ne\bot\land d_s=0$, never merely $d_s=0$. A
terminal root supplies its exact outcome through
`RootFnOutput.terminal_outcome`. An expanded categorical child publishes

$$
z_{s,a}=\operatorname{Align}(z_{s_a};p_{s_a}\to p_s),
\qquad
\tau_{s,a}=1+\tau_{s_a}.
$$

The implementation does not store a categorical action. The action is derived
deterministically from immutable edge certificates whenever readout needs it.

The cache is initialized from the state prior:

$$
C_s\leftarrow V_s.
$$

The edge-alpha array is initialized with the network fallback $Q_{s,a}$. That
fallback occupies $H_{s,a}$ until a real message replaces it. While the edge
is unresolved, the slot is a real message exactly when

$$
m_{s,a}=\mathbf 1[R_{s,a}>0].
$$

This single condition replaces the demo's separate Boolean `m` field.

The node support $n_s$ is maintained incrementally. When an unresolved edge
count changes from $r_{old}$ to $r_{new}$, search applies
$n_s\leftarrow n_s+r_{new}-r_{old}$. Before a count is overwritten by
categorical distance, its final delta is committed to the unresolved parent.
Counts and support are never added to $H$, $C$, $V$, or $Q$.

### 4.1 Effective native action object

Let $s_a$ be the expanded child on edge $(s,a)$ when it exists. The Dirichlet
used for selection and readout is

$$
A_{s,a}=
\begin{cases}
H_{s,a}=B_{s,a},
&z_{s,a}=\bot\text{ and }d_{s,a}>0,\\[3pt]
\operatorname{Align}(V_{s_a};p_{s_a}\to p_s),
&z_{s,a}=\bot\text{ and }d_{s,a}=0\text{ and }s_a\text{ exists},\\[3pt]
H_{s,a}=Q_{s,a},
&z_{s,a}=\bot\text{ and }s_a\text{ does not exist}.
\end{cases}
$$

The precedence matters:

1. a repaired edge message wins;
2. an expanded child without a message contributes its fixed value prior;
3. an unexpanded edge falls back to the node's Q prior.

The expanded-child fallback is deliberately $V_{s_a}$, not the mutable cache
$C_{s_a}$. Once the child has downstream information, the repair rule promotes
its cache into $B_{s,a}$.

The native action object is

$$
Y_{s,a}=
\begin{cases}
\operatorname{Cat}(z_{s,a},d_{s,a}),&z_{s,a}\ne\bot,\\
\operatorname{Dir}(A_{s,a}),&z_{s,a}=\bot.
\end{cases}
$$

Alpha-shaped arrays remain populated for every edge and node because JAX needs
fixed shapes and the unresolved state-cache update needs a learned mass for
its numeric mixture. Categorical utility, propagation, root choice, and neural
loss always decode the outcome tag before interpreting its payload. Once
$z_s\ne\bot$, the node certificate
overrides $C_s$ completely.

When an unresolved node has a mixed set of categorical and Dirichlet edges,
define a cache-only projection

$$
\widetilde A_{s,a}=
\begin{cases}
A_{s,a},&z_{s,a}=\bot,\\[3pt]
\left(\sum_i A_{s,a,i}\right)e_{z_{s,a}},&z_{s,a}\ne\bot.
\end{cases}
$$

Thus the exact outcome tag supplies the direction of a categorical edge, while
its existing effective alpha supplies only the learned concentration. That
mass may come from the Q fallback, an expanded-child prior, or an earlier
repaired message. The projection is computed only as an operand of the
node-cache mixture: it does not mutate $A_{s,a}$, does not replace the tagged
certificate, and is not emitted as a categorical neural target.

### 4.2 Categorical node rules

For legal edges define

$$
\mathcal W_s=\{a:z_{s,a}=W\},\qquad
\mathcal D_s=\{a:z_{s,a}=D\},\qquad
\mathcal L_s=\{a:z_{s,a}=L\},
$$

and the unresolved set

$$
\mathcal U_s=\{a\in\mathcal A(s):z_{s,a}=\bot\}.
$$

Apply these rules in priority order:

1. If $\mathcal W_s\ne\varnothing$, the node is a categorical win. Derive the
   winning edge by minimizing $\tau_{s,a}$ over $\mathcal W_s$, sampling
   uniformly if several edges have the same minimum distance, and store only
   its distance as $\tau_s$; the action itself remains derived.
2. Otherwise, while $\mathcal U_s\ne\varnothing$, the node remains unresolved.
3. If every edge is categorical and $\mathcal D_s\ne\varnothing$, the node is a
   draw. Sample uniformly among $\mathcal D_s$; neither draw distance, policy
   logits, nor action order breaks the tie.
4. Otherwise every legal edge is a loss. The node is a categorical loss and
   derives an action with maximum distance, sampling uniformly among edges at
   that distance.

The resulting $(z_s,\tau_s)$ is absorbing. “Shortest” means shortest among
currently certified winning edges. Immediate absorption cannot prove a
globally shortest forced win through unresolved actions; that stronger claim
would require continuing search or maintaining lower bounds, contrary to the
categorical short-circuit rule.

---

## 5. One native action-selection rule

Associate one utility sample with each native edge object:

$$
X(Y_{s,a})=
\begin{cases}
u_{\mathrm{cat}}(z_{s,a}),
&Y_{s,a}\text{ is categorical},\\
U(\phi_{s,a}),\quad
\phi_{s,a}\sim\operatorname{Dirichlet}(A_{s,a}),
&Y_{s,a}\text{ is Dirichlet}.
\end{cases}
$$

The common selector takes the legal argmax of these utilities. There is no
visit-count selector, UCB score, Gaussian approximation, or separate root
scoring rule.

Traversal restricts this selector to unresolved edges,

$$
\mathcal A_{\mathrm{search}}(s)
=\{a\in\mathcal A(s):z_{s,a}=\bot\},
$$

because a categorical edge needs no more evaluation. Policy estimation at a
noncategorical node instead compares every legal native edge: categorical
utilities are exact, while unresolved Dirichlet utilities are redrawn.
Repeating the primitive $M$ times gives

$$
\widehat\pi_M(a\mid s)
=\frac1M\sum_{j=1}^{M}
\mathbf 1\left[
a=\arg\max_b X(Y_{s,b})^{(j)}
\right].
$$

For $M=1$, this is a one-hot but unbiased Monte Carlo estimate of posterior
optimal-action probability. Larger $M$ reduces readout variance.
`policy_sample_chunk_size` changes only how these draws are batched in memory;
it does not change the distribution being estimated.

---

## 6. A search simulation

One simulation is one sequential

$$
\text{simulate}\;\longrightarrow\;\text{expand}\;\longrightarrow\;
\text{repair bottom-up}
$$

cycle on the persistent tree.

### 6.1 Simulate

Starting at the root:

1. if the root or current node is categorical, emit no active simulation;
2. form $\mathcal A_{\mathrm{search}}(s)$ by removing categorical edges;
3. choose one unresolved action with the Thompson rule in Section 5;
4. follow its child if that edge is expanded and the child's outcome tag is
   unresolved;
5. repeat until the chosen edge is unexpanded or reaches `max_depth`.

The tree is persistent across all simulations in one policy call, so later
simulations see every message and cache repaired by earlier simulations.

### 6.2 Expand

Call `expand_fn` once on the final unresolved parent-action pair. This performs
the environment step and leaf evaluation only for an active simulation. If the
edge is new, initialize the child node with its embedding, $V$, $Q$, player,
exact terminal-outcome tag, and legal-action mask. A terminal result immediately
publishes $\operatorname{Cat}(z,0)$ on the child and its aligned distance-one
edge certificate. A depth-cutoff revisit reuses existing topology and its fresh
uncertain leaf message.

### 6.3 Repair

Begin at the final parent and walk through parent links back to the root. At
every node, repair the unresolved Dirichlet state, decode the tagged payloads,
and apply the categorical rules in Section 4.2. When an edge changes from
unresolved to categorical, its last count delta is first committed to $n_s$;
only then may the same edge payload be overwritten by distance. Likewise, when
a node becomes categorical, its final $n_s$ is propagated to the parent before
the node payload is overwritten by node distance. The repair order is
deepest-first, so one terminal discovery can categorize several ancestors in
the same simulation.

---

## 7. Replaceable posterior-update contract

Search owns traversal, categorical propagation, and storage, but it does not
own the uncertain Dirichlet posterior formula.
At each path node it builds

```python
PosteriorUpdateContext(
    node=NodeView(...),
    children=ChildrenView(...),
    leaf=LeafView(...),
    active=...,
)
```

and invokes

```python
update = posterior_update(rng_key, context)
```

The callback returns a complete

```python
PosteriorUpdate(
    edge_alpha=...,
    edge_payload=...,
    value_alpha=...,
)
```

for that node. The context exposes:

- the current node embedding, fixed $V_s$, mutable $C_s$, node support payload,
  edge alphas/payloads/tags, legality, and player;
- action-indexed child indices, fixed $V_{s_a}$, repaired $C_{s_a}$, tagged
  node payload, player, and categorical outcome;
- the final selected edge and its model value alpha, active only at the deepest
  path node and written as a Dirichlet message only while that edge is
  unresolved;
- categorical state through outcome tags and their overlaid payloads.

Large child embeddings are not copied across the action axis. A custom rule
that needs them can gather the shared embedding table using child indices.

The callback owns the direct unresolved-message write and uncertain
child-to-parent Dirichlet repairs. Search applies count deltas to node support
and owns every count-to-distance transition. Only unresolved positions of the
returned edge alpha and payload are accepted; categorical positions in the
tree remain untouched. Replacing the callback changes posterior mathematics at
every node, but not terminal detection, categorical minimax rules, absorbing
certificates, payload safety, or traversal pruning.

---

## 8. Default posterior repair

The default `update_posterior` is the JAX translation of the message-passing
logic in `tictactoe-demo/app.js`.

### 8.1 Write the final leaf message

Only at the deepest path node $s_d$, an unresolved final action $a_d$
increments its structural count:

$$
R_{s_d,a_d}\leftarrow R_{s_d,a_d}+1.
$$

If the edge remains unresolved, align the model leaf alpha to the node's
perspective and replace the edge message:

$$
B_{s_d,a_d}\leftarrow
\operatorname{Align}(V_{s'};p_{s'}\to p_{s_d}),
$$

This is a replacement, not
$B_{s_d,a_d}\leftarrow B_{s_d,a_d}+L_{s'}$.

If `terminal_outcome` is present, expansion first commits structural support
one to the parent total, then publishes distance one in the same edge payload.
The posterior callback neither increments that payload nor overwrites $B$ with
a terminal alpha. The exact tag owns selection, propagation, readout, and
training.

### 8.2 Refresh messages from repaired children

At the current node $s$, inspect all expanded children. For every unresolved
child $s_a$ whose edge is unresolved and whose downstream
information satisfies $n_{s_a}>0$, replace the parent edge with the child's
repaired cache:

$$
B_{s,a}\leftarrow
\operatorname{Align}(C_{s_a};p_{s_a}\to p_s),
$$

$$
R_{s,a}\leftarrow 1+n_{s_a}.
$$

When a child becomes categorical, search computes the final edge count
$1+n_{s_a}$, commits its delta to the parent total, then overwrites the edge
payload with $1+\tau_{s_a}$. Categorical children remain excluded from
uncertain alpha refresh. A child with no downstream message leaves the alpha
slot on its earlier unresolved message or fixed child-value fallback.

### 8.3 Resolve categorical nodes

After each node's unresolved-posterior repair, apply Section 4.2. A newly
categorical node first publishes its final $1+n_s$ support contribution and
aligned certificate to its parent edge, then overwrites its own node support
with distance. Its certificate is absorbing, and later simulations cannot
enter that node or edge.

### 8.4 Rebuild current action posteriors

After the direct write and child refreshes, recompute all native $Y_{s,a}$
using Section 4.1. Then draw a fresh node-local posterior-best policy:

$$
\pi_s=\widehat\pi_{M_{\mathrm{node}}}(\cdot\mid s;Y_s).
$$

This is a fresh population from the current post-repair native objects: exact
utility for categorical edges and Thompson utility for Dirichlet edges. It is
neither a visit policy nor a historical running average.

### 8.5 Recompute the state cache

Use the incrementally maintained unresolved node support

$$
\gamma_s=\frac{n_s}{\kappa+n_s},
$$

and the policy-weighted action Dirichlet

$$
\bar A_s=\sum_{a\in\mathcal A(s)}\pi_s(a)\widetilde A_{s,a}.
$$

The repaired node cache is

$$
C_s=(1-\gamma_s)V_s+\gamma_s\bar A_s.
$$

If $n_s=0$, then $C_s=V_s$. The sole search constant $\kappa>0$ controls how
quickly a node moves from its fixed value prior toward its repaired
descendants. It is prior strength in this interpolation, not leaf or terminal
evidence.

There is nevertheless a precise depth-decay interpretation for the
**blend-only envelope** of the unresolved numeric-cache channel. Define

$$
g_d=\prod_{i=1}^{d}\gamma_i.
$$

If the structural support were constant, $n_i=n$, then

$$
g_d=\left(\frac{n}{\kappa+n}\right)^d
    =\exp(-d/\ell),
\qquad
\ell(n,\kappa)=\frac{1}{\log(1+\kappa/n)}.
$$

This is not yet the coefficient of one descendant edge. Holding each
node-local policy fixed, a perturbation traveling along selected actions
$a_1,\ldots,a_d$ instead has branch coefficient

$$
w_P^{\mathrm{fixed}}
=\prod_{i=1}^{d}\gamma_i\pi_i(a_i)
\leq g_d.
$$

The policies themselves depend on repaired posteriors, so their Jacobians
contribute to a full perturbation, and a rerun may even change topology. The
logged product $\prod_i\gamma_i$ is therefore a cache-mix attenuation
envelope, not realized information transmission. The effective length
describes only that envelope. It is not a reward horizon: exact terminal and
solved outcomes propagate through categorical sidecars without this
attenuation.

The local sensitivity makes the distinction precise. Let
$\Delta_s=\bar A_s-V_s$. Holding the realized
$n_s,V_s,\bar A_s$ fixed,

$$
C_s=V_s+\gamma_s\Delta_s,
\qquad
\left.
\frac{\partial C_s}{\partial\log\kappa}
\right|_{n_s,V_s,\bar A_s}
=-\gamma_s(1-\gamma_s)\Delta_s.
$$

Consequently,

$$
\left\|
\frac{\partial C_s}{\partial\log\kappa}
\right\|_2
\leq \frac14\|\bar A_s-V_s\|_2.
$$

Because raw Dirichlet alpha distance mixes direction and mass, also let
$c_s=\mathbf 1^\top C_s$, $p_s=C_s/c_s$, and
$G_s=\partial C_s/\partial\log\kappa$. Then the exact local semantic and
relative-concentration sensitivities are

$$
\frac{\partial p_s}{\partial\log\kappa}
=
\frac{G_s-p_s(\mathbf 1^\top G_s)}{c_s},
\qquad
\frac{\partial\log c_s}{\partial\log\kappa}
=
\frac{\mathbf 1^\top G_s}{c_s}.
$$

A large change in $\ell$ can therefore have little local effect for two
different reasons: the mixture is saturated ($n_s\ll\kappa$ or
$n_s\gg\kappa$), or the learned prior already agrees with the repaired
aggregate ($\bar A_s\approx V_s$). The latter is exactly the fixed point
encouraged by search-based self-distillation: if $\bar A_s=V_s$, then
$C_s=V_s$ for every $\kappa$. This can indicate successful assimilation, but
not correctness, because prior and search may share the same bias.

For a heterogeneous unresolved numeric path $P$, holding its supports fixed,

$$
-\log g_P
=\sum_{i\in P}-\log\gamma_i
=\sum_{i\in P}\log\left(1+\frac{\kappa}{n_i}\right),
\qquad
\frac{\partial\log g_P}{\partial\log\kappa}
=-\sum_{i\in P}(1-\gamma_i).
$$

Thus the observed gamma product and its log attenuation are a more faithful
blend-only envelope than a root-local constant-support $\ell$. Even a cache
change must then reach a root edge posterior and cross a commitment boundary.
Paired reruns, rather than this envelope, measure the deployed effect. If
$q_\kappa$ has unique top-two margin
$m_\kappa=q_{\kappa,(1)}-q_{\kappa,(2)}$, a deterministic argmax can change
only if

$$
\|q_{\kappa'}-q_\kappa\|_\infty\geq \frac{m_\kappa}{2}.
$$

For plurality commitment the margin remains diagnostic, but this hard
condition does not apply because the finite vote is random. The implementation
also reports the fraction of policies whose top-two margin is at most $1/M$.
That one-vote spacing is a descriptive reference scale for an $M$-sample
empirical policy, not a decision threshold for argmax, plurality, or posterior
sampling. Finally, an action flip need not change strength: it may select an
outcome-equivalent move, occur on a rarely visited state, or be irrelevant
against the chosen opponent. The direct final-edge message is also installed
before the cache mix, and completed root action readout ranks root edge
posteriors rather than the root's mixed value cache.

This is a convex interpolation of Dirichlet parameter vectors. It is the
algorithm's message-passing rule; it is not a conjugate update with $R$
categorical observations. Individual $R$ values cease to exist when their
edge payloads become distances.

For a categorical edge, $\widetilde A_{s,a}$ is the learned-mass projection
from Section 4.1; the outcome tag still supplies policy utility and exact
semantics.
For a categorical node, $C_s$ is only a numeric cache and its native certificate
is the value. The categorical neural target never consumes this alpha.

### 8.6 Why bottom-up order is essential

For a path

$$
s_0\to s_1\to\cdots\to s_d,
$$

the repair order is

$$
s_d,s_{d-1},\ldots,s_0.
$$

Therefore the write at $s_i$ may use the newly computed $C_{s_{i+1}}$ rather
than a stale cache. This recursively transports information through the whole
path while keeping each callback local to one node and its children.

---

## 9. Completed-root readouts

All active simulations finish before any configurable root readout is applied.
First compute the native root objects

$$
Y_a^{\mathrm{root}}=Y_{s_0,a}
$$

and the backend's native winner-count policy. At an unresolved root this is

$$
\pi_M^{\mathrm{nat}}(a\mid s_0)
=\widehat\pi_M(a\mid s_0;Y^{\mathrm{root}}),
\qquad M=M_{\mathrm{root}}.
$$

The standard configuration has $M=32$, hence the shorthand **M32**. At a
categorical root, let $\mathcal B(s_0)$ be the distance-optimal certified tie
set: shortest certified wins, longest certified losses, or every certified
draw. The backend draws

$$
A_{\mathrm{cat}}\sim\operatorname{Uniform}(\mathcal B(s_0)),
\qquad
\pi_M^{\mathrm{nat}}=e_{A_{\mathrm{cat}}}.
$$

This draw uses the search RNG, not policy logits or action order. The backend's
`PolicyOutput` always contains this native policy, its native action, and the
completed tree. Optional target and action readouts are computed afterwards;
they do not rerun or mutate traversal, expansion, repair, proof propagation,
or the tree.

$M_{\mathrm{node}}$ (`posterior_policy_samples`) and $M_{\mathrm{root}}$
(`policy_samples`) estimate the same posterior-best population but may use
different Monte Carlo budgets. The internal budget affects stochastic cache
repair. The root budget affects only the native completed-root readout.

### 9.1 Guarded binary prefix-CDF readout

For an unresolved binary root, write $x_a=\phi_{a,W}$ for an unresolved
action's Beta-distributed win probability. Because
$U(\phi_a)=2x_a-1$, the exact posterior-best population is

$$
q_a^\star
=\Pr(x_a=\max_b x_b)
=\int_0^1 f_a(x)\prod_{b\ne a}F_b(x)\,dx.
$$

A certified loss has probability zero while any unresolved action remains; a
certified win would already have made the root categorical. The optional
prefix-CDF estimator approximates all these integrals together on an adaptive
sinh-logit grid with

$$
Q=2h+1
$$

points; Q21 is $h=10$. It trapezoidally integrates each transformed Beta
density, forms every $F_a$ by a prefix sum, and allocates each increment of the
maximum CDF $\Delta\prod_aF_a$ among actions in proportion to their local
nonnegative winner contributions. The allocation is permutation equivariant,
costs $O(AQ)$, and conserves total policy mass up to floating-point roundoff.

One symmetric grid half-range is chosen per root:

$$
T=\operatorname{clip}\left(
\operatorname{asinh}\!\left(
\frac{\texttt{tail\_scale}}
{\min_{a:\,z_{s_0,a}=\bot,\ z}\alpha_{a,z}}
\right),
\ T_{\min},T_{\max}
\right).
$$

The estimator is accepted for an unresolved root only if all three numerical
guards pass:

1. the requested adaptive range did not exceed $T_{\max}$;
2. every output is finite;
3. the maximum absolute pre-normalization log density-integral error is at
   most $0.01$.

If any guard fails, that root falls back to its already computed
$\pi_M^{\mathrm{nat}}$. This is a per-root fallback: one unsafe batch lane does
not discard safe estimates in other lanes. Prefix-CDF root readout is defined
only for two-outcome heads.

### 9.2 Independent target and commitment estimators

Two configuration fields select what consumes the completed root:

- `root_policy_target_estimator` selects the replay policy target;
- `root_action_estimator` selects the ephemeral policy used by
  `posterior_argmax`, `posterior_plurality`, or `posterior_sample`.

Each is independently either `winner_mc` or `prefix_cdf`. If either requests
`prefix_cdf`, the quadrature is evaluated once and the safe result is shared.
Both root fields default to `winner_mc`. They are distinct from
`posterior_policy_estimator`, which selects the node-local population used
inside bottom-up cache repair while the tree is being built.
Let $\widetilde q_Q$ denote that guarded result and define the exact categorical
tie population

$$
\pi^{\mathrm{cat}}(a\mid s_0)
=\frac{\mathbf 1[a\in\mathcal B(s_0)]}{|\mathcal B(s_0)|}.
$$

The training target is

$$
\pi_{\mathrm{tgt}}=
\begin{cases}
\pi_M^{\mathrm{nat}},
&\texttt{root\_policy\_target\_estimator}=\texttt{winner\_mc},\\
\widetilde q_Q,
&\texttt{prefix\_cdf},\ z_{s_0}=\bot,\text{ and guards pass},\\
\pi_M^{\mathrm{nat}},
&\texttt{prefix\_cdf},\ z_{s_0}=\bot,\text{ and a guard fails},\\
\pi^{\mathrm{cat}},
&\texttt{prefix\_cdf},\ z_{s_0}\ne\bot.
\end{cases}
$$

Thus a solved prefix target is the exact population of the native random
distance-optimal tie-break, rather than one particular one-hot draw. This
removes tie RNG from the label without introducing a first-index preference.

The ephemeral commitment policy is

$$
\pi_{\mathrm{commit}}=
\begin{cases}
\widetilde q_Q,
&\texttt{root\_action\_estimator}=\texttt{prefix\_cdf},\
z_{s_0}=\bot,\text{ and guards pass},\\
\pi_M^{\mathrm{nat}},&\text{otherwise}.
\end{cases}
$$

In particular, solved commitment remains the native random member of
$\mathcal B(s_0)$ under both estimator settings. The commitment policy is not
stored in replay.

### 9.3 Committed self-play action

Search output and game action remain separate. Scacchi commits by:

- `posterior_argmax`: $\arg\max_a\pi_{\mathrm{commit}}(a)$;
- `posterior_plurality`: the lowest-index plurality of
  `posterior_plurality_samples` categorical draws from
  $\pi_{\mathrm{commit}}$;
- `posterior_sample`: a categorical draw from the power-temperature law
  $\tau_T(\pi_{\mathrm{commit}})$, where
  $T=$ `posterior_sample_temperature`;
- `search_action`: the native backend action.

Consequently `root_action_estimator` can change the first three modes only at an
unresolved root. It never changes `search_action`, and it cannot change the
tree that produced the readout. It can, of course, change the subsequent game
trajectory once a different action is played. The temperature is likewise an
ephemeral commitment parameter: it does not affect traversal, repair, root
target construction, Q/V targets, replay, or the weights used by the current
search.

### 9.4 Finite-population plurality commitment

There is a precise intermediate transform between one draw from a posterior-
best population and its deterministic mode. Index the $L$ legal actions by
$0,\ldots,L-1$. For $q\in\Delta^{L-1}$, draw

$$
C\sim\operatorname{Multinomial}(M,q),
\qquad
A_M=\min\arg\max_a C_a,
$$

where the minimum makes the production `argmax` convention explicit. Define
$g_M(q)$ as the distribution of $A_M$. Its exact coordinates are

$$
\begin{aligned}
\mathcal C_{M,a}
&=\left\{c\in\mathbb N^L:
\sum_i c_i=M,\quad
c_a>c_j\ (j<a),\
c_a\ge c_j\ (j>a)\right\},\\
[g_M(q)]_a
&=\sum_{c\in\mathcal C_{M,a}}
\frac{M!}{\prod_i c_i!}\prod_i q_i^{c_i}.
\end{aligned}
$$

Thus $g_1(q)=q$. If $q$ has a unique mode $a^\star$, the law of large
numbers gives $g_M(q)\to e_{a^\star}$ as $M\to\infty$. At finite $M$ this is
not a temperature transform and need not be coordinatewise monotone in $M$:
multinomial count ties retain the stated lowest-index bias.

`posterior_plurality` uses $M=$ `posterior_plurality_samples`, which defaults
to 32. Before sampling, it zeros illegal, non-finite, and non-positive
entries and renormalizes the remaining legal mass. Zero remaining legal mass
falls back to the uniform legal population; the degenerate no-legal-action
sentinel is action zero. An exact one-hot population bypasses resampling.

This operator exactly describes native winner-MC plus
`posterior_argmax`, conditional on one fixed, completed, unresolved tree
$T$. Let $q^{\mathrm{impl}}(T)$ be the action distribution induced by one
production Thompson-winner draw. The $M$ native winner indices are i.i.d.
categorical draws from $q^{\mathrm{impl}}(T)$, their counts are
$C=M\pi_M^{\mathrm{nat}}$, and therefore

$$
\mathcal L\!\left(
\min\arg\max_a\pi_M^{\mathrm{nat}}(a)\mid T
\right)
=g_M\!\left(q^{\mathrm{impl}}(T)\right).
$$

In particular, sampling 32 categorical winners from an accepted Q21
population $\widetilde q_{21}(T)$ and committing their plurality has
conditional law $g_{32}(\widetilde q_{21}(T))$. It is distributionally
identical to native M32 plus `posterior_argmax` only when
$\widetilde q_{21}(T)=q^{\mathrm{impl}}(T)$ and the count-tie convention is
the same. In general Q21 has deterministic quadrature error relative to the
ideal exact-Beta population $q^\star(T)$, while the production Gamma
primitive's bounded-work fallback can make
$q^{\mathrm{impl}}(T)\ne q^\star(T)$.

The equivalence concerns an accepted Q21 lane. If its guard fails,
`posterior_plurality` bypasses the new vote draw and returns the lowest-index
argmax of the already realized native histogram, preserving native fallback
semantics. By contrast, explicitly supplying a winner-MC histogram to a
$K$-vote plurality readout is a two-stage experiment: conditional on the
histogram its law is $g_K(\pi_M^{\mathrm{nat}})$, and marginally it is
$\mathbb E[g_K(\pi_M^{\mathrm{nat}})\mid T]$, not
$g_M(q^{\mathrm{impl}})$ in general.

For Q21, the transform applies only to an accepted unresolved-root population;
an explicitly selected winner-MC commitment population has the two-stage
semantics above. A categorical root continues to sample uniformly from its
distance-optimal certified tie set; it must not be replaced by the
lowest-index count rule. The identity above is also only conditional on the
same completed tree. A different committed action changes the next state and
hence later trees and trajectories; training with a different behavior
distribution can additionally change the weights and all future trees. It
therefore establishes matching root semantics, not matching whole games or
learning runs.

This distinction explains the two behavioral extremes. A direct categorical
draw from $q$ is exactly the $M=1$ endpoint: it treats posterior uncertainty
about which action is best as a behavior policy, so a high-entropy
posterior-best population can be too diffuse for play. Direct
`posterior_argmax` is the opposite endpoint: it discards every probability
except the mode, is discontinuous at a tie, and can repeatedly amplify a
small model or quadrature asymmetry into collapsed coverage. Finite $g_M$
aggregates repeated winner draws while retaining controlled commitment
randomness. Away from close count ties its unique-mode concentration
suppresses low-mass alternatives; the formula keeps the finite tie bias
visible. $M=32$ is distinguished because it reproduces the existing native
M32 commitment transform on a fixed tree, not because 32 is universally
optimal.

### 9.5 Power-temperature posterior sampling

Let $\mathcal A_{\mathrm{legal}}\ne\varnothing$ be the legal action set and
put $\epsilon=10^{-8}$. For $T>0$, define

$$
\begin{aligned}
\bar q_a
&=\mathbf 1[a\in\mathcal A_{\mathrm{legal}}]\,
  \operatorname{clip}(q_a,\epsilon,1),\\
[\tau_T(q)]_a
&=\frac{\bar q_a^{1/T}}
{\sum_{b\in\mathcal A_{\mathrm{legal}}}\bar q_b^{1/T}},
\qquad
A_T\sim\operatorname{Categorical}(\tau_T(q)).
\end{aligned}
$$

This is exactly the `posterior_sample` commitment law. At $T=1$ it is the
pre-existing direct categorical sampler, including its numerical floor; when
the legal coordinates of $q$ are already normalized and at least $\epsilon$,
$\tau_1(q)=q$. If the floored legal population has a unique maximizer
$a^\star$, then

$$
\lim_{T\downarrow0}\tau_T(q)=e_{a^\star}.
$$

Thus $0<T<1$ sharpens and $T>1$ flattens without changing the completed search
or its targets. For any legal $a,b$,

$$
\frac{[\tau_T(q)]_a}{[\tau_T(q)]_b}
=
\left(\frac{\operatorname{clip}(q_a,\epsilon,1)}
{\operatorname{clip}(q_b,\epsilon,1)}\right)^{1/T}.
$$

The transform therefore preserves the weak ranking induced by the clipped
coordinates; a strict ordering remains strict unless the floor collapses both
coordinates to the same value. Its pairwise odds are independent of every
other coordinate, so it satisfies the Luce independence-of-irrelevant-
alternatives property for a fixed legal set.

Write $\Phi_M=g_M$ for the finite-vote plurality law in Section 9.4.
In a multiclass action space, $\tau_T$ is not $\Phi_M$ for any universal
choice of $T$. The production lowest-index count-tie rule makes the
difference explicit. For every finite $M\ge2$ and $L\ge3$, a full-support
symmetric population assigns positive probability to a maximum-count tie.
Resolving that event by the lowest index breaks permutation symmetry; by
continuity, nearby inputs can rank a higher-index action slightly above a
lower-index action while $\Phi_M$ still ranks the lower index above it. For
the concrete three-action case $M=2$,

$$
[\Phi_2(q)]_0=2q_0-q_0^2,
\qquad
[\Phi_2(q)]_1=q_1^2+2q_1q_2.
$$

At $q=(0.34,0.35,0.31)$, action 1 has the larger input probability but
$[\Phi_2(q)]_0=0.5644>0.3395=[\Phi_2(q)]_1$: the finite count tie-break reverses
their ranking in the output law. Moreover, along
$q=(x,x,1-2x)$,

$$
\frac{[\Phi_2(q)]_0}{[\Phi_2(q)]_1}
=\frac{2-x}{2-3x},
$$

which varies with the third coordinate even though $q_0/q_1=1$. Hence this
plurality law is not IIA. It also carries a literal action-index bias whenever
a positive-probability maximum-count tie is resolved by the lowest index.
These structural properties rule out an exact fixed-temperature
law-equivalence in the multiclass case, apart from trivial or isolated cases
such as the unfloored $M=1,T=1$ endpoint.

Temperature sampling can therefore replace plurality as a simpler
one-parameter causal intervention: with the tree, commitment population, and
RNG protocol held fixed, only the final action law changes. It is not an exact
law-equivalent substitution for $\Phi_M$. Matching one summary such as
entropy, top-action probability, or game strength does not make the two
multiclass distributions identical, and a temperature fitted at one root does
not become a distributional identity at other roots.

An unsafe prefix-CDF action readout, or a solved root whose action readout
requested prefix-CDF, bypasses both plurality and temperature resampling and
returns the native commitment argmax. This preserves the already realized
native histogram fallback and the native random distance-optimal solved action
exactly.

---

## 10. Simulations, capacity, and depth

`num_simulations = N` is a static upper budget of $N$ loop iterations and at
most $N$ active simulate-expand-repair cycles. Once a root is categorical its
later lane iterations are inactive no-ops; if every batch lane is inactive,
the recurrent evaluator and backup are skipped.

For JAX static shapes, the tree allocates capacity

$$
N_{\mathrm{slots}}=N+1
$$

including the root. Simulation $i$ reserves slot $i+1$ for a possible new
node. It initializes that slot only if the selected unresolved edge is
unexpanded. Categorical and terminal edges are never revisited. Stopping on an
already expanded unresolved edge at `max_depth` performs leaf evaluation and
bottom-up repair but creates no node. Consequently,

$$
N_{\mathrm{real\ nodes}}\le N+1.
$$

The fixed array size is a capacity bound, not an instruction to populate every
slot. Multiple simulations on one tree are precisely what allow later
traversals to exploit and repair information created by earlier ones.

`max_depth` independently bounds how many edges a single traversal may follow.
When omitted, it defaults to `num_simulations`. There is no `num_blocks` search
parameter in this algorithm; network depth remains a separate model choice.

---

## 11. Native training targets emitted by search

The semantic Q and V targets are tagged native objects:

$$
T^Q_{s,a}=
\begin{cases}
\operatorname{Pad},&a\text{ is illegal},\\
\operatorname{Cat}(z_{s,a},\tau_{s,a}),&z_{s,a}\ne\bot,\\
\operatorname{Dir}(A_{s,a}),&z_{s,a}=\bot,
\end{cases}
$$

$$
T^V_s=
\begin{cases}
\operatorname{Cat}(z_s,\tau_s),&z_s\ne\bot,\\
\operatorname{Dir}(C_s),&z_s=\bot.
\end{cases}
$$

The policy target $\pi_{\mathrm{tgt}}$ is the independently selected
completed-root target from Section 9.2. It need not equal the policy used for
action commitment.

The fixed-shape replay record carries eight native-target metadata fields,
`q_target_kind/weight/outcome/distance` and
`v_target_kind/weight/outcome/distance`. Unresolved alpha arrays remain in
`tree.summary().alpha` and `tree.summary().value_alpha`; the learner ignores
their entries whenever the corresponding kind is categorical or padded.
`tree.summary().visit_counts` decodes the root edge payload as $R$ only for an
unresolved edge and returns zero for a categorical edge, whose payload now
means distance. Categorical distances are decoded separately.

There is no reconstruction

$$
\beta_Q=\alpha_{\mathrm{base}}+E_Q
$$

and no addition of $R$ to Dirichlet concentration. Doing either would train
the network on a posterior different from the one search actually used.

The Q-loss reduction weight is configurable:

$$
w_a^{\mathrm{reduce}}=
\begin{cases}
R_{s_0,a}, & z_{s_0,a}=\bot\text{ and evidence-mass mode},\\
\pi_M^{\mathrm{nat}}(a\mid s_0),
& z_{s_0,a}=\bot\text{ and policy mode},\\
1,&z_{s_0,a}\ne\bot.
\end{cases}
$$

Here the name `evidence_mass` is historical: in this backend it receives the
structural edge count $R$ only while that edge is unresolved, not added
Dirichlet pseudo-count mass. Every legal categorical Q edge is given unit
effective weight even if a
one-hot solved policy assigns it zero probability, so exact alternative moves
are not silently discarded.

This reduction weight is distinct from replay's `q_target_weight`. The latter
is a native-target validity/scale field and is one for every legal emitted Q
row (zero for padding). The former is stored as `q_loss_weight` and determines
how legal Q rows are combined by the configured Q-loss reduction.

The policy-mode Q reduction deliberately remains native. Changing either root
estimator changes neither $R$, the Q/V native targets, nor
$w_a^{\mathrm{reduce}}$. This prevents a lower-variance policy label or an
ephemeral action policy from silently reweighting a separate learning
objective.

### 11.1 Policy loss

The policy head is trained by cross-entropy on legal actions:

$$
\mathcal L_\pi(s)=
-\sum_{a\in\mathcal A(s)}
\operatorname{stopgrad}(\pi_{\mathrm{tgt}}(a\mid s))
\log\pi_\theta(a\mid s).
$$

For legal-action logits $r$ and $p_\theta=\operatorname{softmax}(r)$,

$$
\nabla_r\mathcal L_\pi(q)=p_\theta-q.
$$

Therefore replacing the exact population $q$ by an estimate $\widetilde q$
produces exactly the same error in policy-logit gradient coordinates:

$$
\nabla_r\mathcal L_\pi(\widetilde q)
-\nabla_r\mathcal L_\pi(q)
=q-\widetilde q,
\qquad
\left\|
\Delta\nabla_r\mathcal L_\pi
\right\|_2^2
=\|\widetilde q-q\|_2^2.
$$

For an unresolved winner-count target based on $M$ independent native draws,
let $q^{\mathrm{impl}}$ be the categorical distribution induced by the
production sampling primitive. Then, conditionally on the completed tree,

$$
M\widehat q_M
\sim\operatorname{Multinomial}(M,q^{\mathrm{impl}}),
\qquad
\mathbb E[\widehat q_M]=q^{\mathrm{impl}},
\qquad
\operatorname{Cov}(\widehat q_M)
=\frac{
\operatorname{Diag}(q^{\mathrm{impl}})
-q^{\mathrm{impl}}(q^{\mathrm{impl}})^\top
}{M},
$$

and hence

$$
\mathbb E\left[
\|\widehat q_M-q^{\mathrm{impl}}\|_2^2
\right]
=\frac{1-\|q^{\mathrm{impl}}\|_2^2}{M}.
$$

This is simultaneously the finite-$M$ target MSE and expected squared
policy-logit-gradient error relative to the implemented population. The
production Gamma sampler has a vanishingly rare bounded-work fallback, so
$q^{\mathrm{impl}}$ and the ideal exact-Beta population $q^\star$ must remain
conceptually distinct. Against $q^\star$,

$$
\mathbb E\left[
\|\widehat q_M-q^\star\|_2^2
\right]
=\frac{1-\|q^{\mathrm{impl}}\|_2^2}{M}
+\|q^{\mathrm{impl}}-q^\star\|_2^2.
$$

Prefix quadrature replaces the finite-$M$ sampling variance with deterministic
discretization bias; it must therefore be validated against a higher-precision
exact-Beta population reference rather than assumed exact.

### 11.2 Measuring information transfer

For a fixed policy target $q$, cross-entropy separates as

$$
\operatorname{CE}(q,p_\theta)
=H(q)+D_{\mathrm{KL}}(q\|p_\theta).
$$

Thus the logged policy displacement

$$
D_0=D_{\mathrm{KL}}(q\|p_{\theta,0})
$$

is the target-prior discrepancy presented by search, in nats. It is useful
supervision only to the extent that optimization captures it. Holding the
same target, legal mask, and probe population fixed across an update window,
define

$$
D_{\mathrm{before}}
=D_{\mathrm{KL}}(q\|p_{\theta,\mathrm{before}}),
\qquad
D_{\mathrm{after}}
=D_{\mathrm{KL}}(q\|p_{\theta,\mathrm{after}}),
$$

$$
\Delta_{\mathrm{capture}}
=D_{\mathrm{before}}-D_{\mathrm{after}},
\qquad
f_{\mathrm{capture}}
=\frac{\Delta_{\mathrm{capture}}}{D_{\mathrm{before}}}.
$$

Positive capture means the weights moved toward that fixed search target;
zero means no measured assimilation; a negative value means the fixed-probe
gap grew. The raw before/after gaps and their difference should always be
reported with the fraction, because the ratio is unstable when the initial
gap is tiny. The fraction is undefined if the before/after populations or
counts differ.

The same fixed-probe construction is logged for V and Q semantic KL, for full
Dirichlet KL on unresolved targets, and for Q under the actual reduction
weights. Root-readout diagnostics additionally report prefix
eligibility/acceptance/fallback, tail clipping, density-integral error,
non-finite output, target or commitment L1 and squared-L2 distance from native
M32, and top-one agreement. Target entropy, support, and effective sample size
show whether a target has collapsed even when its loss is small.

A flat arena response should be localized through the full causal funnel.
At numeric repairs, report raw innovation $\|\bar A-V\|_2$, normalized
semantic innovation, concentration innovation, and the raw-alpha,
normalized-mean, and relative-concentration derivatives with respect to
$\log\kappa$. Along completed unresolved numeric paths, report the
blend-envelope quantities $\prod_i\gamma_i$ and
$\sum_i-\log\gamma_i$, together with
categorical and solved bypass rates. At the root, report paired policy
distance, top-two margin, top-action flips, and occupancy-weighted
exact-oracle regret. Means should be accompanied by stage/support strata and
upper quantiles so rare decisive states are not averaged away. Only after
these measurements should seat-balanced pairwise games be summarized; one
opponent or a one-dimensional Elo projection cannot identify a global
strength effect in a non-transitive league.

These quantities have strict interpretation limits:

- search displacement is not Bayesian information gain or mutual information;
  search and its target depend on the current network and reused tree state;
- target-versus-M32 disagreement measures a readout change, not accuracy,
  because M32 itself is noisy;
- capture measures fit to one fixed probe over one update window, not
  retention on later data or causal credit among simultaneous objectives;
- lower target MSE or higher capture does not by itself imply stronger play.
  Arena results, seat-stratified error, and exact-oracle regret remain separate
  outcome measurements.

### 11.3 Typed Dirichlet-head loss

In `full` mode, a search target $\beta$ trains both mean and concentration:

$$
\mathcal L_{\mathrm{full}}(\beta,\alpha_\theta)=
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}(\operatorname{stopgrad}\beta)
\,\|\,
\operatorname{Dirichlet}(\alpha_\theta)
\right).
$$

In `mean` mode, only the categorical means are matched:

$$
\mathcal L_{\mathrm{mean}}(\beta,\alpha_\theta)=
D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}\mu(\beta)
\,\|\,
\mu(\alpha_\theta)
\right).
$$

For a categorical outcome $z$ in a $K$-outcome space, define the
epsilon-interior point

$$
\phi_z^\epsilon=(1-K\epsilon)e_z+\epsilon\mathbf 1,
\qquad
0<\epsilon<\frac1K.
$$

Its target coordinate is $1-(K-1)\epsilon$ and every other coordinate is
$\epsilon$. This is `training.losses.categorical_epsilon`; search-time exact
tags need no smoothing or message floor. The native categorical loss is the
negative log density of the predicted Dirichlet at this stop-gradient point:

$$
\begin{aligned}
\mathcal L_{\mathrm{cat}}(\alpha_\theta,z)
&=-\log p_{\operatorname{Dirichlet}(\alpha_\theta)}(\phi_z^\epsilon)\\
&=-\log\Gamma(\alpha_0)
+\sum_i\log\Gamma(\alpha_i)
-\sum_i(\alpha_i-1)\log\phi_{z,i}^\epsilon.
\end{aligned}
$$

The complete typed loss is

$$
\mathcal L_{\mathrm{native}}(T,\alpha_\theta)=
\begin{cases}
\mathcal L_{\mathrm{full}}(\beta,\alpha_\theta),
&T=\operatorname{Dir}(\beta),\ \texttt{full},\\
\mathcal L_{\mathrm{mean}}(\beta,\alpha_\theta),
&T=\operatorname{Dir}(\beta),\ \texttt{mean},\\
\mathcal L_{\mathrm{cat}}(\alpha_\theta,z),
&T=\operatorname{Cat}(z,\tau)\text{ in either mode},\\
0,&T=\operatorname{Pad}.
\end{cases}
$$

No target Dirichlet is constructed for a categorical row. Distance controls
proof propagation and action choice but not this neural NLL. As a continuous
density NLL, the loss may legitimately be negative. Epsilon prevents boundary
logs; the finite concentration ceiling from Section 2 makes the optimization
well posed. For numerical defense, the implementation evaluates the formula at
$\max(\alpha_{\theta,i},10^{-6})$ componentwise; this is inert for the positive,
floored head outputs used in normal training.

The value loss applies this to $(T^V,\alpha_\theta^V)$. The Q loss applies it to
$(T^Q(a),\alpha_\theta^Q(a))$ using `q_target_weight` to validate/scale each
native target, then reduces legal action rows using
$w_a^{\mathrm{reduce}}$, the active row mask, and the configured reduction
mode.

### 11.4 Outcome supervision

Scacchi may additionally supervise the eventual game outcome $z$ through the
means of the value head and the played-action Q head:

$$
\mathcal L_{V,\mathrm{out}}=-\log\mu(\alpha_\theta^V)_z,
$$

$$
\mathcal L_{Q,\mathrm{out}}=-\log
\mu(\alpha_\theta^Q(s,a_{\mathrm{played}}))_z.
$$

Optional terminal-edge and terminal-parent handling may additionally mark an
exact replay target categorical. The auxiliary mean-outcome term is masked
where the corresponding native target is already categorical, so that head's
supervision is simply the density NLL rather than duplicated categorical loss.

The configured Dirichlet training objective is

$$
\mathcal L=
\lambda_\pi\mathcal L_\pi
+\lambda_V\mathcal L_V
+\lambda_Q\mathcal L_Q
+\lambda_{V,z}\mathcal L_{V,\mathrm{out}}
+\lambda_{Q,z}\mathcal L_{Q,\mathrm{out}}.
$$

---

## 12. Complete algorithm

```text
DIRICHLET-THOMPSON-POLICY(root, N, posterior_update):
    tree <- fixed-capacity tree with N + 1 slots
    initialize root alphas and cache C_root <- V_root
    initialize root outcome tag from root.terminal_outcome
    initialize every other outcome tag to unresolved
    initialize all integer payloads to zero

    repeat N times:
        if root is categorical:
            continue  # inactive static-loop iteration
        node <- root

        # simulate
        repeatedly sample one Dirichlet per unresolved legal edge
        and follow the sampled-utility argmax over unresolved edges only
        stop at an unexpanded unresolved edge or max_depth

        # expand
        step <- expand_fn(node.embedding, selected_action)
        initialize the child only when the edge was unexpanded
        if step.terminal_outcome != NO_OUTCOME:
            child tag/payload <- (terminal outcome, distance 0)
            parent payload (n_down) <- n_down + 1 - old_edge_count
            parent edge tag/payload <- (aligned outcome, distance 1)

        # repair
        for node on the selected path, deepest to root:
            context <- (node, child summaries, tagged payloads, final leaf)
            update <- posterior_update(key, context)
            apply unresolved edge-count deltas to node payload (n_down)
            try categorical win/draw/loss rules at node
            sample uniformly among tied certified actions
            if node became categorical and has a parent:
                commit final 1 + n_down delta to parent payload
                overwrite incoming edge count with distance + 1
                overwrite node payload n_down with node distance

    if root is categorical:
        native_action <- sample uniformly among tied
                         distance-optimal certified edges
        native_policy <- one_hot(native_action)
    else:
        root_native <- categorical edges plus effective Dirichlet edges
        native_policy <- M repeated native utility samples from root_native
        native_action <- argmax(native_policy)
    return native_action, native_policy, tree
```

For the default callback, `posterior_update` is:

```text
DEFAULT-POSTERIOR-UPDATE(node, children, leaf):
    if this is the deepest path node:
        if leaf edge is unresolved:
            B[leaf.action] <- aligned model leaf Dirichlet
            R[leaf.action] <- R[leaf.action] + 1

    for each expanded, unresolved child with child.n_down > 0:
        B[action] <- aligned child.cache
        R[action] <- 1 + child.n_down

    A <- effective(B, R, expanded child V priors, Q fallbacks)
    Y <- Cat(outcome, payload distance) when tagged, otherwise Dir(A)
    pi <- fresh repeated-native policy from Y
    n_down <- previous n_down + sum of unresolved R deltas
    A_tilde <- A for unresolved edges
               or sum(A) * one_hot(exact outcome) for categorical edges
    gamma <- n_down / (kappa + n_down)
    cache <- (1 - gamma) * V_prior
             + gamma * sum_action pi[action] * A_tilde[action]
    return ephemeral update(
        edge_alpha=B,
        edge_payload=R on unresolved edges; categorical entries preserved,
        value_alpha=cache,
    )
```

The Scacchi wrapper then performs the independent readouts without changing
the completed tree:

```text
ROOT-READOUTS(native_action, native_policy, tree, target_kind, action_kind):
    if target_kind == prefix_cdf or action_kind == prefix_cdf:
        prefix, safe <- one guarded Q(2h + 1) binary prefix-CDF evaluation

    if target_kind == winner_mc:
        target_policy <- native_policy
    else if root is categorical:
        target_policy <- uniform population over distance-optimal ties
    else if safe:
        target_policy <- prefix
    else:
        target_policy <- native_policy  # this root only

    if action_kind == prefix_cdf and root is unresolved and safe:
        commitment_policy <- prefix
    else:
        commitment_policy <- native_policy
    resampling_bypass <- action_kind == prefix_cdf
                           and (root is categorical or not safe)

    q_loss_weight <- structural counts or native_policy, never target_policy
    if commitment_mode == search_action:
        played_action <- native_action
    else if commitment_mode == posterior_argmax:
        played_action <- argmax(commitment_policy)
    else if commitment_mode in {posterior_plurality, posterior_sample}
            and resampling_bypass:
        played_action <- argmax(native_policy)
    else if commitment_mode == posterior_plurality:
        played_action <- plurality_sample(commitment_policy, M)
    else if commitment_mode == posterior_sample:
        played_action <- categorical_sample(
            power_temperature(commitment_policy, T, epsilon=1e-8)
        )
    return target_policy, played_action
```

---

## 13. Core invariants

A correct implementation maintains all of the following:

1. Every Dirichlet and categorical outcome uses its node's player perspective.
2. A terminal child discovered by expansion has distance zero; its parent edge
   adds one ply. A terminal root supplies an exact outcome payload.
3. Categorical certificates are exact, native, and absorbing.
4. One certified winning edge solves a node using the shortest certified win.
5. Draw and loss require every legal edge to be categorical.
6. Loss delays defeat maximally; draw uses the configured draw-only rule.
7. Categorical edges and nodes are never traversed again.
8. Mixed readout compares exact categorical utility with sampled Dirichlet
   utility through the same argmax selector.
9. $A_{s,a}$ follows message, child-value prior, then Q-fallback precedence for
   unresolved Dirichlet objects and the learned mass used by cache projection.
10. $B$ is replaced, not treated as an additive evidence sum.
11. While an edge is unresolved, $R$ controls structure and prior mixing; it
    is not Dirichlet concentration. Its final delta reaches the parent before
    the shared payload becomes categorical distance.
12. The outcome tag is inspected before either shared payload is read: an edge
    payload means $R$ or edge distance, and a node payload means $n_s$ or node
    distance, never both at once.
13. Root summary counts are zero for categorical edges because their payloads
    no longer contain counts.
14. Numeric alphas, cache projections, and caches never override native
    certificates or become categorical targets.
15. Categorical ties use the search RNG to sample uniformly among equally good
    certified actions; they never trigger a policy reevaluation.
16. Categorical V/Q targets use density NLL in both Dirichlet loss modes;
    unresolved targets retain the configured full/mean behavior.
17. Terminal means categorical tag plus node payload zero; zero support alone
    is not terminal.
18. `num_simulations` and $N+1$ are static iteration/capacity bounds;
    categorical short-circuiting may use fewer active cycles and real nodes.
19. The completed tree, native winner-MC policy, and native backend action
    exist before either configurable root readout and are never mutated by it.
20. Root target and unresolved-root commitment estimators are independently
    selectable. If both request prefix-CDF, one guarded quadrature result is
    shared.
21. An unsafe prefix-CDF result falls back to native winner-MC for that root;
    safe batch lanes remain usable.
22. A solved prefix target is the exact uniform population over
    distance-optimal certified ties, while solved commitment retains the
    native random tie draw.
23. `search_action`, native Q/V targets, structural counts, and policy-mode Q
    reduction weights remain native under every root readout configuration.
24. `posterior_sample_temperature` transforms only the ephemeral commitment
    law. Its default $T=1$ preserves the existing direct sampler; it never
    changes the completed tree, replay targets, or current search weights.
25. For cross-entropy, policy-target error equals policy-logit-gradient error;
    capture metrics compare one fixed target and population before and after
    optimization, never two freshly searched targets.

---

## Appendix A. Dirichlet KL

For $\operatorname{Dirichlet}(\beta)$ and
$\operatorname{Dirichlet}(\alpha)$, with

$$
\beta_0=\sum_i\beta_i,
\qquad
\alpha_0=\sum_i\alpha_i,
$$

the full target-to-prediction KL is

$$
\begin{aligned}
D_{\mathrm{KL}}
\left(
\operatorname{Dirichlet}(\beta)
\,\|\,
\operatorname{Dirichlet}(\alpha)
\right)
={}&
\log\Gamma(\beta_0)-\log\Gamma(\alpha_0)\\
&+\sum_i\left[\log\Gamma(\alpha_i)-\log\Gamma(\beta_i)\right]\\
&+\sum_i(\beta_i-\alpha_i)
\left[\psi(\beta_i)-\psi(\beta_0)\right].
\end{aligned}
$$

This appendix applies only to native Dirichlet targets, for which $\beta$ is
stop-gradient. Native categorical targets have no $\beta$ and instead use the
density NLL in Section 11.3.
