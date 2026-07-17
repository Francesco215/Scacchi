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
certified distance to terminal. These categorical certificates are stored in
sidecar arrays rather than encoded by a large or sharply peaked Dirichlet. They
are absorbing and authoritative. Model alphas remain the learned representation
for unresolved leaves and caches; a solved result is never reconstructed from
their shape or concentration.

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
utilities or sampled action Dirichlets plus the legal-action mask. Stored logits
only break categorical draws under `policy_prior`.

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
    terminal=env_state.terminated,
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
    categorical_draw_rule=categorical_draw_rule,
)
```

`categorical_draw_rule` is one of `policy_prior`, `fastest_draw`,
`slowest_draw`, or `fixed_order`. It matters only after the node has proved a
draw and always selects among certified draw edges, never losing edges.
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

It returns the child embedding, child policy logits, $V_{s'}$, all
$Q_{s',b}$, player to move, and one exact `terminal_outcome` tag. The tag is an
outcome index from the child's player perspective when the child is terminal,
and `NO_OUTCOME` otherwise. It is the complete terminal payload: search uses it
to publish the native categorical certificate without constructing a terminal
Dirichlet.

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

For every node $s$, the tree stores:

- the fixed network value prior $V_s$;
- a mutable cached state posterior $C_s$;
- fixed policy logits, used only by the `policy_prior` draw rule;
- a categorical outcome $z_s\in\mathcal Z\cup\{\bot\}$ and distance
  $\tau_s\in\mathbb N_0\cup\{-1\}$;
- topology, player, terminal status, legality, and environment embedding.

For every edge $(s,a)$ it stores:

- a mutable Dirichlet message $B_{s,a}$;
- a non-negative structural count $R_{s,a}$.
- a categorical outcome $z_{s,a}\in\mathcal Z\cup\{\bot\}$ and distance
  $\tau_{s,a}\in\mathbb N\cup\{-1\}$.

Here $\bot$ and distance $-1$ mean unresolved. Terminal children discovered by
expansion have distance zero. A root supplied as already terminal is skipped,
but remains unresolved in these sidecars because `RootFnOutput` carries no
terminal outcome. An expanded categorical child publishes an aligned edge certificate

$$
z_{s,a}=\operatorname{Align}(z_{s_a};p_{s_a}\to p_s),
\qquad
\tau_{s,a}=1+\tau_{s_a}.
$$

The implementation caches both node and edge sidecars, but it does not store a
categorical action. The action is derived deterministically from the immutable
edge certificates whenever readout needs it.

The cache is initialized from the state prior:

$$
C_s\leftarrow V_s.
$$

The action-message array is initialized with the network fallback $Q_{s,a}$.
That fallback occupies the slot until a real message replaces it, and the slot
is considered a real message only when

$$
m_{s,a}=\mathbf 1[R_{s,a}>0].
$$

This single condition replaces the demo's separate Boolean `m` field.

Define the node's downstream count as

$$
n_s=\sum_{a\in\mathcal A(s)}R_{s,a}.
$$

`R` and $n_s$ describe repaired tree structure. They are not Dirichlet
parameters and are never added to $B$, $C$, $V$, or $Q$.

### 4.1 Effective native action object

Let $s_a$ be the expanded child on edge $(s,a)$ when it exists. The Dirichlet
used for selection and readout is

$$
A_{s,a}=
\begin{cases}
B_{s,a},
& R_{s,a}>0,\\[3pt]
\operatorname{Align}(V_{s_a};p_{s_a}\to p_s),
& R_{s,a}=0\text{ and }s_a\text{ exists},\\[3pt]
Q_{s,a},
& \text{otherwise}.
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
\operatorname{Cat}(z_{s,a},\tau_{s,a}),&z_{s,a}\ne\bot,\\
\operatorname{Dir}(A_{s,a}),&z_{s,a}=\bot.
\end{cases}
$$

Alpha-shaped arrays remain populated for every edge and node because JAX needs
fixed shapes and the unresolved state-cache update needs a learned mass for
its numeric mixture. Categorical utility, propagation, root choice, and neural
loss always consult the sidecars. Once $z_s\ne\bot$, the node certificate
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

Thus the exact sidecar supplies the direction of a categorical edge, while its
existing effective alpha supplies only the learned concentration. That mass
may come from the Q fallback, an expanded-child prior, or an earlier repaired
message. The projection is computed only as an operand of the node-cache
mixture: it does not mutate $A_{s,a}$, does not replace the sidecar, and is not
emitted as a categorical neural target.

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
   winning edge by minimizing $(\tau_{s,a},a)$ over $\mathcal W_s$ and store
   only its distance as $\tau_s$; the action itself remains derived.
2. Otherwise, while $\mathcal U_s\ne\varnothing$, the node remains unresolved.
3. If every edge is categorical and $\mathcal D_s\ne\varnothing$, the node is a
   draw. Choose only among $\mathcal D_s$: highest stored policy logit for
   `policy_prior`, minimum distance for `fastest_draw`, maximum distance for
   `slowest_draw`, or lowest action index for `fixed_order`.
4. Otherwise every legal edge is a loss. The node is a categorical loss and
   derives the action with maximum distance, breaking ties by lowest index.

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
4. follow its child if that edge is expanded and the child is unresolved and
   non-terminal;
5. repeat until the chosen edge is unexpanded or reaches `max_depth`.

The tree is persistent across all simulations in one policy call, so later
simulations see every message and cache repaired by earlier simulations.

### 6.2 Expand

Call `expand_fn` once on the final unresolved parent-action pair. This performs
the environment step and leaf evaluation only for an active simulation. If the
edge is new, initialize the child node with its embedding, logits, $V$, $Q$,
player, terminal flag, and legal-action mask. A terminal result immediately
publishes $\operatorname{Cat}(z,0)$ on the child and its aligned distance-one
edge certificate. A depth-cutoff revisit reuses existing topology and its fresh
uncertain leaf message.

### 6.3 Repair

Begin at the final parent and walk through parent links back to the root. At
every node, repair the unresolved Dirichlet posterior, refresh categorical
sidecars, and apply the categorical rules in Section 4.2. A newly categorical
node publishes its aligned certificate to its incoming edge. The order is
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
    edge_categorical_outcome=...,
    edge_categorical_distance=...,
)
```

and invokes

```python
new_posterior = posterior_update(rng_key, context)
```

The callback returns a complete

```python
NodePosterior(
    action_alpha=...,
    action_count=...,
    value_alpha=...,
)
```

for that node. The context exposes:

- the current node embedding, fixed $V_s$, legality, player, and old posterior;
- action-indexed child indices, fixed $V_{s_a}$, repaired $C_{s_a}$, $n_{s_a}$,
  player, and terminal flag;
- the final selected edge and its model value alpha, active only at the deepest
  path node and written as a Dirichlet message only while that edge is
  unresolved;
- the current action-indexed categorical outcome/distance sidecars.

Large child embeddings are not copied across the action axis. A custom rule
that needs them can gather the shared embedding table using child indices.

The callback owns the direct unresolved-message write and uncertain
child-to-parent Dirichlet repairs. Replacing it changes those posterior
mathematics at every node, but not terminal detection, categorical minimax
rules, absorbing certificates, or traversal pruning.

---

## 8. Default posterior repair

The default `update_posterior` is the JAX translation of the message-passing
logic in `tictactoe-demo/app.js`.

### 8.1 Write the final leaf message

Only at the deepest path node $s_d$, for final action $a_d$, increment the
structural count:

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

If `terminal_outcome` is present, expansion has already published the native
categorical node/edge certificate. The update records structural support in
$R$ but does not overwrite $B$ with a terminal alpha. The exact sidecar owns
selection, propagation, readout, and training.

### 8.2 Refresh messages from repaired children

At the current node $s$, inspect all expanded children. For every
non-terminal child $s_a$ whose edge is unresolved and whose downstream
information satisfies $n_{s_a}>0$, replace the parent edge with the child's
repaired cache:

$$
B_{s,a}\leftarrow
\operatorname{Align}(C_{s_a};p_{s_a}\to p_s),
$$

$$
R_{s,a}\leftarrow 1+n_{s_a}.
$$

Categorical children publish aligned categorical edge sidecars with distance
$1+\tau_{s_a}$ and remain excluded from uncertain alpha refresh. A child with
no downstream message leaves the alpha slot on its earlier unresolved message
or fixed child-value fallback.

### 8.3 Resolve categorical nodes

After each node's unresolved-posterior repair, apply Section 4.2. A newly
categorical node immediately publishes its aligned certificate to its parent
edge. Its certificate is absorbing, and later simulations cannot enter that
node or edge.

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

Recompute

$$
n_s=\sum_{a\in\mathcal A(s)}R_{s,a},
\qquad
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

This is a convex interpolation of Dirichlet parameter vectors. It is the
algorithm's message-passing rule; it should not be described as a conjugate
update with $R$ categorical observations.

For a categorical edge, $\widetilde A_{s,a}$ is the learned-mass projection
from Section 4.1; the sidecar still supplies policy utility and exact semantics.
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

## 9. Root readout

After all active simulations, compute the native root objects

$$
Y_a^{\mathrm{root}}=Y_{s_0,a}.
$$

If the root remains unresolved, the public target is another fresh mixed-native
population:

$$
\pi_{\mathrm{search}}(a\mid s_0)
=\widehat\pi_{M_{\mathrm{root}}}
(a\mid s_0;Y^{\mathrm{root}}).
$$

If the root is categorical, derive its certified action $a_{\mathrm{cat}}(s_0)$
from the edge sidecars and draw rule, then return

$$
\pi_{\mathrm{search}}(a\mid s_0)
=\mathbf 1[a=a_{\mathrm{cat}}(s_0)].
$$

The backend's `PolicyOutput` contains

$$
\texttt{action\_weights}=\pi_{\mathrm{search}},
$$

$$
\texttt{action}=\arg\max_a\pi_{\mathrm{search}}(a\mid s_0),
$$

and the completed tree.

$M_{\mathrm{node}}$ (`posterior_policy_samples`) and $M_{\mathrm{root}}$
(`policy_samples`) estimate the same quantity but may use different Monte
Carlo budgets. The internal budget affects the stochastic cache repair; the
root budget affects the variance of the public policy and training target.

### 9.1 Committed self-play action

Search output and game action are separate. Scacchi can commit by:

- `posterior_argmax`: $\arg\max_a\pi_{\mathrm{search}}(a)$;
- `posterior_sample`: a categorical draw from $\pi_{\mathrm{search}}$;
- `search_action`: the backend action returned with the policy output.

For this backend, `search_action` is also the masked argmax of the same public
policy population. The separation is retained so all search backends share one
self-play interface. At a categorical root the policy is one-hot, so all three
commitment modes choose the same certified action.

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

$$
\pi_{\mathrm{tgt}}(a\mid s_0)=
\pi_{\mathrm{search}}(a\mid s_0).
$$

The fixed-shape replay record carries the eight sidecars
`q_target_kind/weight/outcome/distance` and
`v_target_kind/weight/outcome/distance`. Unresolved alpha arrays remain in
`tree.summary().alpha` and `tree.summary().value_alpha`; the learner ignores
their entries whenever the corresponding kind is categorical or padded.

There is no reconstruction

$$
\beta_Q=\alpha_{\mathrm{base}}+E_Q
$$

and no addition of $R$ to Dirichlet concentration. Doing either would train
the network on a posterior different from the one search actually used.

The Q-loss weight is configurable:

$$
w_a=
\begin{cases}
R_{s_0,a}, & \texttt{q\_loss\_weight\_mode=evidence\_mass},\\
\pi_{\mathrm{search}}(a\mid s_0),
& \texttt{q\_loss\_weight\_mode=policy}.
\end{cases}
$$

Here the name `evidence_mass` is historical: in this backend it receives the
structural edge count $R$, not added Dirichlet pseudo-count mass.
Every legal categorical Q edge is given positive effective weight even if a
one-hot solved policy assigns it zero probability, so exact alternative moves
are not silently discarded.

### 11.1 Policy loss

The policy head is trained by cross-entropy on legal actions:

$$
\mathcal L_\pi(s)=
-\sum_{a\in\mathcal A(s)}
\operatorname{stopgrad}(\pi_{\mathrm{tgt}}(a\mid s))
\log\pi_\theta(a\mid s).
$$

### 11.2 Typed Dirichlet-head loss

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
$(T^Q(a),\alpha_\theta^Q(a))$ and reduces actions using the native target weight,
$w_a$, legality, and the configured reduction mode.

### 11.3 Outcome supervision

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
    initialize root priors/logits and cache C_root <- V_root
    initialize every categorical sidecar to unresolved

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
            child.categorical <- (aligned terminal outcome, distance 0)
            parent edge categorical <- (parent outcome, distance 1)

        # repair
        for node on the selected path, deepest to root:
            context <- (node, child summaries, categorical edges, final leaf)
            node.posterior <- posterior_update(key, context)
            try categorical win/draw/loss rules at node
            if node became categorical and has a parent:
                publish aligned outcome and distance + 1 to incoming edge

    if root is categorical:
        root_action <- derive action from root edge certificates and draw rule
        root_policy <- one_hot(root_action)
    else:
        root_native <- categorical edges plus effective Dirichlet edges
        root_policy <- repeated native utility samples from root_native
    return argmax(root_policy), root_policy, tree
```

For the default callback, `posterior_update` is:

```text
DEFAULT-POSTERIOR-UPDATE(node, children, leaf):
    if this is the deepest path node:
        if leaf edge is unresolved:
            B[leaf.action] <- aligned model leaf Dirichlet
        R[leaf.action] <- R[leaf.action] + 1

    for each expanded, nonterminal child with child.n_down > 0:
        B[action] <- aligned child.cache
        R[action] <- 1 + child.n_down

    A <- effective(B, R, expanded child V priors, Q fallbacks)
    Y <- categorical sidecar when present, otherwise Dir(A)
    pi <- fresh repeated-native policy from Y
    n_down <- sum of legal R
    A_tilde <- A for unresolved edges
               or sum(A) * one_hot(exact outcome) for categorical edges
    gamma <- n_down / (kappa + n_down)
    cache <- (1 - gamma) * V_prior
             + gamma * sum_action pi[action] * A_tilde[action]
    return B, R, cache
```

---

## 13. Core invariants

A correct implementation maintains all of the following:

1. Every Dirichlet and categorical outcome uses its node's player perspective.
2. A terminal child discovered by expansion has distance zero; its parent edge
   adds one ply. A pre-terminal root has no outcome payload and is only skipped.
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
11. $R$ controls structure and prior mixing; it is not Dirichlet concentration.
12. Numeric alphas, cache projections, and caches never override native
    certificates or become categorical targets.
13. Categorical V/Q targets use density NLL in both Dirichlet loss modes;
    unresolved targets retain the configured full/mean behavior.
14. `num_simulations` and $N+1$ are static iteration/capacity bounds;
    categorical short-circuiting may use fewer active cycles and real nodes.

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
density NLL in Section 11.2.
