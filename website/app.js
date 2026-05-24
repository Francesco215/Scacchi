"use strict";

const DEFAULT_SAMPLES_PER_SECOND = 300;
const MAX_FRAME_SAMPLES = 80;
const RENDER_INTERVAL_MS = 140;
const POLICY_SAMPLES = 128;
const PLOT_SAMPLES = 180;
const KAPPA_N = 4;
const EPSILON_TERM = 0.01;
const KAPPA_TERM = 80;
const DUMB_ALPHA = Object.freeze([1, 1, 1]);
const WIN_LINES = Object.freeze([
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
]);
const SYMMETRIES = Object.freeze([
  [0, 1, 2, 3, 4, 5, 6, 7, 8],
  [6, 3, 0, 7, 4, 1, 8, 5, 2],
  [8, 7, 6, 5, 4, 3, 2, 1, 0],
  [2, 5, 8, 1, 4, 7, 0, 3, 6],
  [2, 1, 0, 5, 4, 3, 8, 7, 6],
  [6, 7, 8, 3, 4, 5, 0, 1, 2],
  [0, 3, 6, 1, 4, 7, 2, 5, 8],
  [8, 5, 2, 7, 4, 1, 6, 3, 0],
]);

const OUTCOME = Object.freeze({
  LOSS: 0,
  DRAW: 1,
  WIN: 2,
});

class Edge {
  constructor() {
    this.B = null;
    this.m = false;
    this.R = 0;
  }
}

class Node {
  constructor(board, player, parent = null, parentAction = null) {
    this.board = board.slice();
    this.player = player;
    this.parent = parent;
    this.parentAction = parentAction;
    this.children = new Map();
    this.edges = new Map();
    this.alphaV = DUMB_ALPHA.slice();
    this.alphaQ = Array.from({ length: 9 }, () => DUMB_ALPHA.slice());
    this.cache = this.alphaV.slice();
    this.nDown = 0;
  }

  edge(action) {
    if (!this.edges.has(action)) {
      this.edges.set(action, new Edge());
    }
    return this.edges.get(action);
  }
}

let board = Array(9).fill(null);
let currentPlayer = "X";
let analysis = null;
let rootTree = null;
let nodeTable = new Map();
let completedSamples = 0;
let sampleCarry = 0;
let samplesPerSecond = DEFAULT_SAMPLES_PER_SECOND;
let lastTickTime = null;
let lastRenderTime = 0;

const els = {
  board: document.getElementById("board"),
  status: document.getElementById("status"),
  playBest: document.getElementById("play-best"),
  reset: document.getElementById("reset"),
  computeSlider: document.getElementById("compute-slider"),
  computeValue: document.getElementById("compute-value"),
  perspective: document.getElementById("perspective"),
  sims: document.getElementById("sims"),
  recommended: document.getElementById("recommended"),
  alphaTotal: document.getElementById("alpha-total"),
  nodeCount: document.getElementById("node-count"),
  alphaReadout: document.getElementById("alpha-readout"),
  lossBar: document.getElementById("loss-bar"),
  drawBar: document.getElementById("draw-bar"),
  winBar: document.getElementById("win-bar"),
  lossPct: document.getElementById("loss-pct"),
  drawPct: document.getElementById("draw-pct"),
  winPct: document.getElementById("win-pct"),
  simplex: document.getElementById("simplex"),
  actionList: document.getElementById("action-list"),
};

function opponent(player) {
  return player === "X" ? "O" : "X";
}

function legalActions(stateBoard) {
  const actions = [];
  for (let i = 0; i < stateBoard.length; i += 1) {
    if (stateBoard[i] === null) {
      actions.push(i);
    }
  }
  return actions;
}

function boardKey(stateBoard) {
  return stateBoard.map((mark) => mark ?? "-").join("");
}

function transformBoard(stateBoard, symmetry) {
  return symmetry.map((sourceIndex) => stateBoard[sourceIndex]);
}

function canonicalState(stateBoard, player) {
  let bestBoard = null;
  let bestBoardKey = null;

  for (const symmetry of SYMMETRIES) {
    const transformed = transformBoard(stateBoard, symmetry);
    const key = boardKey(transformed);
    if (bestBoardKey === null || key < bestBoardKey) {
      bestBoard = transformed;
      bestBoardKey = key;
    }
  }

  return {
    board: bestBoard,
    key: `${player}:${bestBoardKey}`,
  };
}

function gameResult(stateBoard) {
  for (const line of WIN_LINES) {
    const [a, b, c] = line;
    if (
      stateBoard[a] !== null &&
      stateBoard[a] === stateBoard[b] &&
      stateBoard[a] === stateBoard[c]
    ) {
      return {
        terminal: true,
        winner: stateBoard[a],
        draw: false,
        line,
      };
    }
  }

  if (stateBoard.every((mark) => mark !== null)) {
    return {
      terminal: true,
      winner: null,
      draw: true,
      line: [],
    };
  }

  return {
    terminal: false,
    winner: null,
    draw: false,
    line: [],
  };
}

function terminalOutcomeAlpha(stateBoard, playerToMove) {
  const result = gameResult(stateBoard);
  const alpha = [EPSILON_TERM, EPSILON_TERM, EPSILON_TERM];
  let index = OUTCOME.DRAW;

  if (!result.draw) {
    index = result.winner === playerToMove ? OUTCOME.WIN : OUTCOME.LOSS;
  }

  alpha[index] += KAPPA_TERM;
  return alpha;
}

function placeMove(stateBoard, player, action) {
  const next = stateBoard.slice();
  next[action] = player;
  return next;
}

function flipAlpha(alpha) {
  return [alpha[2], alpha[1], alpha[0]];
}

function alphaMean(alpha) {
  const total = alpha.reduce((sum, value) => sum + value, 0);
  return alpha.map((value) => value / total);
}

function utilityFromAlpha(alpha) {
  const mean = alphaMean(alpha);
  return mean[OUTCOME.WIN] - mean[OUTCOME.LOSS];
}

function edgePosterior(node, action) {
  const edge = node.edge(action);
  if (edge.m) {
    return edge.B.slice();
  }

  const child = node.children.get(action);
  if (child) {
    return flipAlpha(child.alphaV);
  }

  return node.alphaQ[action].slice();
}

function hasChildEvidence(node) {
  for (const edge of node.edges.values()) {
    if (edge.m) {
      return true;
    }
  }
  return false;
}

function thompsonSelect(node, actions) {
  let bestAction = actions[0];
  let bestUtility = -Infinity;

  for (const action of actions) {
    const phi = sampleDirichlet(edgePosterior(node, action));
    const utility = phi[OUTCOME.WIN] - phi[OUTCOME.LOSS];
    if (utility > bestUtility) {
      bestUtility = utility;
      bestAction = action;
    }
  }

  return bestAction;
}

function getOrCreateChild(node, action) {
  const existing = node.children.get(action);
  if (existing) {
    return { child: existing, isNew: false };
  }

  const nextBoard = placeMove(node.board, node.player, action);
  const nextPlayer = opponent(node.player);
  const canonical = canonicalState(nextBoard, nextPlayer);
  let child = nodeTable.get(canonical.key);
  let isNew = false;

  if (!child) {
    child = new Node(canonical.board, nextPlayer);
    nodeTable.set(canonical.key, child);
    isNew = true;
  }

  node.children.set(action, child);
  return { child, isNew };
}

function runSearch(root, simulationCount) {
  for (let i = 0; i < simulationCount; i += 1) {
    runSimulation(root);
  }
  repairSubtree(root);
}

function runSimulation(root) {
  let node = root;
  const path = [];

  while (true) {
    const result = gameResult(node.board);
    if (result.terminal) {
      backupPath(path, terminalOutcomeAlpha(node.board, node.player));
      return;
    }

    const actions = legalActions(node.board);
    if (actions.length === 0) {
      backupPath(path, terminalOutcomeAlpha(node.board, node.player));
      return;
    }

    const action = thompsonSelect(node, actions);
    path.push({ node, action });

    const { child, isNew } = getOrCreateChild(node, action);
    if (isNew) {
      const childResult = gameResult(child.board);
      const beta = childResult.terminal
        ? terminalOutcomeAlpha(child.board, child.player)
        : child.alphaV.slice();
      backupPath(path, beta);
      return;
    }

    node = child;
  }
}

function backupPath(path, betaLeaf) {
  if (path.length === 0) {
    return;
  }

  const finalStep = path[path.length - 1];
  const finalEdge = finalStep.node.edge(finalStep.action);
  finalEdge.B = flipAlpha(betaLeaf);
  finalEdge.m = true;
  finalEdge.R = 1;

  for (let i = path.length - 1; i >= 0; i -= 1) {
    repairNode(path[i].node);
  }
}

function repairSubtree(node, seen = new Set()) {
  if (seen.has(node)) {
    return;
  }
  seen.add(node);

  for (const child of node.children.values()) {
    repairSubtree(child, seen);
  }
  repairNode(node);
}

function repairNode(node) {
  const result = gameResult(node.board);
  if (result.terminal) {
    return;
  }

  for (const action of legalActions(node.board)) {
    const child = node.children.get(action);
    if (!child || gameResult(child.board).terminal || !hasChildEvidence(child)) {
      continue;
    }

    const edge = node.edge(action);
    edge.B = flipAlpha(child.cache);
    edge.m = true;
    edge.R = 1 + child.nDown;
  }

  if (!hasChildEvidence(node)) {
    node.cache = node.alphaV.slice();
    node.nDown = 0;
    return;
  }

  const posterior = computeStateSearchPosterior(node);
  node.cache = posterior.alpha;
  node.nDown = posterior.nDown;
}

function computeStateSearchPosterior(node) {
  const actions = legalActions(node.board);
  const alphaByAction = new Map();
  let nDown = 0;

  for (const action of actions) {
    alphaByAction.set(action, edgePosterior(node, action));
    nDown += node.edge(action).R;
  }

  const policy = posteriorBestPolicy(alphaByAction, actions, POLICY_SAMPLES);
  const searchWeighted = [0, 0, 0];

  for (const action of actions) {
    const weight = policy.get(action) ?? 0;
    const alpha = alphaByAction.get(action);
    for (let i = 0; i < 3; i += 1) {
      searchWeighted[i] += weight * alpha[i];
    }
  }

  const gamma = nDown / (KAPPA_N + nDown);
  return {
    alpha: node.alphaV.map((prior, i) => (1 - gamma) * prior + gamma * searchWeighted[i]),
    nDown,
  };
}

function posteriorBestPolicy(alphaByAction, actions, sampleCount) {
  const counts = new Map(actions.map((action) => [action, 0]));

  for (let sample = 0; sample < sampleCount; sample += 1) {
    let bestAction = actions[0];
    let bestUtility = -Infinity;

    for (const action of actions) {
      const phi = sampleDirichlet(alphaByAction.get(action));
      const utility = phi[OUTCOME.WIN] - phi[OUTCOME.LOSS];
      if (utility > bestUtility) {
        bestUtility = utility;
        bestAction = action;
      }
    }

    counts.set(bestAction, counts.get(bestAction) + 1);
  }

  const policy = new Map();
  for (const action of actions) {
    policy.set(action, counts.get(action) / sampleCount);
  }
  return policy;
}

function analyzePosition() {
  const result = gameResult(board);
  if (result.terminal) {
    return {
      root: null,
      valueAlpha: terminalOutcomeAlpha(board, currentPlayer),
      rootPosteriors: new Map(),
      policy: new Map(),
      bestAction: null,
      simulations: completedSamples,
      nodes: 0,
    };
  }

  if (!rootTree) {
    resetAnalysisTree();
  }

  const actions = legalActions(board);
  const rootPosteriors = new Map();
  for (const action of actions) {
    rootPosteriors.set(action, edgePosterior(rootTree, action));
  }

  const policy = posteriorBestPolicy(rootPosteriors, actions, POLICY_SAMPLES);
  const bestAction = bestPolicyAction(policy, actions);

  return {
    root: rootTree,
    valueAlpha: rootTree.cache.slice(),
    rootPosteriors,
    policy,
    bestAction,
    simulations: completedSamples,
    nodes: nodeTable.size + 1,
  };
}

function bestPolicyAction(policy, actions) {
  let bestAction = null;
  let bestProbability = -Infinity;

  for (const action of actions) {
    const probability = policy.get(action) ?? 0;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestAction = action;
    }
  }

  return bestAction;
}

function actionName(action) {
  if (action === null || action === undefined) {
    return "None";
  }
  const row = Math.floor(action / 3) + 1;
  const col = (action % 3) + 1;
  return `r${row} c${col}`;
}

function formatNumber(value) {
  return value.toFixed(2);
}

function formatPct(value) {
  return `${Math.round(value * 100)}%`;
}

function render() {
  analysis = analyzePosition();
  renderComputeControl();
  renderBoard();
  renderStatus();
  renderPosterior();
  renderActions();
}

function renderComputeControl() {
  els.computeSlider.value = String(samplesPerSecond);
  els.computeValue.textContent =
    samplesPerSecond === 0 ? "paused" : `${samplesPerSecond}/s`;
}

function renderBoard() {
  const result = gameResult(board);
  els.board.innerHTML = "";

  for (let action = 0; action < 9; action += 1) {
    const cell = document.createElement("button");
    const mark = board[action];
    cell.type = "button";
    cell.className = "cell";
    cell.textContent = mark ?? "";
    cell.disabled = result.terminal || mark !== null;
    cell.setAttribute("aria-label", `Square ${actionName(action)}`);

    if (mark) {
      cell.classList.add(mark === "X" ? "mark-x" : "mark-o");
    }
    if (analysis.bestAction === action && !result.terminal && mark === null) {
      cell.classList.add("recommended");
    }
    if (result.line.includes(action)) {
      cell.classList.add("win-line");
    }

    cell.addEventListener("click", () => playMove(action));
    els.board.appendChild(cell);
  }
}

function renderStatus() {
  const result = gameResult(board);

  if (result.terminal) {
    els.status.textContent = result.draw ? "Draw" : `${result.winner} wins`;
  } else {
    els.status.textContent = `${currentPlayer} to move`;
  }

  els.playBest.disabled = result.terminal || analysis.bestAction === null;
}

function renderPosterior() {
  const alpha = analysis.valueAlpha;
  const mean = alphaMean(alpha);
  const total = alpha.reduce((sum, value) => sum + value, 0);

  els.perspective.textContent = currentPlayer;
  els.sims.textContent = String(analysis.simulations);
  els.recommended.textContent = actionName(analysis.bestAction);
  els.alphaTotal.textContent = formatNumber(total);
  els.nodeCount.textContent = String(analysis.nodes);
  els.alphaReadout.textContent = `L=${formatNumber(alpha[0])}  D=${formatNumber(
    alpha[1],
  )}  W=${formatNumber(alpha[2])}`;

  setBar(els.lossBar, els.lossPct, mean[OUTCOME.LOSS]);
  setBar(els.drawBar, els.drawPct, mean[OUTCOME.DRAW]);
  setBar(els.winBar, els.winPct, mean[OUTCOME.WIN]);
  drawSimplex(alpha);
}

function setBar(bar, label, value) {
  bar.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
  label.textContent = formatPct(value);
}

function renderActions() {
  els.actionList.innerHTML = "";
  const result = gameResult(board);
  const actions = legalActions(board);

  if (result.terminal || actions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "action-row";
    empty.textContent = "No legal actions";
    els.actionList.appendChild(empty);
    return;
  }

  for (const action of actions) {
    const row = document.createElement("div");
    const probability = analysis.policy.get(action) ?? 0;
    const alpha = analysis.rootPosteriors.get(action) ?? DUMB_ALPHA;
    const q = utilityFromAlpha(alpha);

    row.className = "action-row";
    if (action === analysis.bestAction) {
      row.classList.add("best");
    }

    row.innerHTML = `
      <strong>${actionName(action)}</strong>
      <div class="policy-track" aria-label="Posterior-best probability">
        <div class="policy-fill" style="width: ${Math.round(probability * 100)}%"></div>
      </div>
      <span>${formatPct(probability)} / q ${q.toFixed(2)}</span>
    `;
    els.actionList.appendChild(row);
  }
}

function playMove(action) {
  const result = gameResult(board);
  if (result.terminal || board[action] !== null) {
    return;
  }

  board = placeMove(board, currentPlayer, action);
  currentPlayer = opponent(currentPlayer);
  resetAnalysisTree();
  render();
}

function resetGame() {
  board = Array(9).fill(null);
  currentPlayer = "X";
  resetAnalysisTree();
  render();
}

function playBestMove() {
  analysis = analyzePosition();
  if (analysis.bestAction !== null) {
    playMove(analysis.bestAction);
  }
}

function setComputeBudget(value) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return;
  }

  samplesPerSecond = parsed;
  renderComputeControl();
}

function resetAnalysisTree() {
  nodeTable = new Map();
  rootTree = gameResult(board).terminal ? null : new Node(board, currentPlayer);
  completedSamples = 0;
  sampleCarry = 0;
}

function searchTick(timestamp) {
  if (lastTickTime === null) {
    lastTickTime = timestamp;
  }

  const elapsedSeconds = Math.min((timestamp - lastTickTime) / 1000, 0.25);
  lastTickTime = timestamp;

  if (!gameResult(board).terminal && rootTree && samplesPerSecond > 0) {
    sampleCarry += samplesPerSecond * elapsedSeconds;
    const samplesToRun = Math.min(Math.floor(sampleCarry), MAX_FRAME_SAMPLES);

    if (samplesToRun > 0) {
      runSearch(rootTree, samplesToRun);
      completedSamples += samplesToRun;
      sampleCarry -= samplesToRun;

      if (timestamp - lastRenderTime >= RENDER_INTERVAL_MS) {
        render();
        lastRenderTime = timestamp;
      }
    }
  }

  window.requestAnimationFrame(searchTick);
}

let spareNormal = null;

function sampleNormal() {
  if (spareNormal !== null) {
    const value = spareNormal;
    spareNormal = null;
    return value;
  }

  let u = 0;
  let v = 0;
  while (u === 0) {
    u = Math.random();
  }
  while (v === 0) {
    v = Math.random();
  }

  const radius = Math.sqrt(-2 * Math.log(u));
  const theta = 2 * Math.PI * v;
  spareNormal = radius * Math.sin(theta);
  return radius * Math.cos(theta);
}

function sampleGamma(shape) {
  const safeShape = Math.max(shape, 1e-9);
  if (safeShape < 1) {
    return sampleGamma(safeShape + 1) * Math.pow(Math.random(), 1 / safeShape);
  }

  const d = safeShape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    const x = sampleNormal();
    let v = 1 + c * x;
    if (v <= 0) {
      continue;
    }
    v = v * v * v;

    const u = Math.random();
    if (u < 1 - 0.0331 * x ** 4) {
      return d * v;
    }
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v;
    }
  }
}

function sampleDirichlet(alpha) {
  const gammas = alpha.map((shape) => sampleGamma(shape));
  const total = gammas.reduce((sum, value) => sum + value, 0);
  return gammas.map((value) => value / total);
}

function drawSimplex(alpha) {
  const canvas = els.simplex;
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const margin = 34;
  const vertices = {
    L: { x: margin, y: rect.height - margin },
    D: { x: rect.width / 2, y: margin },
    W: { x: rect.width - margin, y: rect.height - margin },
  };

  context.lineWidth = 1.4;
  context.strokeStyle = "#252a32";
  context.fillStyle = "#fbfbfc";
  context.beginPath();
  context.moveTo(vertices.L.x, vertices.L.y);
  context.lineTo(vertices.D.x, vertices.D.y);
  context.lineTo(vertices.W.x, vertices.W.y);
  context.closePath();
  context.fill();
  context.stroke();

  context.font = "12px ui-sans-serif, system-ui, sans-serif";
  context.fillStyle = "#4b5563";
  context.textAlign = "center";
  context.fillText("D", vertices.D.x, vertices.D.y - 10);
  context.fillText("L", vertices.L.x - 14, vertices.L.y + 4);
  context.fillText("W", vertices.W.x + 14, vertices.W.y + 4);

  context.fillStyle = "rgba(31, 122, 140, 0.24)";
  for (let i = 0; i < PLOT_SAMPLES; i += 1) {
    const point = simplexPoint(sampleDirichlet(alpha), vertices);
    context.beginPath();
    context.arc(point.x, point.y, 2.2, 0, Math.PI * 2);
    context.fill();
  }

  const meanPoint = simplexPoint(alphaMean(alpha), vertices);
  context.fillStyle = "#c2413d";
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(meanPoint.x, meanPoint.y, 6, 0, Math.PI * 2);
  context.fill();
  context.stroke();
}

function simplexPoint(probabilities, vertices) {
  return {
    x:
      probabilities[OUTCOME.LOSS] * vertices.L.x +
      probabilities[OUTCOME.DRAW] * vertices.D.x +
      probabilities[OUTCOME.WIN] * vertices.W.x,
    y:
      probabilities[OUTCOME.LOSS] * vertices.L.y +
      probabilities[OUTCOME.DRAW] * vertices.D.y +
      probabilities[OUTCOME.WIN] * vertices.W.y,
  };
}

els.playBest.addEventListener("click", playBestMove);
els.reset.addEventListener("click", resetGame);
els.computeSlider.addEventListener("input", (event) => {
  setComputeBudget(event.target.value);
});
window.addEventListener("resize", () => {
  if (analysis) {
    drawSimplex(analysis.valueAlpha);
  }
});

resetAnalysisTree();
render();
window.requestAnimationFrame(searchTick);
