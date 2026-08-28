"use strict";

function reportEmbedHeight() {
  if (window.parent === window) {
    return;
  }

  const app = document.querySelector(".app");
  const contentHeight = app
    ? Math.ceil(app.getBoundingClientRect().bottom)
    : Math.ceil(document.documentElement.scrollHeight);

  window.parent.postMessage(
    {
      type: "tictactoe-demo:resize",
      height: contentHeight,
    },
    "*",
  );
}

if (window.parent !== window) {
  window.addEventListener("load", reportEmbedHeight);
  const embeddedApp = document.querySelector(".app");
  if (embeddedApp) {
    new ResizeObserver(reportEmbedHeight).observe(embeddedApp);
  }

  // An iframe is an independent scroll context, so wheel events do not
  // naturally bubble to the article. Forward them to the parent to make this
  // interactive demo move with the page like an ordinary figure.
  window.addEventListener(
    "wheel",
    (event) => {
      if (event.ctrlKey) {
        return;
      }
      window.parent.scrollBy({ left: event.deltaX, top: event.deltaY });
      event.preventDefault();
    },
    { passive: false },
  );
}

const DEFAULT_SAMPLES_PER_SECOND = 300;
const MAX_FRAME_SAMPLES = 80;
const RENDER_INTERVAL_MS = 140;
const POLICY_SAMPLES = 128;
const PLOT_SAMPLES = 180;
const KAPPA = 4;
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
    this.outcome = null;
    this.distance = null;
    this.m = false;
    this.R = 0;
  }
}

class Node {
  constructor(board, player) {
    this.board = board.slice();
    this.player = player;
    this.children = new Map();
    this.edges = new Map();
    this.alphaV = DUMB_ALPHA.slice();
    this.alphaQ = Array.from({ length: 9 }, () => DUMB_ALPHA.slice());
    this.cache = this.alphaV.slice();
    this.nDown = 0;
    const result = gameResult(this.board);
    this.outcome = result.terminal
      ? terminalOutcome(this.board, this.player)
      : null;
    this.distance = result.terminal ? 0 : null;
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
  xWinBar: document.getElementById("x-win-bar"),
  drawBar: document.getElementById("draw-bar"),
  oWinBar: document.getElementById("o-win-bar"),
  xWinPct: document.getElementById("x-win-pct"),
  drawPct: document.getElementById("draw-pct"),
  oWinPct: document.getElementById("o-win-pct"),
  simplex: document.getElementById("simplex"),
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

function terminalOutcome(stateBoard, playerToMove) {
  const result = gameResult(stateBoard);
  if (result.draw) {
    return OUTCOME.DRAW;
  }
  return result.winner === playerToMove ? OUTCOME.WIN : OUTCOME.LOSS;
}

function categoricalMean(outcome) {
  return [0, 0, 0].map((_, index) => (index === outcome ? 1 : 0));
}

function flipOutcome(outcome) {
  if (outcome === OUTCOME.LOSS) {
    return OUTCOME.WIN;
  }
  if (outcome === OUTCOME.WIN) {
    return OUTCOME.LOSS;
  }
  return OUTCOME.DRAW;
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

function edgePosterior(node, action) {
  const edge = node.edge(action);
  if (edge.m && edge.B !== null) {
    return edge.B.slice();
  }

  if (edge.outcome !== null) {
    return node.alphaQ[action].slice();
  }

  const child = node.children.get(action);
  if (child) {
    return flipAlpha(child.alphaV);
  }

  return node.alphaQ[action].slice();
}

function actionPosterior(node, action) {
  const edge = node.edge(action);
  return {
    alpha: edgePosterior(node, action),
    outcome: edge.m ? edge.outcome : null,
  };
}

function searchableActions(node) {
  return legalActions(node.board).filter(
    (action) => node.edge(action).outcome === null,
  );
}

function posteriorMean(posterior) {
  return posterior.outcome === null
    ? alphaMean(posterior.alpha)
    : categoricalMean(posterior.outcome);
}

function posteriorUtility(posterior) {
  const mean = posteriorMean(posterior);
  return mean[OUTCOME.WIN] - mean[OUTCOME.LOSS];
}

function samplePosteriorUtility(posterior) {
  if (posterior.outcome !== null) {
    return posteriorUtility(posterior);
  }
  const phi = sampleDirichlet(posterior.alpha);
  return phi[OUTCOME.WIN] - phi[OUTCOME.LOSS];
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
    const utility = samplePosteriorUtility(actionPosterior(node, action));
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
  if (root.outcome !== null) {
    return 0;
  }
  let completed = 0;
  for (let i = 0; i < simulationCount; i += 1) {
    if (!runSimulation(root)) {
      break;
    }
    completed += 1;
  }
  if (completed > 0) {
    repairSubtree(root);
  }
  return completed;
}

function runSimulation(root) {
  let node = root;
  const path = [];

  while (true) {
    const result = gameResult(node.board);
    if (result.terminal || node.outcome !== null) {
      return false;
    }

    const actions = searchableActions(node);
    if (actions.length === 0) {
      repairNode(node);
      return false;
    }

    const action = thompsonSelect(node, actions);
    path.push({ node, action });

    const { child, isNew } = getOrCreateChild(node, action);
    if (child.outcome !== null) {
      backupPath(path, null, child.outcome, child.distance);
      return true;
    }
    if (isNew) {
      backupPath(path, child.alphaV.slice());
      return true;
    }

    node = child;
  }
}

function backupPath(
  path,
  betaLeaf,
  categoricalOutcome = null,
  categoricalDistance = null,
) {
  if (path.length === 0) {
    return;
  }

  const finalStep = path[path.length - 1];
  const finalEdge = finalStep.node.edge(finalStep.action);
  if (categoricalOutcome === null) {
    finalEdge.B = flipAlpha(betaLeaf);
    finalEdge.outcome = null;
    finalEdge.distance = null;
  } else {
    // A terminal result is exact. Keep it in a categorical sidecar instead
    // of inventing a concentrated Dirichlet pseudo-observation.
    finalEdge.B = null;
    finalEdge.outcome = flipOutcome(categoricalOutcome);
    finalEdge.distance = categoricalDistance + 1;
  }
  finalEdge.m = true;
  finalEdge.R += 1;

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
    if (!child) {
      continue;
    }

    const edge = node.edge(action);
    if (child.outcome !== null) {
      if (edge.outcome === null) {
        edge.outcome = flipOutcome(child.outcome);
        edge.distance = child.distance + 1;
        edge.m = true;
      }
      continue;
    }
    if (edge.outcome !== null || !hasChildEvidence(child)) {
      continue;
    }
    edge.B = flipAlpha(child.cache);
    edge.outcome = null;
    edge.distance = null;
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
  categorizeNode(node);
}

function categoricalAction(node) {
  const candidates = legalActions(node.board).filter(
    (action) => node.edge(action).outcome === node.outcome,
  );
  if (candidates.length === 0) {
    return null;
  }
  if (node.outcome === OUTCOME.WIN) {
    return candidates.reduce((best, action) =>
      node.edge(action).distance < node.edge(best).distance ? action : best,
    );
  }
  if (node.outcome === OUTCOME.LOSS) {
    return candidates.reduce((best, action) =>
      node.edge(action).distance > node.edge(best).distance ? action : best,
    );
  }
  return candidates[0];
}

function categorizeNode(node) {
  if (node.outcome !== null) {
    return;
  }
  const actions = legalActions(node.board);
  const known = actions.filter(
    (action) => node.edge(action).outcome !== null,
  );
  const winning = known.filter(
    (action) => node.edge(action).outcome === OUTCOME.WIN,
  );

  if (winning.length > 0) {
    node.outcome = OUTCOME.WIN;
  } else if (known.length === actions.length) {
    node.outcome = known.some(
      (action) => node.edge(action).outcome === OUTCOME.DRAW,
    )
      ? OUTCOME.DRAW
      : OUTCOME.LOSS;
  } else {
    return;
  }

  const action = categoricalAction(node);
  node.distance = node.edge(action).distance;
}

function computeStateSearchPosterior(node) {
  const actions = legalActions(node.board);
  const posteriorByAction = new Map();
  let nDown = 0;

  for (const action of actions) {
    posteriorByAction.set(action, actionPosterior(node, action));
    nDown += node.edge(action).R;
  }

  const policy = posteriorBestPolicy(
    posteriorByAction,
    actions,
    POLICY_SAMPLES,
  );
  const searchWeighted = [0, 0, 0];

  for (const action of actions) {
    const weight = policy.get(action) ?? 0;
    const posterior = posteriorByAction.get(action);
    let alpha = posterior.alpha;
    if (posterior.outcome !== null) {
      const learnedMass = alpha.reduce((sum, value) => sum + value, 0);
      alpha = categoricalMean(posterior.outcome).map(
        (probability) => probability * learnedMass,
      );
    }
    for (let i = 0; i < 3; i += 1) {
      searchWeighted[i] += weight * alpha[i];
    }
  }

  const gamma = nDown / (KAPPA + nDown);
  return {
    alpha: node.alphaV.map((prior, i) => (1 - gamma) * prior + gamma * searchWeighted[i]),
    nDown,
  };
}

function posteriorBestPolicy(posteriorByAction, actions, sampleCount) {
  const counts = new Map(actions.map((action) => [action, 0]));

  for (let sample = 0; sample < sampleCount; sample += 1) {
    let bestAction = actions[0];
    let bestUtility = -Infinity;

    for (const action of actions) {
      const utility = samplePosteriorUtility(posteriorByAction.get(action));
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
      valueAlpha: DUMB_ALPHA.slice(),
      valueOutcome: terminalOutcome(board, currentPlayer),
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
    rootPosteriors.set(action, actionPosterior(rootTree, action));
  }

  let policy;
  let bestAction;
  if (rootTree.outcome !== null) {
    bestAction = categoricalAction(rootTree);
    policy = new Map(
      actions.map((action) => [action, action === bestAction ? 1 : 0]),
    );
  } else {
    policy = posteriorBestPolicy(rootPosteriors, actions, POLICY_SAMPLES);
    bestAction = bestPolicyAction(policy, actions);
  }

  return {
    root: rootTree,
    valueAlpha: rootTree.cache.slice(),
    valueOutcome: rootTree.outcome,
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

function formatPct(value) {
  return `${Math.round(value * 100)}%`;
}

function render() {
  analysis = analyzePosition();
  renderComputeControl();
  renderBoard();
  renderStatus();
  renderPosterior();
}

function renderSearchProgress() {
  analysis = analyzePosition();
  renderStatus();
  renderPosterior();
}

function renderComputeControl() {
  els.computeSlider.value = String(samplesPerSecond);
  els.computeValue.textContent =
    samplesPerSecond === 0 ? "paused" : `${samplesPerSecond}/s`;
}

function renderBoard() {
  const result = gameResult(board);
  els.board.dataset.player = currentPlayer;
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
  const posterior = { alpha, outcome: analysis.valueOutcome };
  const mean = posteriorMean(posterior);
  const xWinProbability =
    currentPlayer === "X" ? mean[OUTCOME.WIN] : mean[OUTCOME.LOSS];
  const oWinProbability =
    currentPlayer === "O" ? mean[OUTCOME.WIN] : mean[OUTCOME.LOSS];

  setBar(els.xWinBar, els.xWinPct, xWinProbability);
  setBar(els.drawBar, els.drawPct, mean[OUTCOME.DRAW]);
  setBar(els.oWinBar, els.oWinPct, oWinProbability);
  drawSimplex(alpha, posterior.outcome);
}

function setBar(bar, label, value) {
  bar.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
  label.textContent = formatPct(value);
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
      completedSamples += runSearch(rootTree, samplesToRun);
      sampleCarry -= samplesToRun;

      if (timestamp - lastRenderTime >= RENDER_INTERVAL_MS) {
        // Search updates must not recreate the cell buttons. Replacing a
        // button between pointer-down and pointer-up causes its click to be
        // discarded, which made fast taps appear unreliable.
        renderSearchProgress();
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

function drawSimplex(alpha, categoricalOutcome = null) {
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
  const lossLabel = currentPlayer === "X" ? "O" : "X";
  const winLabel = currentPlayer;
  context.fillText("Draw", vertices.D.x, vertices.D.y - 10);
  context.fillText(`${lossLabel} wins`, vertices.L.x + 4, vertices.L.y + 18);
  context.fillText(`${winLabel} wins`, vertices.W.x - 4, vertices.W.y + 18);

  if (categoricalOutcome === null) {
    context.fillStyle = "rgba(31, 122, 140, 0.24)";
    for (let i = 0; i < PLOT_SAMPLES; i += 1) {
      const point = simplexPoint(sampleDirichlet(alpha), vertices);
      context.beginPath();
      context.arc(point.x, point.y, 2.2, 0, Math.PI * 2);
      context.fill();
    }
  }

  const mean =
    categoricalOutcome === null
      ? alphaMean(alpha)
      : categoricalMean(categoricalOutcome);
  const meanPoint = simplexPoint(mean, vertices);
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
    drawSimplex(analysis.valueAlpha, analysis.valueOutcome);
  }
});

resetAnalysisTree();
render();
window.requestAnimationFrame(searchTick);
