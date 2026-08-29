(() => {
  const mockChessPositions = [
    {
      title: "The Immortal Game",
      context: "Anderssen–Kieseritzky · after 18…Bxg1",
      fen: "rnb1k1nr/p2p1ppp/3B4/1p1N1N1P/4P1P1/3P1Q2/PqP5/R4Kb1 w kq - 0 19",
      source: "https://lichess.org/study/KhHUn7L4/rNe22qqF",
      alpha: 14.1,
      beta: 5.9
    },
    {
      title: "The Opera Game",
      context: "Morphy–Duke Karl / Count Isouard · after 12…Rd8",
      fen: "3rkb1r/p2nqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR3R w k - 3 13",
      source: "https://lichess.org/study/KhHUn7L4/W2Shmeup",
      alpha: 32.5,
      beta: 17.5
    },
    {
      title: "The Evergreen Game",
      context: "Anderssen–Dufresne · after 18…Rg8",
      fen: "1r2k1r1/pbppnp1p/1bn2P2/7q/Q7/B1PB1N2/P4PPP/R3R1K1 w - - 1 19",
      source: "https://lichess.org/study/KhHUn7L4/ylEXCmyO",
      alpha: 5.8,
      beta: 4.2
    },
    {
      title: "The Game of the Century",
      context: "Donald Byrne–Fischer · after 17.Kf1",
      fen: "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R b - - 3 17",
      source: "https://lichess.org/study/KhHUn7L4/6Zdr6YYl",
      alpha: 8.4,
      beta: 19.6
    },
    {
      title: "Fischer's Deep Combination",
      context: "Robert Byrne–Fischer · after 15…Nxf2",
      fen: "r2qr1k1/p4pbp/bp3np1/3p4/8/BPN1P1P1/P1Q1NnBP/R2R2K1 w - - 0 16",
      source: "https://lichess.org/study/KhHUn7L4/QkgPOZ6j",
      alpha: 3.2,
      beta: 10.8
    },
    {
      title: "The King Hunt",
      context: "Edward Lasker–Thomas · after 10…Qe7",
      fen: "rn3rk1/pbppq1pp/1p2pb2/4N2Q/3PN3/3B4/PPP2PPP/R3K2R w KQ - 6 11",
      source: "https://lichess.org/study/KhHUn7L4/XHrjVEFf",
      alpha: 40,
      beta: 11
    },
    {
      title: "The Immortal Draw",
      context: "Hamppe–Meitner · after 8…Na6",
      fen: "r1b1k1nr/ppp2ppp/n7/3pp3/N3q3/1K6/PPPP2PP/R1BQ1BNR w kq - 2 9",
      source: "https://lichess.org/study/KhHUn7L4/1GkIYTTF",
      alpha: 2.7,
      beta: 2.9
    },
    {
      title: "Morphy's Counterattack",
      context: "Bird–Morphy · after 16…Rb8",
      fen: "1rb2rk1/p1p3pp/2pb4/3p4/3Pp3/4B2q/PPPQBP1P/R3K2R w KQ - 2 17",
      source: "https://lichess.org/study/KhHUn7L4/i5aUWe1s",
      alpha: 14,
      beta: 25
    },
    {
      title: "The Immortal Zugzwang",
      context: "Sämisch–Nimzowitsch · after 24…Bd3",
      fen: "6k1/3q2pp/p2bp3/3p1r2/1p1Pp3/3bQ1PP/PP1B1rB1/1NR3RK w - - 6 25",
      source: "https://lichess.org/study/KhHUn7L4/muWR4e04",
      alpha: 2.8,
      beta: 12.2
    },
    {
      title: "Rubinstein's Immortal",
      context: "Rotlewi–Rubinstein · after 21…Qh4",
      fen: "2rr2k1/1b3ppp/pb2p3/1p2P3/1P2BPnq/P1N5/1B2Q1PP/R4R1K w - - 5 22",
      source: "https://lichess.org/study/KhHUn7L4/cdOHySwq",
      alpha: 8.3,
      beta: 21.7
    }
  ];

  const datasetPositionLabels = ["1", "2", "3", "4", "5", "6", "7", "N−2", "N−1", "N"];

  function createSeededRandom(seed) {
    return () => {
      seed |= 0;
      seed = (seed + 0x6D2B79F5) | 0;
      let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleStandardNormal(random) {
    let u = 0;
    let v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function sampleGamma(shape, random) {
    if (shape < 1) {
      return sampleGamma(shape + 1, random) * Math.pow(random(), 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let normal;
      let transformed;
      do {
        normal = sampleStandardNormal(random);
        transformed = 1 + c * normal;
      } while (transformed <= 0);
      transformed **= 3;
      const uniform = random();
      if (
        uniform < 1 - 0.0331 * normal ** 4 ||
        Math.log(uniform) < 0.5 * normal ** 2 + d * (1 - transformed + Math.log(transformed))
      ) {
        return d * transformed;
      }
    }
  }

  function sampleBeta(alpha, beta, random) {
    const alphaSample = sampleGamma(alpha, random);
    const betaSample = sampleGamma(beta, random);
    return alphaSample / (alphaSample + betaSample);
  }

  const oddsRandom = createSeededRandom(0xC0FFEE);
  mockChessPositions.forEach((position) => {
    position.trueOddsA = Number((sampleBeta(position.alpha, position.beta, oddsRandom) * 100).toFixed(1));
  });
  
  const chessGlyphs = {
    K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟",
    k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟"
  };
  
  function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function renderFormula(tex) {
    const span = document.createElement("span");
    span.className = "coin-formula";
    if (window.katex) {
      window.katex.render(tex, span, { throwOnError: false });
    } else {
      span.textContent = tex;
    }
    return span;
  }
  
  function renderChessBoard(position, accessibleLabel = "Chess board position") {
    const rows = position.fen.split(" ")[0].split("/");
    if (rows.length !== 8) throw new Error(`Invalid FEN for ${position.title}`);
  
    const svg = svgElement("svg", {
      class: "chess-board",
      viewBox: "0 0 80 80",
      role: "img",
      "aria-label": accessibleLabel
    });
    const title = svgElement("title");
    title.textContent = accessibleLabel;
    svg.appendChild(title);
  
    for (let rank = 0; rank < 8; rank += 1) {
      for (let file = 0; file < 8; file += 1) {
        svg.appendChild(svgElement("rect", {
          x: file * 10,
          y: rank * 10,
          width: 10,
          height: 10,
          class: (file + rank) % 2 === 0 ? "chess-square chess-square--light" : "chess-square chess-square--dark"
        }));
      }
    }
  
    rows.forEach((row, rank) => {
      let file = 0;
      for (const token of row) {
        if (/\d/.test(token)) {
          file += Number(token);
          continue;
        }
        if (!chessGlyphs[token] || file > 7) throw new Error(`Invalid FEN for ${position.title}`);
        const piece = svgElement("text", {
          x: file * 10 + 5,
          y: rank * 10 + 5.35,
          class: token === token.toUpperCase() ? "chess-piece chess-piece--white" : "chess-piece chess-piece--black"
        });
        piece.textContent = chessGlyphs[token];
        svg.appendChild(piece);
        file += 1;
      }
      if (file !== 8) throw new Error(`Invalid FEN for ${position.title}`);
    });
  
    return svg;
  }
  
  function renderBetaPlot(position) {
    const left = 16;
    const right = 230;
    const top = 9;
    const baseline = 83;
    const modelMean = position.alpha / (position.alpha + position.beta);
    const samples = Array.from({ length: 101 }, (_, index) => {
      const x = 0.001 + (0.998 * index) / 100;
      const logDensity = (position.alpha - 1) * Math.log(x) + (position.beta - 1) * Math.log(1 - x);
      return { x, logDensity };
    });
    const maxLogDensity = Math.max(...samples.map((sample) => sample.logDensity));
    const points = samples.map((sample) => ({
      x: left + sample.x * (right - left),
      y: baseline - Math.exp(sample.logDensity - maxLogDensity) * (baseline - top)
    }));
  
    const svg = svgElement("svg", {
      class: "beta-plot",
      viewBox: "0 0 240 116",
      role: "img",
      "aria-label": `Beta distribution estimating a ${Math.round(modelMean * 100)} percent chance that player A wins`
    });
    const title = svgElement("title");
    title.textContent = `Beta(${position.alpha}, ${position.beta}) over the probability that player A wins.`;
    svg.appendChild(title);
  
    const pointPath = points.map((point) => `${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" L ");
    const targetX = left + (position.trueOddsA / 100) * (right - left);
    const meanX = left + modelMean * (right - left);
    svg.append(
      svgElement("line", { x1: left, y1: baseline, x2: right, y2: baseline, class: "beta-axis" }),
      svgElement("path", { d: `M ${left} ${baseline} L ${pointPath} L ${right} ${baseline} Z`, class: "beta-area" }),
      svgElement("path", { d: `M ${pointPath}`, class: "beta-curve" }),
      svgElement("line", { x1: targetX, y1: top, x2: targetX, y2: baseline, class: "beta-marker beta-marker--target" }),
      svgElement("line", { x1: meanX, y1: top, x2: meanX, y2: baseline, class: "beta-marker beta-marker--mean" })
    );
  
    [[left, "0"], [(left + right) / 2, ".5"], [right, "1"]].forEach(([x, label]) => {
      const tick = svgElement("text", { x, y: 97, class: "beta-tick" });
      tick.textContent = label;
      svg.appendChild(tick);
    });
    const axisLabel = svgElement("text", { x: (left + right) / 2, y: 112, class: "beta-axis-label" });
    axisLabel.textContent = "p(A wins)";
    svg.appendChild(axisLabel);
    return svg;
  }
  
  function renderCoinDataset() {
    const root = document.getElementById("coin-dataset-grid");
    if (!root) return;
  
    const rail = document.createElement("div");
    rail.className = "coin-dataset__rail";
    rail.setAttribute("role", "tablist");
    rail.setAttribute("aria-label", "Choose a chess position");
  
    const detail = document.createElement("article");
    detail.className = "coin-detail";
    detail.id = "coin-position-panel";
    detail.setAttribute("role", "tabpanel");
    const buttons = [];
  
    function showPosition(index, moveFocus = false) {
      const position = mockChessPositions[index];
      const modelMean = position.alpha / (position.alpha + position.beta);
      buttons.forEach((button, buttonIndex) => {
        const selected = buttonIndex === index;
        button.setAttribute("aria-selected", selected ? "true" : "false");
        button.tabIndex = selected ? 0 : -1;
      });
      detail.setAttribute("aria-labelledby", buttons[index].id);
      detail.replaceChildren();
  
      const header = document.createElement("header");
      header.className = "coin-detail__header";
      const sampleNumber = document.createElement("span");
      sampleNumber.className = "coin-sample__number";
      sampleNumber.textContent = `Sample ${datasetPositionLabels[index]}`;
      header.appendChild(sampleNumber);
  
      const body = document.createElement("div");
      body.className = "coin-detail__body";
      const boardColumn = document.createElement("div");
      boardColumn.className = "coin-detail__board";
      boardColumn.appendChild(renderChessBoard(position, `Chess position for ${sampleNumber.textContent}`));
  
      const estimate = document.createElement("div");
      estimate.className = "coin-sample__estimate";
      const oddsLabel = document.createElement("span");
      oddsLabel.className = "coin-sample__eyebrow";
      oddsLabel.append("empirical odds from repeated games ", renderFormula("p_A(s)"));
      const odds = document.createElement("div");
      odds.className = "coin-sample__odds";
      odds.innerHTML = `<span>A wins ${position.trueOddsA.toFixed(1)}%</span><span>B wins ${(100 - position.trueOddsA).toFixed(1)}%</span>`;
      const oddsBar = document.createElement("div");
      oddsBar.className = "coin-sample__odds-bar";
      const oddsFill = document.createElement("span");
      oddsFill.style.width = `${position.trueOddsA}%`;
      oddsBar.appendChild(oddsFill);
      const model = document.createElement("p");
      model.className = "coin-sample__model";
      model.innerHTML = `model prediction: Beta(${position.alpha}, ${position.beta})<br>mean estimate for A: <strong>${(modelMean * 100).toFixed(1)}%</strong>`;
      const key = document.createElement("div");
      key.className = "coin-sample__key";
      [
        ["density", "\\operatorname{Beta}(\\,\\cdot\\mid\\theta)"],
        ["target", "p_A(s)"],
        ["mean", "\\mathbb{E}[p\\mid\\theta]"]
      ].forEach(([className, formula]) => {
        const item = document.createElement("span");
        item.className = className;
        item.appendChild(renderFormula(formula));
        key.appendChild(item);
      });
      estimate.append(oddsLabel, odds, oddsBar, model, renderBetaPlot(position), key);
  
      body.append(boardColumn, estimate);
      detail.append(header, body);
      if (moveFocus) buttons[index].focus();
    }
  
    mockChessPositions.forEach((position, index) => {
      if (index === 7) {
        const ellipsis = document.createElement("span");
        ellipsis.className = "coin-dataset__ellipsis";
        ellipsis.setAttribute("aria-hidden", "true");
        ellipsis.textContent = "···";
        rail.appendChild(ellipsis);
      }

      const button = document.createElement("button");
      button.className = "coin-position-button";
      button.type = "button";
      button.id = `coin-position-tab-${index}`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", detail.id);
      const datasetPositionLabel = datasetPositionLabels[index];
      button.setAttribute("aria-label", `Select sample ${datasetPositionLabel}`);
      const board = renderChessBoard(position);
      board.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.className = "coin-position-button__label";
      label.textContent = datasetPositionLabel;
      button.append(board, label);
      button.addEventListener("click", () => showPosition(index));
      button.addEventListener("keydown", (event) => {
        let nextIndex = index;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") nextIndex = (index + 1) % mockChessPositions.length;
        if (event.key === "ArrowLeft" || event.key === "ArrowUp") nextIndex = (index - 1 + mockChessPositions.length) % mockChessPositions.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = mockChessPositions.length - 1;
        if (nextIndex !== index) {
          event.preventDefault();
          showPosition(nextIndex, true);
          buttons[nextIndex].scrollIntoView({ block: "nearest", inline: "nearest" });
        }
      });
      buttons.push(button);
      rail.appendChild(button);
    });
  
    root.append(rail, detail);
    showPosition(0);
  }

  document.addEventListener("DOMContentLoaded", renderCoinDataset);
})();
