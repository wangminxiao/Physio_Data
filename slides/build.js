const pptx = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// ---------- palette ----------
const NAVY  = "0F2A43";
const NAVY2 = "1B3D5C";
const NAVY3 = "27506F";
const MINT  = "2EC4B6";
const CORAL = "FF6B5C";  // RESERVED: gate / human-check semantic only
const AMBER = "F4A24E";
const PAPER = "F4F7FA";
const CARD  = "FFFFFF";
const INK   = "16222E";
const MUTE  = "5C6B78";
const LINEC = "D9E2EA";
const SOFT  = "EAF1F6";

// ---------- section system (signature colour per part) ----------
const SECTIONS = {
  design: { name: "Design", color: "0E8388", light: "7FD8D0", no: "1" },
  build:  { name: "Build",  color: "4F5BD5", light: "AEB6F0", no: "2" },
  use:    { name: "Use",    color: "CC7A33", light: "F0B878", no: "3" },
};
const TEAL = SECTIONS.design.color; // alias

const HF = "Trebuchet MS";
const BF = "Calibri";
const MF = "Consolas";

// ---------- icons ----------
function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}
async function iconPng(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICONS = {};
async function buildIcons() {
  const spec = {
    search: fa.FaSearch, design: fa.FaDraftingCompass, puzzle: fa.FaPuzzlePiece,
    rocket: fa.FaRocket, wave: fa.FaWaveSquare, list: fa.FaListUl, link: fa.FaLink,
    bolt: fa.FaBolt, layers: fa.FaLayerGroup, db: fa.FaDatabase, check: fa.FaCheckDouble,
    code: fa.FaCode, branch: fa.FaCodeBranch, shuffle: fa.FaRandom, hand: fa.FaHandPaper,
    steps: fa.FaListOl, bulb: fa.FaLightbulb, gears: fa.FaCogs, book: fa.FaBookOpen,
    diagram: fa.FaProjectDiagram, flask: fa.FaFlask, pulse: fa.FaHeartbeat,
    shield: fa.FaShieldAlt, cubes: fa.FaCubes, lock: fa.FaLock,
  };
  for (const [k, C] of Object.entries(spec)) ICONS[k] = C;
}
const _cache = {};
async function ic(key, color) {
  const id = key + color;
  if (!_cache[id]) _cache[id] = await iconPng(ICONS[key], "#" + color, 256);
  return _cache[id];
}

// ---------- layout helpers ----------
const shadow = () => ({ type: "outer", color: "0F2A43", blur: 9, offset: 3, angle: 135, opacity: 0.16 });
const W = 10, H = 5.625, MX = 0.55;

let PRES = new pptx();
PRES.layout = "LAYOUT_16x9";
PRES.author = "Minxiao Wang";
PRES.title = "Designing, Building & Using the physio-data Skill";

function footer(slide, n, dark) {
  const c = dark ? "8FA6B8" : MUTE;
  slide.addText("physio-data  ·  Claude Code skills for research", {
    x: MX, y: H - 0.36, w: 6, h: 0.28, fontSize: 8.5, color: c, fontFace: BF, align: "left", margin: 0,
  });
  slide.addText(String(n), {
    x: W - 1.0, y: H - 0.36, w: 0.45, h: 0.28, fontSize: 9, color: c, fontFace: BF, align: "right", margin: 0,
  });
}

// progress tracker: Design › Build › Use  (active pill lit in its colour)
function tracker(slide, activeKey) {
  const order = ["design", "build", "use"];
  const pillW = 1.0, gap = 0.1, h = 0.28, y = 0.4;
  const startX = (W - MX) - (3 * pillW + 2 * gap); // ends flush at right margin
  order.forEach((k, i) => {
    const x = startX + i * (pillW + gap);
    const active = k === activeKey;
    slide.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
      x, y, w: pillW, h, rectRadius: 0.14,
      fill: { color: active ? SECTIONS[k].color : "E6ECF2" }, line: { type: "none" },
    });
    slide.addText(SECTIONS[k].no + "  " + SECTIONS[k].name, {
      x, y, w: pillW, h, align: "center", valign: "middle",
      fontSize: 10, bold: true, color: active ? "FFFFFF" : "93A2AE", fontFace: HF, margin: 0,
    });
  });
}

// header: role kicker (section colour) + title + one section-coloured key-line + tracker
function head(slide, kicker, ttl, sectionKey, keyline) {
  const C = sectionKey ? SECTIONS[sectionKey].color : TEAL;
  slide.addText(kicker.toUpperCase(), {
    x: MX, y: 0.4, w: 5, h: 0.3, fontSize: 12, bold: true, color: C, fontFace: HF, charSpacing: 2, margin: 0,
  });
  slide.addText(ttl, {
    x: MX, y: 0.72, w: W - 2 * MX, h: 0.5, fontSize: 25, bold: true, color: NAVY, fontFace: HF, margin: 0,
  });
  if (keyline) slide.addText(keyline, {
    x: MX, y: 1.19, w: W - 2 * MX, h: 0.34, fontSize: 13.5, italic: true, bold: true, color: C, fontFace: BF, margin: 0,
  });
  if (sectionKey) tracker(slide, sectionKey);
}

async function iconChip(slide, key, x, y, d, circleFill, iconColor) {
  slide.addShape(PRES.shapes.OVAL, { x, y, w: d, h: d, fill: { color: circleFill }, line: { type: "none" } });
  const inset = d * 0.26;
  slide.addImage({ data: await ic(key, iconColor), x: x + inset, y: y + inset, w: d - 2 * inset, h: d - 2 * inset });
}

function card(slide, x, y, w, h, fill = CARD) {
  slide.addShape(PRES.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.09, fill: { color: fill }, line: { color: LINEC, width: 1 }, shadow: shadow(),
  });
}

// filled section-colour takeaway bar (used only on the two meta-lesson slides)
async function bigTakeaway(slide, sectionKey, text) {
  const C = SECTIONS[sectionKey].color, y = 4.42, h = 0.62;
  slide.addShape(PRES.shapes.RECTANGLE, { x: MX, y, w: W - 2 * MX, h, fill: { color: C }, line: { type: "none" } });
  slide.addImage({ data: await ic("bulb", "FFFFFF"), x: MX + 0.24, y: y + 0.17, w: 0.3, h: 0.3 });
  slide.addText([
    { text: "TAKEAWAY   ", options: { bold: true, color: "FFFFFF", charSpacing: 1, fontSize: 10.5 } },
    { text, options: { color: "FFFFFF" } },
  ], { x: MX + 0.7, y, w: W - 2 * MX - 0.9, h, fontSize: 13, bold: true, fontFace: BF, valign: "middle", margin: 0 });
}

// =====================================================================
// SLIDE 1 — TITLE
// =====================================================================
async function slideTitle() {
  const s = PRES.addSlide();
  s.background = { color: NAVY };
  s.addImage({ path: "assets/ppg_white.png", x: -0.5, y: 0.12, w: 11, h: 0.6, transparency: 82 });
  s.addImage({ path: "assets/ecg_white.png", x: 4.7, y: 4.82, w: 6.0, h: 0.66, transparency: 80 });

  s.addText("A CASE STUDY IN DOING RESEARCH WITH CLAUDE CODE", {
    x: MX, y: 0.95, w: 9, h: 0.35, fontSize: 13, bold: true, color: MINT, fontFace: HF, charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "From One-Off Scripts", options: { breakLine: true, color: "FFFFFF" } },
    { text: "to a Reusable Skill", options: { color: MINT } },
  ], { x: MX, y: 1.45, w: 9, h: 1.7, fontSize: 46, bold: true, fontFace: HF, lineSpacingMultiple: 1.0, margin: 0 });

  s.addText("Designing, building & using the  physio-data  skill — turning a recurring research task into something Claude Code does the same way every time.", {
    x: MX, y: 3.25, w: 8.4, h: 0.8, fontSize: 15, color: "CFE0EC", fontFace: BF, margin: 0,
  });

  s.addShape(PRES.shapes.RECTANGLE, { x: MX, y: 4.55, w: 0.06, h: 0.5, fill: { color: CORAL }, line: { type: "none" } });
  s.addText([
    { text: "Minxiao Wang", options: { bold: true, color: "FFFFFF", breakLine: true, fontSize: 14 } },
    { text: "Emory University  ·  2026-06-10", options: { color: "9FB8CB", fontSize: 11 } },
  ], { x: MX + 0.18, y: 4.5, w: 6, h: 0.65, fontFace: BF, valign: "middle", margin: 0 });

  s.addNotes("BRIDGE: (open here). I'll use one real skill I built — physio-data — to teach two habits: using Claude Code as a research partner, and packaging recurring work into a reusable skill. Three parts: design, build, use.");
}

// =====================================================================
// SLIDE 2 — OVERVIEW
// =====================================================================
async function slideOverview() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Overview", "What you'll take away", null,
    "Two habits to steal — and one skill that shows them in action.");

  const cards = [
    { key: "search", c: TEAL, t: "Use Claude Code\nfor research", d: "A thinking + building partner: explore data, argue design trade-offs, then write code." },
    { key: "puzzle", c: SECTIONS.build.color, t: "Capture recurring\nwork as a skill", d: "Same shape of problem, again and again? Write it down once so Claude repeats it reliably." },
    { key: "pulse", c: SECTIONS.use.color, t: "The case study:\nphysio-data", d: "One canonical format for messy physiological datasets — already used on 7 of them." },
  ];
  const cw = 2.80, gap = 0.25, x0 = MX, y0 = 1.72, ch = 2.4;
  for (let i = 0; i < cards.length; i++) {
    const x = x0 + i * (cw + gap);
    card(s, x, y0, cw, ch);
    await iconChip(s, cards[i].key, x + 0.28, y0 + 0.28, 0.78, SOFT, cards[i].c);
    s.addText(cards[i].t, { x: x + 0.28, y: y0 + 1.16, w: cw - 0.56, h: 0.7, fontSize: 16.5, bold: true, color: NAVY, fontFace: HF, margin: 0, lineSpacingMultiple: 0.95 });
    s.addText(cards[i].d, { x: x + 0.28, y: y0 + 1.82, w: cw - 0.56, h: 0.5, fontSize: 11.5, color: MUTE, fontFace: BF, margin: 0 });
  }

  // roadmap strip = preview of the journey
  const ry = 4.45;
  card(s, MX, ry, W - 2 * MX, 0.62, NAVY);
  const parts = [["design", "understand the problem"], ["build", "encode it as a skill"], ["use", "onboard new datasets"]];
  const pw = (W - 2 * MX) / 3;
  for (let i = 0; i < 3; i++) {
    const x = MX + i * pw, k = parts[i][0];
    s.addText([
      { text: SECTIONS[k].no + "  ", options: { color: SECTIONS[k].light, bold: true, fontSize: 15 } },
      { text: SECTIONS[k].name.toUpperCase() + "  ", options: { color: "FFFFFF", bold: true, fontSize: 14 } },
      { text: "· " + parts[i][1], options: { color: "9FB8CB", fontSize: 11 } },
    ], { x: x + 0.2, y: ry, w: pw - 0.3, h: 0.62, valign: "middle", fontFace: BF, margin: 0 });
    if (i < 2) s.addShape(PRES.shapes.LINE, { x: x + pw, y: ry + 0.12, w: 0, h: 0.38, line: { color: NAVY3, width: 1 } });
  }
  footer(s, 2, false);
  s.addNotes("BRIDGE: Here's the map for the whole talk. Two meta-lessons, one running example. The three colours at the bottom — teal, indigo, amber — are Design, Build, Use; you'll see them track across the top of every slide so you always know where we are.");
}

// =====================================================================
// SLIDE 3 — THE RECURRING PROBLEM
// =====================================================================
async function slideProblem() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Motivation", "The same job, over and over", null,
    "Same shape, different details — the signature of a task worth turning into a skill.");

  const y0 = 2.05, h = 2.1;
  const lw = 4.32, rw = 4.32, gap = 0.26;
  card(s, MX, y0, lw, h, CARD);
  s.addShape(PRES.shapes.RECTANGLE, { x: MX, y: y0, w: 0.07, h: h, fill: { color: TEAL }, line: { type: "none" } });
  await iconChip(s, "shuffle", MX + 0.26, y0 + 0.24, 0.6, SOFT, TEAL);
  s.addText("ALWAYS THE SAME", { x: MX + 0.98, y: y0 + 0.28, w: lw - 1.2, h: 0.3, fontSize: 12, bold: true, color: TEAL, fontFace: HF, charSpacing: 1, margin: 0, valign: "middle" });
  s.addText([
    "Waveforms (ECG/PPG/ABP) + clinical events",
    "Resample → segment into fixed windows",
    "Align events to signal segments",
    "Keep patients with BOTH signals & events",
    "Verify, then split train/val/test by subject",
  ].map((t) => ({ text: t, options: { bullet: { code: "2022" }, color: INK, fontSize: 11.5, breakLine: true, paraSpaceAfter: 5 } })),
    { x: MX + 0.3, y: y0 + 0.74, w: lw - 0.58, h: h - 0.9, fontFace: BF, margin: 0 });

  const rx = MX + lw + gap;
  card(s, rx, y0, rw, h, CARD);
  s.addShape(PRES.shapes.RECTANGLE, { x: rx, y: y0, w: 0.07, h: h, fill: { color: CORAL }, line: { type: "none" } });
  await iconChip(s, "db", rx + 0.26, y0 + 0.24, 0.6, "FBE7E4", CORAL);
  s.addText("DIFFERENT EVERY TIME", { x: rx + 0.98, y: y0 + 0.28, w: rw - 1.2, h: 0.3, fontSize: 12, bold: true, color: CORAL, fontFace: HF, charSpacing: 1, margin: 0, valign: "middle" });
  s.addText([
    "Raw format: WFDB · EDF · CSV · Parquet · XML",
    "Sample rates: 40 / 125 / 240 / 500 Hz …",
    "How patients/encounters link to signals",
    "Timezones, sentinels, missing-channel quirks",
    "Which labs / vitals / meds exist + their units",
  ].map((t) => ({ text: t, options: { bullet: { code: "2022" }, color: INK, fontSize: 11.5, breakLine: true, paraSpaceAfter: 5 } })),
    { x: rx + 0.3, y: y0 + 0.74, w: rw - 0.58, h: h - 0.9, fontFace: BF, margin: 0 });

  s.addText([
    { text: "Do it ad-hoc each time = repeated mistakes, no consistency, no checks.  ", options: { color: MUTE, italic: true } },
    { text: "That's exactly when a skill pays off.", options: { color: NAVY, bold: true } },
  ], { x: MX, y: y0 + h + 0.14, w: W - 2 * MX, h: 0.3, fontSize: 12, fontFace: BF, margin: 0 });

  footer(s, 3, false);
  s.addNotes("BRIDGE: Why build this at all? Spot the pattern — the left column never changes, the right column changes every dataset. Fixed shape, varying details. That gap is the trigger for a skill. So, part one: how I designed the solution.");
}

// =====================================================================
// SECTION DIVIDER (bridge + agenda + section colour)
// =====================================================================
async function divider(sectionKey, bridge, agenda, waveKind) {
  const s = PRES.addSlide();
  s.background = { color: NAVY };
  const S = SECTIONS[sectionKey];
  s.addImage({ path: `assets/${waveKind}.png`, x: -0.5, y: 4.5, w: 11, h: 0.9, transparency: 74 });
  tracker(s, sectionKey);

  s.addText("PART " + S.no, { x: MX, y: 1.15, w: 4, h: 0.36, fontSize: 14, bold: true, color: S.light, fontFace: HF, charSpacing: 3, margin: 0 });
  s.addShape(PRES.shapes.RECTANGLE, { x: MX, y: 1.66, w: 0.12, h: 1.04, fill: { color: S.color }, line: { type: "none" } });
  s.addText(S.name, { x: MX + 0.32, y: 1.52, w: 4.9, h: 1.1, fontSize: 50, bold: true, color: "FFFFFF", fontFace: HF, margin: 0 });
  s.addText(bridge, { x: MX + 0.34, y: 2.78, w: 4.95, h: 1.0, fontSize: 15, color: "CFE0EC", fontFace: BF, margin: 0 });

  // agenda card
  const ax = 6.05, aw = (W - MX) - ax;
  s.addShape(PRES.shapes.ROUNDED_RECTANGLE, { x: ax, y: 1.45, w: aw, h: 2.75, rectRadius: 0.08, fill: { color: NAVY2 }, line: { color: NAVY3, width: 1 } });
  s.addText("IN THIS PART", { x: ax + 0.26, y: 1.66, w: aw - 0.5, h: 0.3, fontSize: 11, bold: true, color: S.light, fontFace: HF, charSpacing: 2, margin: 0 });
  const step = Math.min(0.44, 2.05 / agenda.length);
  agenda.forEach((it, i) => {
    const yy = 2.06 + i * step;
    s.addText([
      { text: String(i + 1), options: { color: S.color, bold: true } },
      { text: "   " + it, options: { color: "D7E3EE" } },
    ], { x: ax + 0.26, y: yy, w: aw - 0.5, h: step, fontSize: 11.5, fontFace: BF, valign: "middle", margin: 0 });
  });
  return s;
}

// =====================================================================
// SLIDE — DESIGN: signals vs events
// =====================================================================
async function slideSignalsEvents() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Key insight", "Signals ≠ events", "design",
    "Different kinds of data deserve different storage.");

  const y0 = 1.72, h = 2.3, cw = 4.32, gap = 0.26;
  card(s, MX, y0, cw, h);
  await iconChip(s, "wave", MX + 0.28, y0 + 0.26, 0.66, SOFT, TEAL);
  s.addText("SIGNALS", { x: MX + 1.05, y: y0 + 0.3, w: cw - 1.3, h: 0.3, fontSize: 14, bold: true, color: TEAL, fontFace: HF, charSpacing: 1, valign: "middle", margin: 0 });
  s.addText("dense · regular · high-rate", { x: MX + 1.05, y: y0 + 0.62, w: cw - 1.3, h: 0.25, fontSize: 11, italic: true, color: MUTE, fontFace: BF, margin: 0 });
  s.addText([
    "Store as mmap-ready  .npy  arrays",
    "row = one fixed-length window",
    "float16, C-contiguous → zero-copy reads",
  ].map((t) => ({ text: t, options: { bullet: { code: "2022" }, color: INK, fontSize: 12.5, breakLine: true, paraSpaceAfter: 6 } })),
    { x: MX + 0.32, y: y0 + 1.1, w: cw - 0.6, h: 1.1, fontFace: BF, margin: 0 });

  const rx = MX + cw + gap;
  card(s, rx, y0, cw, h);
  await iconChip(s, "list", rx + 0.28, y0 + 0.26, 0.66, SOFT, AMBER);
  s.addText("EVENTS", { x: rx + 1.05, y: y0 + 0.3, w: cw - 1.3, h: 0.3, fontSize: 14, bold: true, color: "C77F2E", fontFace: HF, charSpacing: 1, valign: "middle", margin: 0 });
  s.addText("sparse · irregular · point-in-time", { x: rx + 1.05, y: y0 + 0.62, w: cw - 1.3, h: 0.25, fontSize: 11, italic: true, color: MUTE, fontFace: BF, margin: 0 });
  s.addText([
    "Store as a sparse structured array",
    "row = one measurement / annotation",
    "no dense padding, only what was recorded",
  ].map((t) => ({ text: t, options: { bullet: { code: "2022" }, color: INK, fontSize: 12.5, breakLine: true, paraSpaceAfter: 6 } })),
    { x: rx + 0.32, y: y0 + 1.1, w: cw - 0.6, h: 1.1, fontFace: BF, margin: 0 });

  const ay = y0 + h + 0.16;
  card(s, MX, ay, W - 2 * MX, 0.72, NAVY);
  await iconChip(s, "link", MX + 0.28, ay + 0.12, 0.48, NAVY3, MINT);
  s.addText([
    { text: "Alignment is an index, not a copy.  ", options: { color: "FFFFFF", bold: true } },
    { text: "Each event carries a  ", options: { color: "CFE0EC" } },
    { text: "seg_idx", options: { color: MINT, fontFace: MF } },
    { text: "  into the signal array — no duplication, no padding.", options: { color: "CFE0EC" } },
  ], { x: MX + 0.92, y: ay, w: W - 2 * MX - 1.1, h: 0.72, fontSize: 12.5, fontFace: BF, valign: "middle", margin: 0 });

  footer(s, 4, false);
  s.addNotes("BRIDGE: The design starts from one idea — signals and events have opposite statistics, so they get opposite storage, linked by an index. Next: the principles that fell out of this.");
}

// =====================================================================
// SLIDE — DESIGN: five principles
// =====================================================================
async function slidePrinciples() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Principles", "Five rules, written before any code", "design",
    "Decide the rules once, up front — they settle a hundred later choices.");

  const items = [
    { key: "shuffle", t: "Different data, different storage", d: "Signals → dense mmap arrays; events → sparse records. Don't force one shape." },
    { key: "link", t: "Alignment is an index", d: "Events point into segments via seg_idx. No copies, no dense padding." },
    { key: "bolt", t: "Zero CPU on the hot path", d: "True mmap, no compression. GPU-server CPUs are weak — keep reads zero-copy." },
    { key: "layers", t: "Extensible without breakage", d: "New variable = new row; new channel = new file. Old shapes never change." },
    { key: "db", t: "Raw values in storage", d: "No normalization / interpolation at rest — those are runtime choices." },
  ];
  const colW = 4.32, gap = 0.26, rowH = 0.84, y0 = 1.62;
  for (let i = 0; i < items.length; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (colW + gap), y = y0 + row * (rowH + 0.12);
    card(s, x, y, colW, rowH);
    await iconChip(s, items[i].key, x + 0.2, y + 0.17, 0.5, SOFT, TEAL);
    s.addText(items[i].t, { x: x + 0.82, y: y + 0.1, w: colW - 1.0, h: 0.32, fontSize: 13, bold: true, color: NAVY, fontFace: HF, margin: 0, valign: "middle" });
    s.addText(items[i].d, { x: x + 0.82, y: y + 0.4, w: colW - 1.0, h: 0.38, fontSize: 10.5, color: MUTE, fontFace: BF, margin: 0 });
  }
  const x = MX + (colW + gap), y = y0 + 2 * (rowH + 0.12);
  card(s, x, y, colW, rowH, NAVY);
  await iconChip(s, "design", x + 0.2, y + 0.17, 0.5, NAVY3, MINT);
  s.addText([
    { text: "Design first, code later.  ", options: { color: "FFFFFF", bold: true, fontSize: 12.5 } },
    { text: "All five were written before any extraction code.", options: { color: "BCD2E2", fontSize: 10.5 } },
  ], { x: x + 0.82, y: y, w: colW - 1.0, h: rowH, fontFace: BF, valign: "middle", margin: 0 });

  footer(s, 5, false);
  s.addNotes("BRIDGE: Principles are cheap to write, expensive to skip. Each one became a hard rule inside the skill, so Claude never re-argues them. Now — what they produce: the format itself.");
}

// =====================================================================
// SLIDE — DESIGN: canonical format
// =====================================================================
async function slideFormat() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "The format", "One canonical format for all datasets", "design",
    "One layout fits every dataset: two channels + one universal event record.");

  const lx = MX, ly = 1.7, lw = 4.75, lh = 2.95;
  card(s, lx, ly, lw, lh, NAVY);
  s.addText("{entity_id}/   one patient-encounter", { x: lx + 0.28, y: ly + 0.16, w: lw - 0.5, h: 0.3, fontSize: 11, italic: true, color: MINT, fontFace: MF, margin: 0 });
  s.addText([
    { text: "PLETH40.npy", options: { color: "FFFFFF", fontFace: MF, breakLine: true } },
    { text: "   [N_seg, 1200]  PPG @ 40 Hz  (base)", options: { color: "8FB8C9", fontFace: MF, fontSize: 9.5, breakLine: true } },
    { text: "II120.npy", options: { color: "FFFFFF", fontFace: MF, breakLine: true } },
    { text: "   [N_seg, 3600]  ECG @ 120 Hz  (NaN if absent)", options: { color: "8FB8C9", fontFace: MF, fontSize: 9.5, breakLine: true } },
    { text: "time_ms.npy", options: { color: "FFFFFF", fontFace: MF, breakLine: true } },
    { text: "   [N_seg]  int64, monotonic", options: { color: "8FB8C9", fontFace: MF, fontSize: 9.5, breakLine: true } },
    { text: "ehr_events.npy", options: { color: "FFFFFF", fontFace: MF, breakLine: true } },
    { text: "   [N_events]  sparse structured array", options: { color: "8FB8C9", fontFace: MF, fontSize: 9.5, breakLine: true } },
    { text: "meta.json", options: { color: "FFFFFF", fontFace: MF, breakLine: true } },
    { text: "   manifest + provenance", options: { color: "8FB8C9", fontFace: MF, fontSize: 9.5 } },
  ], { x: lx + 0.3, y: ly + 0.54, w: lw - 0.5, h: lh - 0.66, fontSize: 11, fontFace: MF, margin: 0, lineSpacingMultiple: 0.98 });

  const rx = MX + lw + 0.28, rw = W - MX - rx, ry = 1.7;
  card(s, rx, ry, rw, 1.3);
  s.addText("Every event = one tiny record", { x: rx + 0.24, y: ry + 0.13, w: rw - 0.4, h: 0.28, fontSize: 12, bold: true, color: NAVY, fontFace: HF, margin: 0 });
  s.addText("( time_ms , seg_idx , var_id , value )", { x: rx + 0.24, y: ry + 0.47, w: rw - 0.4, h: 0.32, fontSize: 13, bold: true, color: TEAL, fontFace: MF, margin: 0 });
  s.addText("sorted by time · seg_idx links to a window · var_id looks up name/unit in a shared registry", {
    x: rx + 0.24, y: ry + 0.79, w: rw - 0.45, h: 0.45, fontSize: 10, color: MUTE, fontFace: BF, margin: 0 });

  const by = ry + 1.45;
  card(s, rx, by, rw, 1.5);
  s.addText("var_id ranges encode category", { x: rx + 0.24, y: by + 0.13, w: rw - 0.4, h: 0.28, fontSize: 12, bold: true, color: NAVY, fontFace: HF, margin: 0 });
  const ranges = [["0–99", "Labs", TEAL], ["100–199", "Vitals", MINT], ["200–299", "Actions", AMBER], ["300–399", "Scores", CORAL]];
  for (let i = 0; i < ranges.length; i++) {
    const yy = by + 0.48 + i * 0.235;
    s.addShape(PRES.shapes.RECTANGLE, { x: rx + 0.26, y: yy + 0.02, w: 0.16, h: 0.16, fill: { color: ranges[i][2] }, line: { type: "none" } });
    s.addText([
      { text: ranges[i][0].padEnd(9, " "), options: { fontFace: MF, color: NAVY, bold: true } },
      { text: "  " + ranges[i][1], options: { color: INK } },
    ], { x: rx + 0.52, y: yy - 0.04, w: rw - 0.7, h: 0.26, fontSize: 11, fontFace: BF, valign: "middle", margin: 0 });
  }

  footer(s, 6, false);
  s.addNotes("BRIDGE: This is the contract — the thing the skill will guarantee on every dataset. Two channels, monotonic time, one universal event record. One more design point, and it's the most important habit: how you actually arrive at a design like this with Claude.");
}

// =====================================================================
// SLIDE — DESIGN: meta-lesson #1 (design WITH Claude Code)
// =====================================================================
async function slideDesignWithCC() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Meta-lesson #1", "Design it WITH Claude Code", "design",
    "Explore and design with Claude before you ask it for code.");

  const steps = [
    { key: "book", t: "Research first", d: "Read docs & papers, explore the raw files. Claude searches, reads, runs probes with you." },
    { key: "design", t: "Write a design doc", d: "Make Claude argue trade-offs in a PLAN.md before code. Decisions get recorded, not re-debated." },
    { key: "flask", t: "Demo, then verify", d: "Build one entity end-to-end, plot signal + event alignment, eyeball it before scaling." },
  ];
  const cw = 2.80, gap = 0.25, y0 = 1.85, ch = 2.05;
  for (let i = 0; i < 3; i++) {
    const x = MX + i * (cw + gap);
    card(s, x, y0, cw, ch);
    s.addText(String(i + 1), { x: x + cw - 0.78, y: y0 + 0.16, w: 0.6, h: 0.6, fontSize: 30, bold: true, color: SOFT, fontFace: HF, align: "right", margin: 0 });
    await iconChip(s, steps[i].key, x + 0.26, y0 + 0.28, 0.62, SOFT, TEAL);
    s.addText(steps[i].t, { x: x + 0.26, y: y0 + 1.05, w: cw - 0.5, h: 0.35, fontSize: 15, bold: true, color: NAVY, fontFace: HF, margin: 0 });
    s.addText(steps[i].d, { x: x + 0.26, y: y0 + 1.42, w: cw - 0.5, h: 0.55, fontSize: 11, color: MUTE, fontFace: BF, margin: 0 });
  }

  await bigTakeaway(s, "design", "Don't open with “write me a script.” Explore and design with Claude first.");
  footer(s, 7, false);
  s.addNotes("BRIDGE: This is the first thing to steal. The biggest leverage is using Claude to THINK with you — read the data, profile it, argue the format — before any code. The one-patient demo with a visual check is the cheapest bug-catcher you'll ever build. That ends design. Part two: turning all of this into a skill.");
}

// =====================================================================
// SLIDE — BUILD: what is a skill
// =====================================================================
async function slideWhatIsSkill() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Anatomy", "A skill is just a SKILL.md file", "build",
    "A markdown file Claude auto-loads when its description matches your request.");

  const cx = MX, cy = 1.72, cw2 = 5.35, ch2 = 2.5;
  card(s, cx, cy, cw2, ch2, NAVY);
  s.addText("SKILL.md", { x: cx + 0.28, y: cy + 0.14, w: 3, h: 0.3, fontSize: 11, italic: true, color: "8FB8C9", fontFace: MF, margin: 0 });
  s.addText([
    { text: "---", options: { color: "6E8DA0", breakLine: true } },
    { text: "name:", options: { color: MINT, breakLine: false } },
    { text: " physio-data", options: { color: "FFFFFF", breakLine: true } },
    { text: "description:", options: { color: MINT } },
    { text: " Onboard & preprocess", options: { color: "FFFFFF", breakLine: true } },
    { text: "  physiological time-series datasets into", options: { color: "FFFFFF", breakLine: true } },
    { text: "  a canonical mmap-ready format…", options: { color: "FFFFFF", breakLine: true } },
    { text: "argument-hint:", options: { color: MINT } },
    { text: " [dataset | \"explore\"…]", options: { color: "FFFFFF", breakLine: true } },
    { text: "---", options: { color: "6E8DA0", breakLine: true } },
    { text: "# Physio_Data Preprocessing Skill", options: { color: "FFD9A0", breakLine: true } },
    { text: "  → principles · format · workflow · gates", options: { color: "8FB8C9" } },
  ], { x: cx + 0.3, y: cy + 0.48, w: cw2 - 0.55, h: ch2 - 0.62, fontSize: 11.5, fontFace: MF, margin: 0, lineSpacingMultiple: 1.05 });

  const rx = cx + cw2 + 0.28, rw = W - MX - rx;
  const parts = [
    { c: CORAL, t: "description = the trigger", d: "The one field that decides whether Claude reaches for this skill. Write it for recall." },
    { c: SECTIONS.build.color, t: "name + argument-hint", d: "How it's invoked: /physio-data mimic3." },
    { c: SECTIONS.build.color, t: "body = the expertise", d: "Everything Claude should know & do — loaded only when relevant." },
  ];
  let yy = 1.72;
  for (const p of parts) {
    card(s, rx, yy, rw, 0.76);
    s.addShape(PRES.shapes.RECTANGLE, { x: rx, y: yy, w: 0.07, h: 0.76, fill: { color: p.c }, line: { type: "none" } });
    s.addText(p.t, { x: rx + 0.24, y: yy + 0.11, w: rw - 0.4, h: 0.3, fontSize: 12.5, bold: true, color: NAVY, fontFace: HF, margin: 0 });
    s.addText(p.d, { x: rx + 0.24, y: yy + 0.41, w: rw - 0.4, h: 0.3, fontSize: 10, color: MUTE, fontFace: BF, margin: 0 });
    yy += 0.86;
  }

  footer(s, 8, false);
  s.addNotes("BRIDGE: Let's demystify the word 'skill'. It's a markdown file with frontmatter. The description is the single most important line — it's the trigger. The body is only paid for, in tokens, when the skill fires. Next: what we actually put in that body.");
}

// =====================================================================
// SLIDE — BUILD: what goes in the body
// =====================================================================
async function slideEncoded() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "What's inside", "Inside the body: hard-won knowledge", "build",
    "A skill is your mistakes, written down as rules — enforced every run.");

  const items = [
    { key: "lock", t: "The format spec", d: "The canonical contract: dtypes, channels, the event record. Non-negotiable." },
    { key: "steps", t: "The ordered workflow", d: "Step 0 → Stages 1–5 → post-stages. \"Do not skip ahead.\"" },
    { key: "check", t: "Verification gates", d: "A check after every stage, with explicit abort conditions." },
    { key: "shield", t: "Hard constraints", d: "no compression · split by subject, not admission · raw values only." },
  ];
  const cw = 4.32, gap = 0.26, rowH = 1.18, y0 = 1.72;
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (cw + gap), y = y0 + row * (rowH + 0.16);
    card(s, x, y, cw, rowH);
    await iconChip(s, items[i].key, x + 0.24, y + 0.32, 0.58, SOFT, SECTIONS.build.color);
    s.addText(items[i].t, { x: x + 0.96, y: y + 0.22, w: cw - 1.16, h: 0.34, fontSize: 14, bold: true, color: NAVY, fontFace: HF, margin: 0, valign: "middle" });
    s.addText(items[i].d, { x: x + 0.96, y: y + 0.58, w: cw - 1.16, h: 0.5, fontSize: 11, color: MUTE, fontFace: BF, margin: 0 });
  }

  footer(s, 9, false);
  s.addNotes("BRIDGE: The body is everything you learned the hard way. Every abort condition maps to a real failure I hit. 'Split by subject, not admission' is there because admission splits leak data — a mistake you make only once if it's in the skill. Now let's see the workflow it enforces.");
}

// =====================================================================
// SLIDE — BUILD: the workflow as stages
// =====================================================================
async function slideWorkflow() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "The workflow", "The workflow, encoded as ordered stages", "build",
    "A fixed order, with a verification gate after every stage.");

  const y0 = 1.66;
  card(s, MX, y0, W - 2 * MX, 0.62, SOFT);
  await iconChip(s, "search", MX + 0.2, y0 + 0.07, 0.48, "FFFFFF", SECTIONS.build.color);
  s.addText([
    { text: "STEP 0   ", options: { bold: true, color: SECTIONS.build.color } },
    { text: "research  →  explore raw data  →  single-entity demo  →  write & review  API.md", options: { color: INK } },
  ], { x: MX + 0.84, y: y0, w: W - 2 * MX - 1, h: 0.62, fontSize: 12.5, fontFace: BF, valign: "middle", margin: 0 });

  const BC = SECTIONS.build.color;
  const stages = [
    ["1", "Scan\nsignals", BC, "have required channels?"],
    ["2", "Extract\nevents", BC, "→ target variables"],
    ["3", "Cross-\ncheck", CORAL, "keep BOTH (the gate)"],
    ["4", "Extract\nsignals", BC, "resample·segment·align"],
    ["5", "Manifest\n+ splits", BC, "split by subject"],
  ];
  const fy = 2.52, bw = 1.62, bh = 1.5, gapx = (W - 2 * MX - 5 * bw) / 4;
  for (let i = 0; i < 5; i++) {
    const x = MX + i * (bw + gapx);
    const hot = stages[i][2] === CORAL;
    card(s, x, fy, bw, bh, hot ? CORAL : CARD);
    s.addText(stages[i][0], { x: x + 0.16, y: fy + 0.12, w: 0.6, h: 0.5, fontSize: 24, bold: true, color: hot ? "FFFFFF" : BC, fontFace: HF, margin: 0 });
    s.addText(stages[i][1], { x: x + 0.16, y: fy + 0.58, w: bw - 0.3, h: 0.5, fontSize: 13.5, bold: true, color: hot ? "FFFFFF" : NAVY, fontFace: HF, margin: 0, lineSpacingMultiple: 0.9 });
    s.addText(stages[i][3], { x: x + 0.16, y: fy + 1.04, w: bw - 0.28, h: 0.42, fontSize: 8.7, color: hot ? "FFE3DF" : MUTE, fontFace: BF, margin: 0, lineSpacingMultiple: 0.95 });
    if (i < 4) s.addImage({ data: ARROW, x: x + bw + gapx / 2 - 0.11, y: fy + bh / 2 - 0.11, w: 0.22, h: 0.22 });
  }

  const vy = fy + bh + 0.18;
  card(s, MX, vy, W - 2 * MX, 0.55, NAVY);
  await iconChip(s, "check", MX + 0.2, vy + 0.06, 0.42, NAVY3, MINT);
  s.addText([
    { text: "Verify after every stage  ", options: { color: "FFFFFF", bold: true } },
    { text: "·  test with  --limit 5  before any full run  ·  Stage 3 aborts on zero overlap", options: { color: "BCD2E2" } },
  ], { x: MX + 0.78, y: vy, w: W - 2 * MX - 1, h: 0.55, fontSize: 11.5, fontFace: BF, valign: "middle", margin: 0 });

  footer(s, 10, false);
  s.addNotes("BRIDGE: Order matters — we check who has clinical data BEFORE the expensive waveform extraction. Stage 3, in coral, is the gate that catches ID-linkage bugs early. Now the second habit to steal: how one skill covers many different datasets.");
}

// =====================================================================
// SLIDE — BUILD: meta-lesson #2 (shared vs variable)
// =====================================================================
async function slideArchitecture() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Meta-lesson #2", "Same + different: a layered design", "build",
    "Separate the shared 90% from the per-dataset 10%.");

  const BC = SECTIONS.build.color;
  const layers = [
    { key: "code", c: BC, big: "SHARED", t: "SKILL.md — the workflow & rules", d: "Same for every dataset: principles, format, stages, gates." },
    { key: "cubes", c: BC, big: "SHARED", t: "physio_data package — the code", d: "Reusable utilities: resample, segment, align, verify, manifest." },
    { key: "db", c: AMBER, big: "PER-DATASET", t: "API.md + scripts — the variable 10%", d: "Paths, sample rates, ID linkage, timezones, which labs/vitals exist." },
  ];
  let yy = 1.95;
  for (const L of layers) {
    const h = 0.7;
    card(s, MX, yy, W - 2 * MX, h);
    s.addShape(PRES.shapes.RECTANGLE, { x: MX, y: yy, w: 0.08, h: h, fill: { color: L.c }, line: { type: "none" } });
    await iconChip(s, L.key, MX + 0.26, yy + 0.12, 0.46, SOFT, L.c);
    s.addText(L.big, { x: MX + 0.9, y: yy + 0.1, w: 1.65, h: 0.5, fontSize: 11, bold: true, color: L.c, fontFace: HF, charSpacing: 1, valign: "middle", margin: 0 });
    s.addText(L.t, { x: MX + 2.55, y: yy + 0.1, w: 6.35, h: 0.3, fontSize: 13, bold: true, color: NAVY, fontFace: HF, margin: 0 });
    s.addText(L.d, { x: MX + 2.55, y: yy + 0.4, w: 6.35, h: 0.26, fontSize: 10.5, color: MUTE, fontFace: BF, margin: 0 });
    yy += h + 0.1;
  }

  await bigTakeaway(s, "build", "New dataset = fill one API.md + a thin extractor. The other ~90% is reused.");
  footer(s, 11, false);
  s.addNotes("BRIDGE: This is the second habit. The art of a good skill is drawing the line between what's shared and what varies — put the shared 90% in the skill and package, isolate the variable 10% in a per-dataset spec. That's how one skill covers MIMIC and VitalDB. Last build slide: how to grow your own.");
}

// =====================================================================
// SLIDE — BUILD: grow your own
// =====================================================================
async function slideBuildEvolve() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Grow your own", "How to build & grow a skill", "build",
    "Every task you repeat is a candidate skill — write it once.");

  const BC = SECTIONS.build.color;
  const items = [
    { key: "flask", t: "Extract after 2–3 reps", d: "Don't pre-plan a skill. Do the task a couple of times, then lift out the pattern." },
    { key: "design", t: "Scaffold with skill-creator", d: "Use the built-in skill-creator for structure; then sharpen the description." },
    { key: "gears", t: "Keep skill + code in sync", d: "Learn something new? Update the skill in the same change. Drift kills trust." },
    { key: "layers", t: "It compounds", d: "Every dataset onboarded made the skill sharper — guardrails accrue over time." },
  ];
  const cw = 4.32, gap = 0.26, rowH = 1.18, y0 = 1.72;
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (cw + gap), y = y0 + row * (rowH + 0.16);
    card(s, x, y, cw, rowH);
    await iconChip(s, items[i].key, x + 0.24, y + 0.32, 0.58, SOFT, BC);
    s.addText(items[i].t, { x: x + 0.96, y: y + 0.22, w: cw - 1.16, h: 0.34, fontSize: 14, bold: true, color: NAVY, fontFace: HF, margin: 0, valign: "middle" });
    s.addText(items[i].d, { x: x + 0.96, y: y + 0.58, w: cw - 1.16, h: 0.5, fontSize: 11, color: MUTE, fontFace: BF, margin: 0 });
  }

  footer(s, 12, false);
  s.addNotes("BRIDGE: Skills are emergent, not upfront. Do it manually a few times, notice the repetition, extract it. Keep it in sync — a stale skill is worse than none. Intern message: every task you repeat this summer is a candidate skill. Now part three: actually using it.");
}

// =====================================================================
// SLIDE — USE: invoking
// =====================================================================
async function slideInvoke() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Invoke it", "Two ways to start", "use",
    "Two ways in; both lead to the same guided workflow.");

  const UC = SECTIONS.use.color;
  const y0 = 2.0, cw = 4.32, gap = 0.26, ch = 1.95;
  card(s, MX, y0, cw, ch);
  await iconChip(s, "rocket", MX + 0.28, y0 + 0.28, 0.6, SOFT, UC);
  s.addText("Explicit", { x: MX + 1.02, y: y0 + 0.3, w: cw - 1.2, h: 0.3, fontSize: 14, bold: true, color: NAVY, fontFace: HF, valign: "middle", margin: 0 });
  s.addShape(PRES.shapes.ROUNDED_RECTANGLE, { x: MX + 0.3, y: y0 + 1.0, w: cw - 0.6, h: 0.55, rectRadius: 0.06, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("/physio-data  mimic3", { x: MX + 0.46, y: y0 + 1.0, w: cw - 0.8, h: 0.55, fontSize: 14, color: MINT, fontFace: MF, valign: "middle", margin: 0 });

  const rx = MX + cw + gap;
  card(s, rx, y0, cw, ch);
  await iconChip(s, "search", rx + 0.28, y0 + 0.28, 0.6, SOFT, UC);
  s.addText("Automatic", { x: rx + 1.02, y: y0 + 0.3, w: cw - 1.2, h: 0.3, fontSize: 14, bold: true, color: NAVY, fontFace: HF, valign: "middle", margin: 0 });
  s.addText("Just say “help me onboard this physiological dataset” — the description triggers it automatically.", {
    x: rx + 0.3, y: y0 + 1.0, w: cw - 0.6, h: 0.8, fontSize: 12, color: INK, fontFace: BF, margin: 0 });

  footer(s, 13, false);
  s.addNotes("BRIDGE: Two ways in — type the slash command, or just describe the task and the description field makes Claude pick the skill. Either way it runs the same checkpointed process — which is exactly what the next slide walks through.");
}

// =====================================================================
// SLIDE — USE: run-through
// =====================================================================
async function slideRunthrough() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "Step by step", "A real run-through, with you in the loop", "use",
    "Claude runs the steps; you stay at the two human checkpoints.");

  const UC = SECTIONS.use.color;
  const steps = [
    { t: "Research & explore", d: "Claude probes raw files, profiles channels & rates", gate: false },
    { t: "Single-entity demo", d: "you eyeball the signal + event alignment plot", gate: true },
    { t: "Review API.md", d: "you approve paths, channels, variables, timezones", gate: true },
    { t: "Test: --limit 5", d: "verify output format on 5 patients before scaling", gate: false },
    { t: "Full extraction", d: "parallel workers in tmux, ≤ 50% of cluster cores", gate: false },
    { t: "Verify + splits", d: "checks pass, subject-level train/val/test written", gate: false },
  ];
  const x0 = MX + 0.15, y0 = 1.72, rowH = 0.5;
  s.addShape(PRES.shapes.LINE, { x: x0, y: y0 + 0.1, w: 0, h: rowH * (steps.length - 1) + 0.1, line: { color: LINEC, width: 2 } });
  for (let i = 0; i < steps.length; i++) {
    const y = y0 + i * rowH;
    const col = steps[i].gate ? CORAL : UC;
    s.addShape(PRES.shapes.OVAL, { x: x0 - 0.13, y: y - 0.02, w: 0.26, h: 0.26, fill: { color: col }, line: { color: "FFFFFF", width: 2 } });
    s.addText([
      { text: steps[i].t, options: { bold: true, color: NAVY, fontSize: 13 } },
      { text: "   —  " + steps[i].d, options: { color: MUTE, fontSize: 11 } },
    ], { x: x0 + 0.32, y: y - 0.06, w: 6.0, h: 0.34, fontFace: BF, valign: "middle", margin: 0 });
    if (steps[i].gate) {
      s.addShape(PRES.shapes.ROUNDED_RECTANGLE, { x: x0 + 6.35, y: y - 0.02, w: 1.55, h: 0.28, rectRadius: 0.04, fill: { color: "FBE7E4" }, line: { type: "none" } });
      s.addText("✋ human check", { x: x0 + 6.35, y: y - 0.03, w: 1.55, h: 0.28, fontSize: 9.5, bold: true, color: CORAL, fontFace: BF, align: "center", valign: "middle", margin: 0 });
    }
  }

  footer(s, 14, false);
  s.addNotes("BRIDGE: The skill is collaborative, not autonomous. The two coral gates — eyeballing the demo plot, approving API.md — are where a human must look. Everything else Claude runs. So does it actually work across datasets? Here's the proof.");
}

// =====================================================================
// SLIDE — USE: proof
// =====================================================================
async function slideProof() {
  const s = PRES.addSlide();
  s.background = { color: PAPER };
  head(s, "The payoff", "Proof: one skill, seven datasets", "use",
    "Seven very different datasets, one unchanged format.");

  const rows = [
    [{ text: "Dataset", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "Source", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "Entities", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } },
     { text: "Note", options: { bold: true, color: "FFFFFF", fill: { color: NAVY } } }],
    ["MIMIC-III", "PhysioNet ICU", "✓ complete", "A–G pipeline"],
    ["MC-MED", "ED monitoring", "81,745", "largest cohort"],
    ["MOVER / SIS", "UCI surgical", "6,993", "37.2% yield"],
    ["MOVER / EPIC", "UCI peri-op", "1,819", "peer dataset"],
    ["VitalDB", "Open surgical", "6,152", "96.3% yield"],
    ["Emory", "ICU (internal)", "✓ complete", "A–G stages"],
    ["UCSF", "ICU (internal)", "✓ complete", "+ CA task"],
  ];
  const tableData = rows.map((r, ri) =>
    r.map((c) => {
      const base = typeof c === "string" ? { text: c } : c;
      const opts = Object.assign({ fontFace: BF, fontSize: 10.8, color: INK, valign: "middle" }, base.options || {});
      if (ri > 0 && ri % 2 === 0 && !opts.fill) opts.fill = { color: SOFT };
      if (ri > 0 && !opts.fill) opts.fill = { color: CARD };
      return { text: base.text, options: opts };
    })
  );
  s.addTable(tableData, {
    x: MX, y: 1.7, w: 6.0, colW: [1.55, 1.6, 1.25, 1.6], rowH: 0.33,
    border: { type: "solid", pt: 0.5, color: LINEC }, align: "left", margin: [2, 4, 2, 4],
  });

  const rx = MX + 6.35, rw = W - MX - rx;
  const stats = [["7", "datasets onboarded", SECTIONS.design.color], ["1", "canonical format", SECTIONS.build.color], ["100k+", "patient-encounters", SECTIONS.use.color]];
  let yy = 1.7;
  for (const st of stats) {
    card(s, rx, yy, rw, 0.95);
    s.addText(st[0], { x: rx + 0.22, y: yy + 0.08, w: 1.5, h: 0.78, fontSize: 33, bold: true, color: st[2], fontFace: HF, valign: "middle", margin: 0 });
    s.addText(st[1], { x: rx + 1.5, y: yy + 0.08, w: rw - 1.65, h: 0.78, fontSize: 12, color: NAVY, fontFace: BF, valign: "middle", margin: 0 });
    yy += 1.06;
  }

  footer(s, 15, false);
  s.addNotes("BRIDGE: The payoff. Seven datasets — PhysioNet, hospital, open surgical, two internal — same format, same skill. Yields differ because strictness differs, but the format never changed. That's the point of a skill. Let me close with what to take home.");
}

// =====================================================================
// SLIDE — TAKEAWAYS (dark closing)
// =====================================================================
async function slideTakeaways() {
  const s = PRES.addSlide();
  s.background = { color: NAVY };
  s.addImage({ path: "assets/ecg_white.png", x: -0.5, y: 4.45, w: 11, h: 0.95, transparency: 74 });

  s.addText("TAKEAWAYS", { x: MX, y: 0.5, w: 6, h: 0.4, fontSize: 14, bold: true, color: MINT, fontFace: HF, charSpacing: 3, margin: 0 });
  s.addText("Find your recurring problem.\nWrite the skill once. Reap it forever.", {
    x: MX, y: 0.9, w: 9, h: 1.0, fontSize: 27, bold: true, color: "FFFFFF", fontFace: HF, margin: 0, lineSpacingMultiple: 1.0 });

  const items = [
    { key: "search", c: SECTIONS.design.light, t: "Research & design before code", d: "Use Claude Code to explore, profile, and argue trade-offs first." },
    { key: "puzzle", c: SECTIONS.design.light, t: "Spot the same-shape task", d: "Fixed shape + varying details = a skill waiting to be written." },
    { key: "branch", c: SECTIONS.build.light, t: "Split shared from variable", d: "Skill + package = the 90%; a per-case spec = the 10%." },
    { key: "hand", c: SECTIONS.use.light, t: "Keep humans at the gates", d: "Automate the work; verify at the moments that matter." },
  ];
  const cw = 4.32, gap = 0.26, rowH = 0.96, y0 = 2.15;
  for (let i = 0; i < 4; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * (cw + gap), y = y0 + row * (rowH + 0.14);
    s.addShape(PRES.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: rowH, rectRadius: 0.07, fill: { color: NAVY2 }, line: { color: NAVY3, width: 1 } });
    await iconChip(s, items[i].key, x + 0.22, y + 0.22, 0.52, NAVY3, items[i].c);
    s.addText(items[i].t, { x: x + 0.88, y: y + 0.14, w: cw - 1.05, h: 0.32, fontSize: 13, bold: true, color: "FFFFFF", fontFace: HF, valign: "middle", margin: 0 });
    s.addText(items[i].d, { x: x + 0.88, y: y + 0.46, w: cw - 1.05, h: 0.42, fontSize: 10.5, color: "BCD2E2", fontFace: BF, margin: 0 });
  }

  footer(s, 16, true);
  s.addNotes("CLOSE: Two habits — Claude Code is for thinking + building, not just code generation; and the moment a task recurs, capture it. physio-data is one instance; the pattern generalizes to almost anything you do repeatedly. Thank you — happy to take questions.");
}

// ---------- arrow ----------
let ARROW;

(async () => {
  await buildIcons();
  ARROW = await iconPng(fa.FaArrowRight, "#" + SECTIONS.build.color, 128);

  await slideTitle();
  await slideOverview();
  await slideProblem();
  await divider("design",
    "We have a recurring problem. First, design the solution — before any code.",
    ["The key insight", "Five design principles", "The canonical format", "Meta-lesson #1: design with Claude"],
    "ppg_white");
  await slideSignalsEvents();
  await slidePrinciples();
  await slideFormat();
  await slideDesignWithCC();
  await divider("build",
    "We have the design. Now encode it as a skill Claude can reuse.",
    ["Anatomy of a skill", "What goes inside", "The encoded workflow", "Meta-lesson #2: shared vs variable", "Grow your own"],
    "ecg_white");
  await slideWhatIsSkill();
  await slideEncoded();
  await slideWorkflow();
  await slideArchitecture();
  await slideBuildEvolve();
  await divider("use",
    "We have the skill. Now point it at a brand-new dataset.",
    ["Two ways to invoke it", "A step-by-step run", "The payoff: 7 datasets"],
    "ppg_white");
  await slideInvoke();
  await slideRunthrough();
  await slideProof();
  await slideTakeaways();

  await PRES.writeFile({ fileName: "physio-data-skill-talk.pptx" });
  console.log("WROTE physio-data-skill-talk.pptx");
})();
