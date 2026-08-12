// Generate waveform motif PNGs (PPG smooth + ECG spiky) used across the deck.
const sharp = require("sharp");

const W = 2600, H = 320;

// Build an SVG polyline from a sampled function f(t)->[0,1] (1 = top).
function polyline(fn, samples = 1300, pad = 10) {
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const x = pad + t * (W - 2 * pad);
    const yv = fn(t);                      // 0..1, 1 = top of band
    const y = (H - 2 * pad) * (1 - yv) + pad;
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return pts.join(" ");
}

// PPG: quick systolic upstroke, dicrotic notch, slow decay. ~6 beats.
function ppg(t) {
  const beats = 6;
  const p = (t * beats) % 1;            // phase within a beat
  let v;
  if (p < 0.18) {
    v = 0.5 + 0.45 * Math.sin((p / 0.18) * (Math.PI / 2)); // upstroke
  } else if (p < 0.30) {
    v = 0.95 - 0.30 * ((p - 0.18) / 0.12);                 // initial fall
  } else if (p < 0.42) {
    v = 0.65 + 0.08 * Math.sin(((p - 0.30) / 0.12) * Math.PI); // dicrotic bump
  } else {
    v = 0.60 - 0.10 * ((p - 0.42) / 0.58);                 // slow decay
    v = Math.max(0.5, v);
  }
  return v;
}

// ECG: flat baseline with P-QRS-T complex. ~6 beats.
function ecg(t) {
  const beats = 6;
  const p = (t * beats) % 1;
  let v = 0.5;
  // P wave
  v += 0.06 * Math.exp(-Math.pow((p - 0.15) / 0.03, 2));
  // Q dip
  v -= 0.07 * Math.exp(-Math.pow((p - 0.33) / 0.012, 2));
  // R spike
  v += 0.45 * Math.exp(-Math.pow((p - 0.37) / 0.012, 2));
  // S dip
  v -= 0.14 * Math.exp(-Math.pow((p - 0.41) / 0.014, 2));
  // T wave
  v += 0.11 * Math.exp(-Math.pow((p - 0.60) / 0.045, 2));
  return Math.min(0.99, Math.max(0.01, v));
}

function svg(fn, color, strokeW = 7, opacity = 1) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    <polyline points="${polyline(fn)}" fill="none" stroke="${color}" stroke-width="${strokeW}"
      stroke-linejoin="round" stroke-linecap="round" opacity="${opacity}"/>
  </svg>`;
}

async function render(name, fn, color, strokeW, opacity) {
  const buf = await sharp(Buffer.from(svg(fn, color, strokeW, opacity))).png().toBuffer();
  require("fs").writeFileSync(`assets/${name}.png`, buf);
  console.log("wrote", name);
}

(async () => {
  // teal / mint / white / faint variants for different backgrounds
  await render("ppg_mint",   ppg, "#2EC4B6", 8, 1);
  await render("ecg_mint",   ecg, "#2EC4B6", 7, 1);
  await render("ppg_teal",   ppg, "#0E8388", 9, 1);
  await render("ecg_teal",   ecg, "#0E8388", 8, 1);
  await render("ppg_faint",  ppg, "#9FB3C8", 6, 0.55);
  await render("ecg_faint",  ecg, "#9FB3C8", 6, 0.55);
  await render("ppg_white",  ppg, "#BFE9E6", 7, 0.85);
  await render("ecg_white",  ecg, "#7FD8D0", 6, 0.7);
})();
