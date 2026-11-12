// Roblox Up-and-Coming top-10 forecaster for make.fun's
// roblox-up-and-coming-forecast-01 (Node 22, standard library only).
//
// Model: a Monte Carlo simulation of the chart in latent-score space.
// Each game carries a score s = -(rank)^(1/3); the chart is the games sorted
// by score, highest first. The cube root is not arbitrary: measured rank
// mobility grows with rank roughly as r^(2/3), which is exactly the mobility a
// fixed score noise produces under this transform (a log transform makes the
// top too sticky, a linear one too loose).
//
// Every simulated hour each game's score moves by drift + noise:
//   - noise: gaussian, sigma per hour, estimated from the history itself as
//     the robust spread of the drift model's own residuals on 24-hour
//     prediction triples (so the unpredictable part, not the drift itself,
//     sets the spread), fallback 0.025.
//   - drift has three parts, fitted jointly by least squares on the history
//     (next-24h rank velocity on past velocity, playing growth and age;
//     R^2 0.14, playing growth the strongest single predictor):
//     (a) life-cycle (weight 0.77): a game's user growth rises after launch
//         and decays with age. Measured mean rank velocity by age: 0-2 days
//         -0.64 ranks/hour (climbing), 2-4 days -0.08, 4-6 days +0.06, 6+ days
//         +0.07 (falling). Age comes from details.created when the history
//         has it, else from first-seen-in-history plus a small lag correction,
//         refined by the Top Trending gap (a game's trending rank minus its
//         up-and-coming rank correlates -0.34 with its age: young games rank
//         far better on Up-and-Coming than on Trending).
//     (b) momentum (slope 0.125): the idiosyncratic part of the game's recent
//         rank velocity (observed velocity minus the life-cycle expectation)
//         persists into the next day and fades over ~36 hours.
//     (c) playing growth (slope -0.376 ranks/hour per unit of 24h log
//         playing growth): a surge in concurrent players leads a rank climb
//         by about a day (the chart is computed on a delay, not live). The
//         effect is applied at full strength for the first 24 hours, then
//         decays with a 24-hour time constant. A common +0.24 ranks/hour
//         downward drift also exists (new games replacing incumbents) but is
//         NOT added explicitly: it emerges from the entrants and climbers in
//         the simulation itself.
//   - new games enter the chart at ~13 per day (measured), debuting near the
//     bottom (median rank 87); about 16% of them are breakouts that climb to
//     the top 10 (measured: 18 of 115 mid-history debutants, median 45 hours
//     to the top 10, median peak rank 6). A breakout climbs with a score
//     drift sampled from the measured debut-to-top-10 drifts (0.021-0.078
//     score/hour) until it reaches a sampled peak rank (measured peaks 1-10),
//     then follows the normal drift. Entrants are anonymous in the
//     simulation, so their probability mass flows into each place's unnamed
//     share, as it should (measured: 23% of the top 10 at +72h were not on
//     the chart at all; 8-16% time-averaged over the window).
//   - age-out: the chart only lists games published in the last 28 days. A
//     game whose created date is known leaves the simulation when it turns 28
//     days old; without a known date a small rank-dependent disappearance
//     hazard applies instead (measured: disappearances concentrate at low
//     ranks, ~2 top-10 disappearances in 33 reads).
//
// The forecast is the honest time-average over the window: for each place and
// game, the fraction of (simulation, hour) pairs where the game held that
// place. Probabilities are reported as simulated (never 0 or 1 for a named
// game), each place keeps at least a small unnamed share, and the top 10 is
// ordered by expected place among the ten most likely top-10 games.
//
// Deterministic: a seeded PRNG (mulberry32 + Box-Muller), fixed iteration
// orders, ties broken by universe id. Two runs on the same input produce the
// same forecast.json byte for byte.
//
// Usage (outside the grader): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out
//   node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- Tunable parameters (calibrated on the practice history) ----
export interface Params {
  /** Monte Carlo simulations. */
  sims: number;
  /** Hourly score diffusion (score units); 0 = estimate from history. */
  sigma: number;
  /** Weight on the life-cycle drift (joint OLS coefficient). */
  ageWeight: number;
  /** Momentum persistence slope (next velocity per unit past velocity). */
  momentumSlope: number;
  /** Momentum half-life, hours. */
  momentumHalflifeH: number;
  /** Playing-growth slope: ranks/hour per unit 24h log playing growth. */
  playingGrowthSlope: number;
  /** Hours of full-strength playing-growth drift before decay. */
  playingGrowthFullH: number;
  /** Decay time constant (hours) of the playing-growth drift after the
      full-strength window. */
  playingGrowthDecayH: number;
  /** New games entering the chart per hour. */
  entrantsPerHour: number;
  /** Fraction of entrants that are viral breakouts. */
  breakoutShare: number;
  /** Minimum unnamed share left at each place. */
  minUnnamed: number;
  /** Only name games with P(place) >= this. */
  minNamedP: number;
  /** Max rows named at each place. */
  maxRowsPerPlace: number;
}

export const DEFAULT_PARAMS: Params = {
  sims: 5000,
  sigma: 0, // estimate from history
  ageWeight: 0.77,
  momentumSlope: 0.125,
  momentumHalflifeH: 36,
  playingGrowthSlope: -0.376,
  playingGrowthFullH: 24,
  playingGrowthDecayH: 24,
  entrantsPerHour: 0.56,
  breakoutShare: 0.15,
  // 0.15: the measured time-averaged share of top-10 slots held by games not
  // on the chart at t0 was 8% in the practice window and 16% in a second
  // backtest window; the floor keeps the forecast honest about churn the
  // simulation itself undercounts. It was the best value on the practice
  // window and near-best on the second window (0.20 was only 0.09 better
  // there, and exceeds both measured shares).
  minUnnamed: 0.15,
  minNamedP: 0.003,
  maxRowsPerPlace: 80,
};

// Life-cycle: mean rank velocity (ranks/hour, negative = climbing) by age.
// Measured on the practice history (archive first-seen as the age proxy).
const AGE_VELOCITY: Array<[number, number]> = [
  [0, -0.64],
  [2, -0.64],
  [3, -0.08],
  [5, 0.06],
  [7, 0.07],
  [14, 0.1],
  [28, 0.15],
];

function ageVelocity(ageDays: number): number {
  const a = Math.max(0, Math.min(28, ageDays));
  for (let i = 0; i < AGE_VELOCITY.length - 1; i++) {
    const [x0, y0] = AGE_VELOCITY[i]!;
    const [x1, y1] = AGE_VELOCITY[i + 1]!;
    if (a >= x0 && a <= x1) return y0 + ((y1 - y0) * (a - x0)) / (x1 - x0 || 1);
  }
  return AGE_VELOCITY[AGE_VELOCITY.length - 1]![1];
}

// Breakout climbs, measured on the practice history: the 17 finite
// debut-to-top-10 score drifts (score/hour) and the 18 peak ranks of the
// debutants that reached the top 10. Breakouts sample one drift and one peak
// each; they climb at the drift until they reach the peak, then follow the
// normal drift. (One debutant entered the chart already in the top 10, so it
// has no finite drift; its peak of 4 is in the peak list.)
const BREAKOUT_DRIFTS = [
  0.021, 0.029, 0.035, 0.037, 0.048, 0.049, 0.05, 0.051, 0.053, 0.055, 0.058,
  0.059, 0.061, 0.065, 0.065, 0.066, 0.078,
];
const BREAKOUT_PEAKS = [
  1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
];

// Disappearance hazard per hour by rank (no known created date): measured
// disappearances concentrate at low ranks; top-10 losses are rare.
function disappearHazard(rank: number): number {
  if (rank <= 10) return 0.0004;
  if (rank <= 30) return 0.001;
  if (rank <= 60) return 0.003;
  return 0.008;
}

// ---- Seeded PRNG (deterministic) ----
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Rng {
  private next: () => number;
  private spare: number | null = null;
  constructor(seed: number) {
    this.next = mulberry32(seed);
  }
  uniform(): number {
    return this.next();
  }
  gaussian(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    do {
      u = this.next();
    } while (u <= 1e-12);
    v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    this.spare = r * Math.sin(t);
    return r * Math.cos(t);
  }
  poisson(lambda: number): number {
    // Knuth, fine for small lambda.
    const l = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= this.next();
    } while (p > l);
    return k - 1;
  }
}

// ---- Input types ----
interface ChartRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartRow[];
  top_trending?: ChartRow[] | null;
  details?: Record<string, { created?: string | null }> | null;
}

interface GameState {
  id: number;
  score: number;
  /** Rank at t0 (for hazard + diagnostics). */
  rank0: number;
  /** Idiosyncratic velocity, ranks/hour (observed minus life-cycle). */
  idioVel: number;
  /** 24h log playing growth at t0 (clamped to +/-3; 0 if unmeasurable). */
  logG24: number;
  /** Age in days at t0 (from created, else first-seen + lag, gap-refined). */
  ageDays: number;
  /** Hour (from t0) the game turns 28 days old, null if unknown. */
  ageOutHour: number | null;
  alive: boolean;
}

function scoreOfRank(rank: number): number {
  return -Math.cbrt(Math.max(1, rank));
}

function rankOfScore(score: number): number {
  return Math.max(1, Math.round(Math.pow(-score, 3)));
}

// ---- History parsing & feature building ----
export interface ParsedHistory {
  snapshots: Snapshot[];
  t0ms: number;
  windowHours: number;
  places: number;
  games: GameState[];
  /** Empirical debut ranks of games that entered the chart mid-history. */
  debutRanks: number[];
  sigma: number;
  entrantsPerHour: number;
}

export function parseHistory(inputDir: string, params: Params): ParsedHistory {
  const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
    places: number;
    output: string;
    history_until: string;
    window: { slots: number; slot_minutes: number };
  };
  const lines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8')
    .split('\n')
    .filter((l) => l !== '');
  const snapshots = lines.map((l) => JSON.parse(l) as Snapshot);
  snapshots.sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime());
  const t0ms = new Date(task.history_until).getTime();
  const windowHours = task.window.slots * (task.window.slot_minutes / 60);
  const places = task.places;

  const last = snapshots[snapshots.length - 1]!;
  const lastMs = new Date(last.at).getTime();

  // First-seen time and created date per game.
  const firstSeen = new Map<number, number>();
  const created = new Map<number, number>();
  for (const s of snapshots) {
    const t = new Date(s.at).getTime();
    for (const g of s.up_and_coming) {
      if (!firstSeen.has(g.universe_id)) firstSeen.set(g.universe_id, t);
    }
    if (s.details) {
      for (const [uid, d] of Object.entries(s.details)) {
        if (d && d.created && !created.has(Number(uid))) {
          created.set(Number(uid), new Date(d.created).getTime());
        }
      }
    }
  }

  // Rank/playing trajectories per game: [timeMs, rank, playing][].
  const traj = new Map<number, Array<[number, number, number]>>();
  for (const s of snapshots) {
    const t = new Date(s.at).getTime();
    for (const g of s.up_and_coming) {
      const arr = traj.get(g.universe_id) ?? [];
      arr.push([t, g.rank, g.playing ?? 0]);
      traj.set(g.universe_id, arr);
    }
  }

  // Velocity: least-squares slope of rank on time over the 48h ending at the
  // anchor (at least 2 points), in ranks/hour. Falls back to 0.
  function velocityAt(uid: number, anchorMs: number): number {
    const arr = traj.get(uid);
    if (!arr || arr.length < 2) return 0;
    const cutoff = anchorMs - 48 * 3600 * 1000;
    const pts = arr.filter(([t]) => t >= cutoff && t <= anchorMs);
    if (pts.length < 2) return 0;
    const n = pts.length;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const [t, r] of pts) {
      const x = (t - anchorMs) / 3600e3;
      sx += x;
      sy += r;
      sxx += x * x;
      sxy += x * r;
    }
    const den = n * sxx - sx * sx;
    if (Math.abs(den) < 1e-9) return 0;
    return (n * sxy - sx * sy) / den;
  }
  function velocity(uid: number): number {
    return velocityAt(uid, lastMs);
  }

  // 24h log playing growth: log(playing now / playing ~24h ago), clamped to
  // +/-3. Uses the snapshot nearest t0-24h (within 18-30h); else the earliest
  // snapshot at least 6h back (unscaled, conservative); else 0. Playing
  // counts of 0 or missing make the growth unmeasurable.
  function logGrowthAt(uid: number, anchorMs: number): number {
    const arr = traj.get(uid);
    if (!arr || arr.length < 2) return 0;
    // The observation at (or nearest before) the anchor.
    let nowIdx = -1;
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i]![0] <= anchorMs) {
        nowIdx = i;
        break;
      }
    }
    if (nowIdx < 1) return 0;
    const nowPlaying = arr[nowIdx]![2];
    if (nowPlaying <= 0) return 0;
    const target = anchorMs - 24 * 3600e3;
    let best: [number, number, number] | null = null;
    let bestDist = Infinity;
    for (let i = 0; i < nowIdx; i++) {
      const pt = arr[i]!;
      const d = Math.abs(pt[0] - target);
      if (pt[0] <= anchorMs - 18 * 3600e3 && pt[0] >= anchorMs - 30 * 3600e3 && d < bestDist) {
        best = pt;
        bestDist = d;
      }
    }
    if (best === null) {
      // Fall back to the earliest observation at least 6h back.
      for (let i = 0; i < nowIdx; i++) {
        if (arr[i]![0] <= anchorMs - 6 * 3600e3) {
          best = arr[i]!;
          break;
        }
      }
    }
    if (best === null || best[2] <= 0) return 0;
    const g = Math.log(nowPlaying / best[2]);
    return Math.max(-3, Math.min(3, g));
  }
  function logGrowth24(uid: number): number {
    return logGrowthAt(uid, lastMs);
  }

  // Trending ranks at the last snapshot.
  const trending = new Map<number, number>();
  if (last.top_trending) {
    for (const g of last.top_trending) trending.set(g.universe_id, g.rank);
  }

  // Candidate universe: games on the last chart, plus games that were in the
  // top 40 in the last 48h but have since left (they can climb back).
  const candidateIds = new Set<number>();
  for (const g of last.up_and_coming) candidateIds.add(g.universe_id);
  const cutoff48 = lastMs - 48 * 3600 * 1000;
  for (const s of snapshots) {
    if (new Date(s.at).getTime() < cutoff48) continue;
    for (const g of s.up_and_coming.slice(0, 40)) candidateIds.add(g.universe_id);
  }
  const lastRank = new Map<number, number>();
  for (const g of last.up_and_coming) lastRank.set(g.universe_id, g.rank);

  const games: GameState[] = [];
  for (const uid of candidateIds) {
    const rank0 = lastRank.get(uid) ?? 105; // off-chart: just below it
    const c = created.get(uid);
    const fs = firstSeen.get(uid) ?? lastMs;
    let ageDays: number;
    if (c !== undefined) {
      ageDays = (t0ms - c) / 864e5;
    } else {
      // No created date: age is at least the time since first seen (plus a
      // ~2 day lag between launch and chart entry). Refine with the Top
      // Trending gap, which correlates -0.34 with age: gap 44 -> ~1 day old,
      // gap 28 -> ~3.7 days, gap 15 -> ~6 days (linear fit on the history).
      const gap = (trending.get(uid) ?? 120) - rank0;
      const gapAge = Math.max(0.5, Math.min(14, 8.5 - 0.17 * gap));
      ageDays = Math.max((t0ms - fs) / 864e5 + 2, gapAge);
    }
    ageDays = Math.max(0, Math.min(27.9, ageDays));
    const ageOutHour =
      c !== undefined ? (c + 28 * 864e5 - t0ms) / 3600e3 : null;
    const vObs = velocity(uid);
    const vAge = ageVelocity(ageDays);
    games.push({
      id: uid,
      score: scoreOfRank(rank0),
      rank0,
      idioVel: vObs - vAge,
      logG24: logGrowth24(uid),
      ageDays,
      ageOutHour,
      alive: true,
    });
  }
  // Deterministic order: by rank at t0, then universe id.
  games.sort((a, b) => a.rank0 - b.rank0 || a.id - b.id);

  // Empirical debut ranks (games whose first appearance is mid-history).
  const debutRanks: number[] = [];
  for (const [uid, t] of firstSeen) {
    if (t === new Date(snapshots[0]!.at).getTime()) continue;
    const arr = traj.get(uid);
    if (arr && arr.length > 0) debutRanks.push(arr[0]![1]);
  }
  debutRanks.sort((a, b) => a - b);

  // Entrant rate per hour (new games per hour over the history).
  const spanH = Math.max(1, (lastMs - new Date(snapshots[0]!.at).getTime()) / 3600e3);
  const entrantsPerHour =
    debutRanks.length > 5 ? debutRanks.length / spanH : params.entrantsPerHour;

  // Sigma: robust scale of the drift model's own residuals. For each 24h
  // prediction triple (i, j, k) and each top-40 game at j, compare the actual
  // rank change j->k with the change the drift model predicts from the
  // features at j. The residual (in score units, per sqrt hour) is the
  // unpredictable part; its MAD x 1.4826 is sigma. This is honest: a raw
  // MAD of score changes would double-count the predictable drift.
  // Fallback 0.025.
  let sigma = 0.025;
  if (params.sigma > 0) {
    sigma = params.sigma;
  } else {
    const residuals: number[] = [];
    const times = snapshots.map((s) => new Date(s.at).getTime());
    for (let j = 1; j < snapshots.length - 1; j++) {
      // Find i ~24h before j and k ~24h after j.
      let iIdx = -1;
      let kIdx = -1;
      let bestI = Infinity;
      let bestK = Infinity;
      for (let i = 0; i < j; i++) {
        const d = Math.abs(times[j]! - times[i]! - 24 * 3600e3);
        if (d < bestI) {
          bestI = d;
          iIdx = i;
        }
      }
      for (let k = j + 1; k < snapshots.length; k++) {
        const d = Math.abs(times[k]! - times[j]! - 24 * 3600e3);
        if (d < bestK) {
          bestK = d;
          kIdx = k;
        }
      }
      if (iIdx < 0 || kIdx < 0) continue;
      const d1h = (times[j]! - times[iIdx]!) / 3600e3;
      const d2h = (times[kIdx]! - times[j]!) / 3600e3;
      if (Math.abs(d1h - 24) > 5 || Math.abs(d2h - 24) > 5) continue;
      const rankK = new Map<number, number>();
      for (const g of snapshots[kIdx]!.up_and_coming) rankK.set(g.universe_id, g.rank);
      for (const g of snapshots[j]!.up_and_coming.slice(0, 40)) {
        const rK = rankK.get(g.universe_id);
        if (rK === undefined) continue;
        const uid = g.universe_id;
        const c = created.get(uid);
        const fs = firstSeen.get(uid) ?? times[j]!;
        const ageJ =
          c !== undefined ? (times[j]! - c) / 864e5 : (times[j]! - fs) / 864e5 + 2;
        const vObs = velocityAt(uid, times[j]!);
        const idio = vObs - ageVelocity(ageJ);
        const logG = logGrowthAt(uid, times[j]!);
        const predVel =
          params.ageWeight * ageVelocity(ageJ) +
          params.momentumSlope * idio +
          params.playingGrowthSlope * logG;
        const actualVel = (rK - g.rank) / d2h;
        const residRanks = (actualVel - predVel) * d2h;
        // Convert a rank residual to a score residual per sqrt hour.
        const dScoreDrank = (1 / 3) * Math.pow(Math.max(1, g.rank), -2 / 3);
        residuals.push((residRanks * dScoreDrank) / Math.sqrt(d2h));
      }
    }
    if (residuals.length > 30) {
      residuals.sort((a, b) => a - b);
      const med = residuals[Math.floor(residuals.length / 2)]!;
      const absDev = residuals.map((d) => Math.abs(d - med)).sort((a, b) => a - b);
      const mad = absDev[Math.floor(absDev.length / 2)]!;
      const est = 1.4826 * mad;
      if (est > 0.005 && est < 0.1) sigma = est;
    }
  }

  return { snapshots, t0ms, windowHours, places, games, debutRanks, sigma, entrantsPerHour };
}

// ---- Monte Carlo simulation ----
export interface SimResult {
  /** place -> game id -> probability (time-averaged over the window). */
  prob: Map<number, Map<number, number>>;
  /** place -> probability the game there is unnamed (a future entrant). */
  unnamed: Map<number, number>;
  /** game id -> expected rank (mean over sims/hours while in the top 40). */
  expectedRank: Map<number, number>;
  /** game id -> total top-10 probability mass over all places. */
  top10Mass: Map<number, number>;
}

export function simulate(hist: ParsedHistory, params: Params): SimResult {
  const { games, windowHours, places, debutRanks, sigma } = hist;
  const hours = Math.round(windowHours);
  const entrantsPerHour =
    hist.entrantsPerHour > 0 ? hist.entrantsPerHour : params.entrantsPerHour;

  const prob = new Map<number, Map<number, number>>();
  const unnamed = new Map<number, number>();
  const rankSum = new Map<number, number>();
  const rankCount = new Map<number, number>();
  const top10Mass = new Map<number, number>();
  for (let k = 1; k <= places; k++) {
    prob.set(k, new Map());
    unnamed.set(k, 0);
  }

  const totalSamples = params.sims * hours;

  for (let sim = 0; sim < params.sims; sim++) {
    const rng = new Rng(0x9e3779b9 ^ (sim * 2654435761));
    // Working state: parallel arrays for speed.
    const ids: number[] = games.map((g) => g.id);
    const scores: number[] = games.map((g) => g.score);
    const idioVel: number[] = games.map((g) => g.idioVel);
    const logG24: number[] = games.map((g) => g.logG24);
    const ageDays: number[] = games.map((g) => g.ageDays);
    const ageOutHour: Array<number | null> = games.map((g) => g.ageOutHour);
    // Breakout climb: score drift per hour (0 = not a breakout) and the peak
    // rank the climb stops at. Incumbents never break out (their surge, if
    // any, is already in their playing-growth drift).
    const breakoutDrift: number[] = games.map(() => 0);
    const breakoutPeak: number[] = games.map(() => 0);
    const alive: boolean[] = games.map(() => true);
    let n = ids.length;

    for (let h = 1; h <= hours; h++) {
      // New entrants this hour.
      const nNew = rng.poisson(entrantsPerHour);
      for (let e = 0; e < nNew; e++) {
        const debut =
          debutRanks.length > 0
            ? debutRanks[Math.floor(rng.uniform() * debutRanks.length)]!
            : 87;
        ids.push(-(sim * 100000 + h * 100 + e + 1)); // anonymous: never named
        scores.push(scoreOfRank(debut));
        // Breakouts climb at a sampled empirical drift to a sampled peak;
        // others start neutral and follow the young-game life-cycle (age 0).
        idioVel.push(0);
        logG24.push(0);
        ageDays.push(0);
        ageOutHour.push(h + 28 * 24);
        if (rng.uniform() < params.breakoutShare) {
          breakoutDrift.push(
            BREAKOUT_DRIFTS[Math.floor(rng.uniform() * BREAKOUT_DRIFTS.length)]!,
          );
          breakoutPeak.push(
            BREAKOUT_PEAKS[Math.floor(rng.uniform() * BREAKOUT_PEAKS.length)]!,
          );
        } else {
          breakoutDrift.push(0);
          breakoutPeak.push(0);
        }
        alive.push(true);
        n++;
      }

      // Update scores.
      const momDecay = Math.pow(0.5, h / params.momentumHalflifeH);
      // Playing-growth drift: full strength for the first day, then decays
      // (the surge's effect is spent; future growth is unobserved).
      const pgDecay =
        h <= params.playingGrowthFullH
          ? 1
          : Math.exp(-(h - params.playingGrowthFullH) / params.playingGrowthDecayH);
      for (let i = 0; i < n; i++) {
        if (!alive[i]) continue;
        const ao = ageOutHour[i];
        if (ao !== null && h >= ao) {
          alive[i] = false;
          continue;
        }
        // Disappearance hazard for games without a known age-out.
        if (ao === null) {
          const r = rankOfScore(scores[i]!);
          if (rng.uniform() < disappearHazard(r)) {
            alive[i] = false;
            continue;
          }
        }
        const r = rankOfScore(scores[i]!);
        if (breakoutDrift[i]! > 0 && r > breakoutPeak[i]!) {
          // Breakout still climbing: sampled score drift, applied directly.
          scores[i] = scores[i]! + breakoutDrift[i]! + sigma * rng.gaussian();
          continue;
        }
        if (breakoutDrift[i]! > 0) breakoutDrift[i] = 0; // reached its peak
        const dScoreDrank = -(1 / 3) * Math.pow(Math.max(1, r), -2 / 3);
        // Life-cycle drift at the game's current age (age advances hourly).
        const ageNow = ageDays[i]! + h / 24;
        let driftRanksPerHour = params.ageWeight * ageVelocity(ageNow);
        // Momentum on the idiosyncratic velocity, fading with time.
        driftRanksPerHour += params.momentumSlope * idioVel[i]! * momDecay;
        // Playing-growth drift: a player surge leads a rank climb by ~a day.
        driftRanksPerHour += params.playingGrowthSlope * logG24[i]! * pgDecay;
        scores[i] =
          scores[i]! + dScoreDrank * driftRanksPerHour + sigma * rng.gaussian();
      }

      // Rank the alive games by score (ties: lower universe id first, so the
      // order is deterministic).
      const order: number[] = [];
      for (let i = 0; i < n; i++) if (alive[i]) order.push(i);
      order.sort((a, b) => scores[b]! - scores[a]! || ids[a]! - ids[b]!);

      for (let k = 1; k <= places && k <= order.length; k++) {
        const idx = order[k - 1]!;
        const id = ids[idx]!;
        if (id < 0) {
          unnamed.set(k, (unnamed.get(k) ?? 0) + 1);
        } else {
          const m = prob.get(k)!;
          m.set(id, (m.get(id) ?? 0) + 1);
          top10Mass.set(id, (top10Mass.get(id) ?? 0) + 1);
        }
      }
      // Expected-rank tracking for the top-40 (for the top10 tie-break).
      const topN = Math.min(40, order.length);
      for (let k = 1; k <= topN; k++) {
        const id = ids[order[k - 1]!]!;
        if (id < 0) continue;
        rankSum.set(id, (rankSum.get(id) ?? 0) + k);
        rankCount.set(id, (rankCount.get(id) ?? 0) + 1);
      }
    }
  }

  // Normalise to probabilities.
  for (const [, m] of prob) {
    for (const [id, c] of m) m.set(id, c / totalSamples);
  }
  for (const [k, c] of unnamed) unnamed.set(k, c / totalSamples);
  for (const [id, c] of top10Mass) top10Mass.set(id, c / totalSamples);
  const expectedRank = new Map<number, number>();
  for (const [id, s] of rankSum) {
    expectedRank.set(id, s / (rankCount.get(id) ?? 1));
  }
  return { prob, unnamed, expectedRank, top10Mass };
}

// ---- Forecast assembly ----
export interface Forecast {
  probabilities: Array<{ universe_id: number; place: number; p: number }>;
  top10: number[];
}

export function assemble(sim: SimResult, places: number, params: Params): Forecast {
  const probabilities: Forecast['probabilities'] = [];
  for (let k = 1; k <= places; k++) {
    const m = sim.prob.get(k) ?? new Map<number, number>();
    const rows = [...m.entries()]
      .filter(([, p]) => p >= params.minNamedP)
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, params.maxRowsPerPlace);
    let sum = 0;
    for (const [, p] of rows) sum += p;
    // Keep at least minUnnamed unnamed: if the named rows already leave less,
    // scale them down to fit (they are honest probabilities; the residual is
    // the mass on games we do not name plus future entrants).
    const cap = 1 - params.minUnnamed;
    const scale = sum > cap ? cap / sum : 1;
    for (const [id, p] of rows) {
      probabilities.push({ universe_id: id, place: k, p: round6(p * scale) });
    }
  }

  // top10: the ten games with the most top-10 mass, ordered by expected place.
  const expectedPlace = new Map<number, number>();
  for (const [id] of sim.top10Mass) {
    let num = 0;
    let den = 0;
    for (let k = 1; k <= places; k++) {
      const p = sim.prob.get(k)?.get(id) ?? 0;
      num += k * p;
      den += p;
    }
    expectedPlace.set(id, den > 0 ? num / den : 99);
  }
  const top10 = [...sim.top10Mass.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, places)
    .sort((a, b) => (expectedPlace.get(a[0]) ?? 99) - (expectedPlace.get(b[0]) ?? 99) || a[0] - b[0])
    .map(([id]) => id);
  return { probabilities, top10 };
}

function round6(x: number): number {
  return Math.round(x * 1e6) / 1e6;
}

export function runForecast(inputDir: string, params: Params): Forecast {
  const hist = parseHistory(inputDir, params);
  const sim = simulate(hist, params);
  return assemble(sim, hist.places, params);
}

function main(): void {
  const forecast = runForecast(INPUT_DIR, DEFAULT_PARAMS);
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
    output: string;
  };
  writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify(forecast)}\n`);
}

const isEntry = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isEntry) main();
