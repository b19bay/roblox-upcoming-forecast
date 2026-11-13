// Roblox Up-and-Coming top-10 forecaster, v3 (make.fun
// roblox-up-and-coming-forecast-01; Node 22, standard library only).
// Monte Carlo in latent-score space: s = -(rank)^(1/3), 5,000 seeded
// simulations, time-averaged over the 72-hour window. v3 changes:
// momentum slope 0.05 (raw velocity is mostly noise), a 5% persistence
// blend of each place's distribution toward the game holding that place
// in the last snapshot, and top-10 ordering by median place. Local
// scores: practice -32.116, backtest2 -32.775, held-out -31.151.
// Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---- Tunable parameters (fitted on the practice history, see header) ----
export interface Params {
  /** Monte Carlo simulations. */
  sims: number;
  /** Hourly score diffusion (score units); 0 = estimate from history. */
  sigma: number;
  /** Latent exponent: score = -(rank)^exponent. */
  exponent: number;
  /** Weight on the life-cycle drift (joint OLS coefficient). */
  ageWeight: number;
  /** Momentum persistence slope (next velocity per unit past velocity). */
  momentumSlope: number;
  /** Persistence blend: weight on the game holding each place in the last
      snapshot, mixed into that place's Monte Carlo distribution. */
  lastSnapBlend: number;
  /** Momentum half-life, hours. */
  momentumHalflifeH: number;
  /** Playing-growth slope: ranks/hour per unit 24h log playing growth. */
  playingGrowthSlope: number;
  /** Hours of full-strength playing-growth drift before decay. */
  playingGrowthFullH: number;
  /** Decay time constant (hours) of the playing-growth drift after the
      full-strength window. */
  playingGrowthDecayH: number;
  /** New games entering the chart per hour (fallback; estimated when the
      history has enough debuts). */
  entrantsPerHour: number;
  /** Fraction of entrants that are viral breakouts. */
  breakoutShare: number;
  /** Minimum unnamed share left at each place. */
  minUnnamed: number;
  /** Only name games with P(place) >= this. */
  minNamedP: number;
  /** Max rows named at each place. */
  maxRowsPerPlace: number;
  /** Age-from-trending-gap linear fit: age ~= gapIntercept + gapSlope*gap. */
  gapIntercept: number;
  gapSlope: number;
  /** Life-cycle profile: [age in days, mean rank velocity in ranks/hour]. */
  ageProfile: Array<[number, number]>;
}

// Life-cycle: mean rank velocity (ranks/hour, negative = climbing) by age in
// days. Direct measurement on the practice history gives a steeper old-age
// fall (6-10d: +0.20), but the 72h backtests favor this flatter profile:
// the measured level includes the common downward drift that the simulation
// already produces through entrants, so the steep profile double-counts it
// over a 72h horizon, and the top-10 incumbents the drift applies to are a
// selected, stickier set than the population the profile was measured on.
// The young end is measured directly: debutants move -0.62 ranks/hour in
// their first 48 hours.
const AGE_PROFILE: Array<[number, number]> = [
  [0, -0.64],
  [2, -0.64],
  [3, -0.08],
  [5, 0.06],
  [7, 0.07],
  [14, 0.1],
  [28, 0.15],
];

function ageVelocity(profile: Array<[number, number]>, ageDays: number): number {
  const a = Math.max(0, Math.min(28, ageDays));
  for (let i = 0; i < profile.length - 1; i++) {
    const [x0, y0] = profile[i]!;
    const [x1, y1] = profile[i + 1]!;
    if (a >= x0 && a <= x1) return y0 + ((y1 - y0) * (a - x0)) / (x1 - x0 || 1);
  }
  return profile[profile.length - 1]![1];
}

export const DEFAULT_PARAMS: Params = {
  sims: 5000,
  sigma: 0, // estimate from history
  exponent: 1 / 3,
  ageWeight: 0.73,
  momentumSlope: 0.05,
  lastSnapBlend: 0.05,
  momentumHalflifeH: 36,
  playingGrowthSlope: -0.4,
  playingGrowthFullH: 24,
  playingGrowthDecayH: 24,
  entrantsPerHour: 0.56,
  breakoutShare: 0.16,
  minUnnamed: 0.15,
  minNamedP: 0.003,
  maxRowsPerPlace: 80,
  gapIntercept: 8.67,
  gapSlope: -0.048,
  ageProfile: AGE_PROFILE,
};

// Breakout climbs, measured on the practice history: the 17 debutants that
// reached the top 10, as [debut rank, hours from debut to first top-10 read,
// peak rank]. The score drift is derived from these at run time in whatever
// latent transform is in use, so the table stays valid if the exponent
// changes. (One more debutant entered the chart already inside the top 10,
// peak 4; it has no climb, so it appears only in the peak list.)
const BREAKOUT_CLIMBS: Array<[number, number, number]> = [
  [20, 23.3, 4],
  [29, 23.3, 7],
  [42, 39.5, 9],
  [55, 48.3, 1],
  [56, 23.3, 8],
  [62, 31.7, 7],
  [66, 39.5, 3],
  [69, 48.0, 1],
  [72, 54.8, 10],
  [73, 40.0, 10],
  [88, 44.6, 5],
  [90, 66.0, 2],
  [91, 66.3, 1],
  [92, 49.3, 6],
  [93, 87.5, 8],
  [96, 129.7, 6],
  [97, 42.7, 9],
];
const BREAKOUT_PEAKS = [1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10];

// Disappearance hazard per hour by rank, measured on the practice history
// (games present at one read, gone at the next, per game-hour of exposure).
function disappearHazard(rank: number): number {
  if (rank <= 10) return 0.001;
  if (rank <= 30) return 0.002;
  if (rank <= 60) return 0.002;
  return 0.018;
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
  next: () => number;
  spare: number | null = null;
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
    do {
      u = this.next();
    } while (u <= 1e-12);
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    this.spare = r * Math.sin(t);
    return r * Math.cos(t);
  }
  poisson(lambda: number): number {
    // Knuth's algorithm; fine for the small lambdas used here.
    const limit = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= this.next();
    } while (p > limit);
    return k - 1;
  }
}

// ---- Latent score transform (parameterised by the exponent) ----
function makeTransform(exponent: number) {
  return {
    scoreOfRank(rank: number): number {
      return -Math.pow(Math.max(1, rank), exponent);
    },
    rankOfScore(score: number): number {
      return Math.max(1, Math.round(Math.pow(-score, 1 / exponent)));
    },
    // d(score)/d(rank), negative: a rank increase lowers the score.
    dScoreDrank(rank: number): number {
      return -exponent * Math.pow(Math.max(1, rank), exponent - 1);
    },
  };
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
  /** Rank at t0 (105 for games off the chart). */
  rank0: number;
  /** Idiosyncratic velocity, ranks/hour (observed minus life-cycle). */
  idioVel: number;
  /** 24h log playing growth at t0 (clamped to +/-3; 0 if unmeasurable). */
  logG24: number;
  /** Age in days at t0. */
  ageDays: number;
  /** Hour (from t0) the game turns 28 days old, null if unknown. */
  ageOutHour: number | null;
  alive: boolean;
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

const HOUR_MS = 3600e3;
const DAY_MS = 864e5;

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
  const transform = makeTransform(params.exponent);

  const last = snapshots[snapshots.length - 1]!;
  const lastMs = new Date(last.at).getTime();
  const firstMs = new Date(snapshots[0]!.at).getTime();

  // First-seen time and publication date per game.
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

  // Rank velocity: least-squares slope of rank on time over the 48h ending
  // at the anchor (at least 2 points), in ranks/hour. Falls back to 0.
  function velocityAt(uid: number, anchorMs: number): number {
    const arr = traj.get(uid);
    if (!arr || arr.length < 2) return 0;
    const pts = arr.filter(([t]) => t >= anchorMs - 48 * HOUR_MS && t <= anchorMs);
    if (pts.length < 2) return 0;
    const n = pts.length;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const [t, r] of pts) {
      const x = (t - anchorMs) / HOUR_MS;
      sx += x;
      sy += r;
      sxx += x * x;
      sxy += x * r;
    }
    const den = n * sxx - sx * sx;
    if (Math.abs(den) < 1e-9) return 0;
    return (n * sxy - sx * sy) / den;
  }

  // 24h log playing growth at the anchor: log(playing now / playing ~24h
  // ago), clamped to +/-3. Prefers the snapshot nearest 24h back (within
  // 18-30h); else the earliest snapshot at least 6h back (unscaled,
  // conservative); else 0. Zero or missing playing counts are unmeasurable.
  function logGrowthAt(uid: number, anchorMs: number): number {
    const arr = traj.get(uid);
    if (!arr || arr.length < 2) return 0;
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
    const target = anchorMs - 24 * HOUR_MS;
    let best: [number, number, number] | null = null;
    let bestDist = Infinity;
    for (let i = 0; i < nowIdx; i++) {
      const pt = arr[i]!;
      if (pt[0] <= anchorMs - 18 * HOUR_MS && pt[0] >= anchorMs - 30 * HOUR_MS) {
        const d = Math.abs(pt[0] - target);
        if (d < bestDist) {
          best = pt;
          bestDist = d;
        }
      }
    }
    if (best === null) {
      for (let i = 0; i < nowIdx; i++) {
        if (arr[i]![0] <= anchorMs - 6 * HOUR_MS) {
          best = arr[i]!;
          break;
        }
      }
    }
    if (best === null || best[2] <= 0) return 0;
    return Math.max(-3, Math.min(3, Math.log(nowPlaying / best[2])));
  }

  // Trending ranks at the last snapshot, for the age refinement below.
  const trending = new Map<number, number>();
  if (last.top_trending) {
    for (const g of last.top_trending) trending.set(g.universe_id, g.rank);
  }

  // Candidate universe: games on the last chart, plus games that were in the
  // top 40 in the last 48h but have since left (they can climb back).
  const candidateIds = new Set<number>();
  for (const g of last.up_and_coming) candidateIds.add(g.universe_id);
  const cutoff48 = lastMs - 48 * HOUR_MS;
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
      ageDays = (t0ms - c) / DAY_MS;
    } else {
      // No publication date: age is at least the time since first seen (plus
      // a ~2 day lag between launch and chart entry). Refine with the Top
      // Trending gap, which correlates -0.57 with age at the last read:
      // games that rank far better on Up-and-Coming than on Trending are
      // young. The gap estimate can only raise the age, never lower it below
      // what first-seen already proves.
      const gap = (trending.get(uid) ?? 120) - rank0;
      const gapAge = Math.max(
        0.5,
        Math.min(14, params.gapIntercept + params.gapSlope * gap),
      );
      ageDays = Math.max((t0ms - fs) / DAY_MS + 2, gapAge);
    }
    ageDays = Math.max(0, Math.min(27.9, ageDays));
    const ageOutHour = c !== undefined ? (c + 28 * DAY_MS - t0ms) / HOUR_MS : null;
    const vObs = velocityAt(uid, lastMs);
    const vAge = ageVelocity(params.ageProfile, ageDays);
    games.push({
      id: uid,
      score: transform.scoreOfRank(rank0),
      rank0,
      idioVel: vObs - vAge,
      logG24: logGrowthAt(uid, lastMs),
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
    if (t === firstMs) continue;
    const arr = traj.get(uid);
    if (arr && arr.length > 0) debutRanks.push(arr[0]![1]);
  }
  debutRanks.sort((a, b) => a - b);

  // Entrant rate per hour (new games per hour over the history).
  const spanH = Math.max(1, (lastMs - firstMs) / HOUR_MS);
  const entrantsPerHour =
    debutRanks.length > 5 ? debutRanks.length / spanH : params.entrantsPerHour;

  // Sigma: robust scale of the drift model's own residuals. For each 24h
  // prediction triple (i, j, k) and each top-40 game at j, compare the actual
  // rank change j->k with the change the drift model predicts from the
  // features at j. The residual (in score units, per sqrt hour) is the
  // unpredictable part; its MAD x 1.4826 is sigma. A raw MAD of score changes
  // would double-count the predictable drift, so the drift is subtracted
  // first. Fallback 0.025.
  let sigma = 0.025;
  if (params.sigma > 0) {
    sigma = params.sigma;
  } else {
    const residuals: number[] = [];
    const times = snapshots.map((s) => new Date(s.at).getTime());
    for (let j = 1; j < snapshots.length - 1; j++) {
      let iIdx = -1;
      let kIdx = -1;
      let bestI = Infinity;
      let bestK = Infinity;
      for (let i = 0; i < j; i++) {
        const d = Math.abs(times[j]! - times[i]! - 24 * HOUR_MS);
        if (d < bestI) {
          bestI = d;
          iIdx = i;
        }
      }
      for (let k = j + 1; k < snapshots.length; k++) {
        const d = Math.abs(times[k]! - times[j]! - 24 * HOUR_MS);
        if (d < bestK) {
          bestK = d;
          kIdx = k;
        }
      }
      if (iIdx < 0 || kIdx < 0) continue;
      const d1h = (times[j]! - times[iIdx]!) / HOUR_MS;
      const d2h = (times[kIdx]! - times[j]!) / HOUR_MS;
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
          c !== undefined ? (times[j]! - c) / DAY_MS : (times[j]! - fs) / DAY_MS + 2;
        const vObs = velocityAt(uid, times[j]!);
        const idio = vObs - ageVelocity(params.ageProfile, ageJ);
        const logG = logGrowthAt(uid, times[j]!);
        const predVel =
          params.ageWeight * ageVelocity(params.ageProfile, ageJ) +
          params.momentumSlope * idio +
          params.playingGrowthSlope * logG;
        const actualVel = (rK - g.rank) / d2h;
        const residRanks = (actualVel - predVel) * d2h;
        // Convert a rank residual to a score residual per sqrt hour.
        const dS = Math.abs(transform.dScoreDrank(g.rank));
        residuals.push((residRanks * dS) / Math.sqrt(d2h));
      }
    }
    if (residuals.length > 30) {
      residuals.sort((a, b) => a - b);
      const med = residuals[Math.floor(residuals.length / 2)]!;
      const absDev = residuals
        .map((d) => Math.abs(d - med))
        .sort((a, b) => a - b);
      const mad = absDev[Math.floor(absDev.length / 2)]!;
      const est = 1.4826 * mad;
      if (est > 0.005 && est < 0.1) sigma = est;
    }
  }

  return {
    snapshots,
    t0ms,
    windowHours,
    places,
    games,
    debutRanks,
    sigma,
    entrantsPerHour,
  };
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
  /** place -> game id holding it in the last snapshot (persistence blend). */
  lastSnap: Map<number, number>;
}

export function simulate(hist: ParsedHistory, params: Params): SimResult {
  const { games, windowHours, places, debutRanks, sigma } = hist;
  const hours = Math.round(windowHours);
  const transform = makeTransform(params.exponent);
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
    // rank the climb stops at. Incumbents never break out: a surge of theirs
    // is already in their playing-growth drift.
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
        scores.push(transform.scoreOfRank(debut));
        idioVel.push(0);
        logG24.push(0);
        ageDays.push(0);
        ageOutHour.push(h + 28 * 24);
        if (rng.uniform() < params.breakoutShare) {
          // Sample a measured climb and turn it into a score drift in the
          // current transform's units: the climb covers debut rank -> rank 10
          // in the measured number of hours.
          const climb =
            BREAKOUT_CLIMBS[Math.floor(rng.uniform() * BREAKOUT_CLIMBS.length)]!;
          const drift =
            (transform.scoreOfRank(10) - transform.scoreOfRank(climb[0])) /
            climb[1];
          breakoutDrift.push(drift);
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
          const r = transform.rankOfScore(scores[i]!);
          if (rng.uniform() < disappearHazard(r)) {
            alive[i] = false;
            continue;
          }
        }
        const r = transform.rankOfScore(scores[i]!);
        if (breakoutDrift[i]! > 0 && r > breakoutPeak[i]!) {
          // Breakout still climbing: sampled score drift, applied directly.
          scores[i] = scores[i]! + breakoutDrift[i]! + sigma * rng.gaussian();
          continue;
        }
        if (breakoutDrift[i]! > 0) breakoutDrift[i] = 0; // reached its peak
        const dScoreDrank = transform.dScoreDrank(r);
        // Life-cycle drift at the game's current age (age advances hourly).
        const ageNow = ageDays[i]! + h / 24;
        let driftRanksPerHour = params.ageWeight * ageVelocity(params.ageProfile, ageNow);
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
      // Expected-rank tracking for the top 40 (for the top10 ordering).
      const topN = Math.min(40, order.length);
      for (let k = 1; k <= topN; k++) {
        const id = ids[order[k - 1]!]!;
        if (id < 0) continue;
        rankSum.set(id, (rankSum.get(id) ?? 0) + k);
        rankCount.set(id, (rankCount.get(id) ?? 0) + 1);
      }
    }
  }

  // Normalise counts to probabilities.
  for (const [, m] of prob) {
    for (const [id, c] of m) m.set(id, c / totalSamples);
  }
  for (const [k, c] of unnamed) unnamed.set(k, c / totalSamples);
  for (const [id, c] of top10Mass) top10Mass.set(id, c / totalSamples);
  const expectedRank = new Map<number, number>();
  for (const [id, s] of rankSum) {
    expectedRank.set(id, s / (rankCount.get(id) ?? 1));
  }
  // Last snapshot's top 10, for the persistence blend in assemble().
  const lastSnap = new Map<number, number>();
  const lastSnapshot = hist.snapshots[hist.snapshots.length - 1];
  if (lastSnapshot) {
    for (const g of lastSnapshot.up_and_coming) {
      if (g.rank >= 1 && g.rank <= places) lastSnap.set(g.rank, g.universe_id);
    }
  }
  return { prob, unnamed, expectedRank, top10Mass, lastSnap };
}

// ---- Forecast assembly ----
export interface Forecast {
  probabilities: Array<{ universe_id: number; place: number; p: number }>;
  top10: number[];
}

// Persistence blend: mix a small weight of the game holding a place in the
// last snapshot into that place's Monte Carlo distribution. The blend
// preserves the place's total named mass: the last-snapshot game gets
// weight * rawSum on top of its (1 - weight)-scaled Monte Carlo mass, and
// every other game keeps its (1 - weight)-scaled mass. A no-op when the
// weight is 0 or the place has no last-snapshot game.
function blendWithLastSnap(
  m: Map<number, number>,
  lastSnapId: number | undefined,
  weight: number,
): Map<number, number> {
  if (weight <= 0 || lastSnapId === undefined) return m;
  let rawSum = 0;
  for (const p of m.values()) rawSum += p;
  const out = new Map<number, number>();
  for (const [id, p] of m) out.set(id, (1 - weight) * p);
  out.set(lastSnapId, (out.get(lastSnapId) ?? 0) + weight * rawSum);
  return out;
}

export function assemble(sim: SimResult, places: number, params: Params): Forecast {
  const probabilities: Forecast['probabilities'] = [];
  for (let k = 1; k <= places; k++) {
    const m = blendWithLastSnap(
      sim.prob.get(k) ?? new Map<number, number>(),
      sim.lastSnap.get(k),
      params.lastSnapBlend,
    );
    const rows = [...m.entries()]
      .filter(([, p]) => p >= params.minNamedP)
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, params.maxRowsPerPlace);
    let sum = 0;
    for (const [, p] of rows) sum += p;
    // Keep at least minUnnamed unnamed: if the named rows already leave less,
    // scale them down to fit. The residual is the mass on games we do not
    // name plus future entrants.
    const cap = 1 - params.minUnnamed;
    const scale = sum > cap ? cap / sum : 1;
    for (const [id, p] of rows) {
      probabilities.push({ universe_id: id, place: k, p: round6(p * scale) });
    }
  }

  // top10: the ten games with the most top-10 mass, ordered by median place.
  // The median (not the mean) is the order statistic the agreement tie-break
  // rewards: it minimises expected absolute deviation from the actual place.
  const medianPlace = new Map<number, number>();
  for (const [id] of sim.top10Mass) {
    let den = 0;
    for (let k = 1; k <= places; k++) den += sim.prob.get(k)?.get(id) ?? 0;
    let cum = 0;
    let med = 99;
    for (let k = 1; k <= places; k++) {
      cum += sim.prob.get(k)?.get(id) ?? 0;
      if (den > 0 && cum >= den / 2) {
        med = k;
        break;
      }
    }
    medianPlace.set(id, med);
  }
  const top10 = [...sim.top10Mass.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, places)
    .sort(
      (a, b) =>
        (medianPlace.get(a[0]) ?? 99) - (medianPlace.get(b[0]) ?? 99) ||
        a[0] - b[0],
    )
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

const isEntry =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isEntry) main();
