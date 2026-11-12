# Roblox Up-and-Coming Top 10 Forecaster

Entry for the [make.fun](https://make.fun) bounty
**"Predict Roblox's Up-and-Coming top 10"**
(`roblox-up-and-coming-forecast-01`).

## What it does

`main.ts` reads hourly snapshots of Roblox's Up-and-Coming chart from
`/input` (`history.ndjson` + `task.json`) and writes one forecast to
`/output/forecast.json`: for each of the top 10 places, the probability of
each game it names holding that place, plus its guess at the top 10 in
order — covering every hour of the 72 hours after entries close.

## How it works

A Monte Carlo simulation of the chart in latent-score space. Each game
carries a score `s = -(rank)^(1/3)` (rank mobility grows with rank, roughly
as `r^(2/3)`, which is what a fixed score noise produces under this
transform; a square-root transform was tested and scored worse on both
backtest windows). Every simulated hour each game's score moves by drift +
noise:

- **Drift** has three parts, fitted jointly by least squares on the history
  (next-24h rank velocity on past velocity, playing growth and age;
  R² 0.17):
  - *life-cycle* (weight 0.73): games climb fast in their first 2 days
    (−0.62 ranks/hour, measured from debutants' first 48 hours), then
    flatten to a slow fall (+0.06 to +0.07 ranks/hour past day 5). The
    old-age end is deliberately flatter than the raw measurement (+0.20 at
    6–10 days): the raw level includes the common downward drift that the
    simulation already produces through entrants, and the 72-hour backtests
    confirm the flatter curve. Age comes from `details.created` when
    present, else first-seen plus a ~2 day lag, refined by the Top Trending
    gap (a game's trending rank minus its up-and-coming rank correlates
    −0.57 with its age; fitted age = 8.67 − 0.048 × gap, used only to raise
    the age above what first-seen already proves).
  - *momentum* (slope 0.11): the idiosyncratic part of recent rank velocity
    persists and fades over ~36 hours.
  - *playing growth* (slope −0.40 ranks/hour per unit of 24h log playing
    growth): a surge in concurrent players leads a rank climb by about a
    day — the chart is computed on a delay, not live. The slope is the
    compromise between the joint-OLS fits on the two backtest windows
    (−0.34 and −0.42); it costs almost nothing on the practice window and
    gains half a point on the second. Full strength for the
    first 24 hours, then decaying.
- **Diffusion**: hourly Gaussian noise with σ estimated from the history
  as the robust spread of the drift model's own residuals on 24-hour
  prediction triples (σ ≈ 0.09 on the practice history).
- **New entrants** arrive at the measured rate (~13/day), debuting near
  the bottom (median rank 87); ~16% are breakouts that climb to the top 10
  with a drift and peak rank sampled from the 17 measured debut-to-top-10
  climbs (median 45 hours to the top 10). Entrants are anonymous, so their
  mass flows into each place's unnamed share — as it should, since 23% of
  top-10 slots 72 hours out are games that were not on the chart at all.
- **Age-out**: games leave the chart 28 days after publication. Games
  without a known publication date instead face a measured disappearance
  hazard (0.001/hour in the top 10, 0.002 at ranks 11–60, 0.018 below 60).

The forecast is the honest time-average over the window: the fraction of
(simulation, hour) pairs where each game held each place. Each place keeps
at least a 15% unnamed share (the measured time-averaged share of top-10
slots held by games not on the chart was 8% in the practice window and 16%
in a second backtest window). 5,000 simulations, seeded PRNG — two runs on
the same input produce the same `forecast.json` byte for byte.

## Scores (local scorer replica — this competition has no local grade)

| Window | House baseline (persistence) | This entry |
| --- | --- | --- |
| Practice, 10 reads 8–10 Nov 2026 | −43.512 | **−32.976** |
| Second backtest, 8 reads 6–7 Nov 2026 | −52.125 | **−32.583** |

The scorer replica implements the kit's `RULES.txt` scoring exactly and was
validated by reproducing the house baseline's published practice score
(−43.512039) to the digit.

Parameter choices were checked on both windows, and a tuning was adopted
only if it did not sacrifice one window for the other: the square-root
latent transform, a steeper trending-gap age fit, and fixed σ values were
all tested and rejected on that basis.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+ (type stripping), standard library only, no network, no model
calls. Runtime ~25 s, well within the 300 s limit.

## License

MIT — see `LICENSE`.
