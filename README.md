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
carries a score `s = -(rank)^(1/3)` (the cube root matches the measured
rank mobility, which grows roughly as `r^(2/3)`). Every simulated hour each
game's score moves by drift + noise:

- **Drift** has three parts, fitted jointly by least squares on the
  history (next-24h rank velocity on past velocity, playing growth and
  age; R² 0.14):
  - *life-cycle* (weight 0.77): games climb fast in their first 2 days
    (−0.64 ranks/hour), then flatten and fall as they age toward the
    28-day window edge. Age comes from `details.created` when present,
    else first-seen plus a lag correction refined by the Top Trending gap.
  - *momentum* (slope 0.125): the idiosyncratic part of recent rank
    velocity persists and fades over ~36 hours.
  - *playing growth* (slope −0.376 ranks/hour per unit of 24h log playing
    growth): a surge in concurrent players leads a rank climb by about a
    day — the chart is computed on a delay, not live. Full strength for
    the first 24 hours, then decaying.
- **Diffusion**: hourly Gaussian noise with σ estimated from the history
  as the robust spread of the drift model's own residuals on 24-hour
  prediction triples.
- **New entrants** arrive at the measured rate (~13/day), debuting near
  the bottom (median rank 87); ~16% are breakouts that climb to the top
  10 with a drift and peak rank sampled from the measured debut-to-top-10
  climbs. Entrants are anonymous, so their mass flows into each place's
  unnamed share — as it should, since 8–16% of top-10 slots turn out to
  be games that were not on the chart at all.
- **Age-out**: games leave the chart 28 days after publication.

The forecast is the honest time-average over the window: the fraction of
(simulation, hour) pairs where each game held each place. Each place keeps
at least a 15% unnamed share. 5,000 simulations, seeded PRNG — two runs on
the same input produce the same `forecast.json` byte for byte.

## Scores (local scorer replica — this competition has no local grade)

| Window | House baseline (persistence) | This entry |
| --- | --- | --- |
| Practice, 10 reads 8–10 Nov 2026 | −43.512 | **−33.518** |
| Second backtest, 8 reads 6–7 Nov 2026 | −52.125 | **−32.561** |

The scorer replica implements the kit's `RULES.txt` scoring exactly and was
validated by reproducing the house baseline's published practice score
(−43.512039) to the digit.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+ (type stripping), standard library only, no network, no model
calls. Runtime ~30 s, well within the 300 s limit.

## License

MIT — see `LICENSE`.
