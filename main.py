"""gta6-trophy-rarity-forecast-01 entry: a counts-decomposition forecaster.

Why not just extrapolate each trophy's rate? A rate is earners / players_tracked, and in the
first week of a launch the two move very differently: the tracked pool grows in a rush as new
players register the game, while earners accumulate as people actually play. A rate can fall
for days while its earners only grow (pure dilution), then turn up once pool growth slows.
Fitting the rate itself confuses the two motions, so this program fits them separately:

  * pool:    ln(players_tracked) is fit against ln(1 + hours since release) over the recent
             window, and read at the target from the last observed level;
  * earners: for each trophy, ln(earners + 0.5) is fit the same way, its slope shrunk toward
             the cross-trophy median slope (tiny counts borrow the field's behaviour),
             damped before extrapolation (cumulative counts bend as they saturate), and
             read at the target from the last observed count;
  * the median rate is 100 * earners / pool at the target, with the story chain's counts
    forced non-increasing (Chapter 1 >= Chapter 2 >= ... >= the final mission) and the
    platinum's count capped at the smallest other count, as the game's logic requires.

Consecutive identical reads are one observation (the page had not refreshed), not many.

The spread is on the logit scale, per trophy: a random-walk term growing with the horizon,
the fits' own slope uncertainty over the extrapolation distance, the pool's uncertainty, and
a counting-noise term for trophies with few earners. Quantiles are that normal's.

Runs on the standard library alone, offline, deterministically:
`python3 -P -s /bot/main.py` with the case at /input (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for those outside the grader).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside
# this file would import. (This entry is a single file; the line costs nothing.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# --- Tunables (fixed in advance of the final; see README for the reasoning) ---------------

# Random-walk volatility of a rate on the logit scale, per square-root day of horizon.
SIGMA_RW = 0.40
# Prior slopes in ln(1 + hours since release) space, used when a series is too short or too
# flat to fit: the tracked pool of a just-launched game grows fast (diluting early rates),
# while earners accumulate more slowly at first.
POOL_SLOPE_PRIOR = 1.25
EARNER_SLOPE_PRIOR = 0.50
# How many observations of evidence outweigh the prior / the cross-trophy median slope.
PRIOR_FADE = 5.0
# Recency weighting for the slope fits: a read's weight halves every HALF_LIFE_H hours.
HALF_LIFE_H = 20.0
# Slope fits use the last SLOPE_WINDOW_H hours of distinct reads (all of them when the
# history is shorter): older reads describe a regime the launch has already left.
SLOPE_WINDOW_H = 36.0
# A fitted slope is damped before it is extrapolated: cumulative counts bend as they
# saturate, so the average slope over the next stretch is lower than the recent one.
# The pool bends less than earners do. Priors are not damped: a prior already names the
# forward slope it expects, it is not a measured steepness.
EARNER_DAMP = 0.62
POOL_DAMP = 0.80
# A median rate may not run away from the last observed one: generous, purely a net.
MEDIAN_CAP_RISE = 2.2
MEDIAN_CAP_ADD = 18.0
# Slopes are clipped to sane ranges for a launch week.
POOL_SLOPE_CLIP = (0.0, 1.8)
EARNER_SLOPE_CLIP = (0.0, 2.4)
# Predictive spread (logit sd) is kept inside these bounds.
SPREAD_FLOOR = 0.45
SPREAD_CAP = 2.40
# Default fit uncertainties when a series is too short to estimate its own.
DEFAULT_FIT_SE = 0.55
DEFAULT_POOL_SE = 0.35

Z = NormalDist()


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 does.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


# --- Small weighted-least-squares helper ---------------------------------------------------


def wls_slope(points: list[tuple[float, float, float]]) -> tuple[float | None, float]:
    """Slope and its standard error for (x, y, weight) points; (None, inf) when unusable."""
    n = len(points)
    if n < 2:
        return None, math.inf
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None, math.inf
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None, math.inf
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    if n > 2:
        resid = sum(w * (y - (my + slope * (x - mx))) ** 2 for x, y, w in points)
        dof = max(1.0, (n - 2) * (sw / n))
        se = math.sqrt(max(resid, 0.0) / dof / sxx)
    else:
        se = math.inf
    return slope, se


def shrink(value: float, reference: float, evidence: float) -> float:
    """Pull `value` toward `reference`; `evidence` observations outweigh the reference."""
    lam = evidence / (evidence + PRIOR_FADE) if evidence > 0 else 0.0
    return lam * value + (1.0 - lam) * reference


# --- Case preparation ----------------------------------------------------------------------


def distinct_reads(case: dict) -> list[dict]:
    """The case's reads with consecutive duplicates collapsed (a stale page is one read).

    Each kept read carries the LAST slot at which its values were seen: the values held
    until then. Reads with no usable pool count are dropped.
    """
    kept: list[dict] = []
    for read in case["history"]:
        players = read.get("players_tracked") or 0
        if players <= 0:
            continue
        rates = read.get("rates") or []
        if kept and kept[-1]["players_tracked"] == players and kept[-1]["rates"] == rates:
            kept[-1]["slot"] = read["slot"]
            kept[-1]["at"] = read.get("at", read["slot"])
        else:
            kept.append(
                {
                    "slot": read["slot"],
                    "at": read.get("at", read["slot"]),
                    "players_tracked": players,
                    "rates": list(rates),
                }
            )
    return kept


def story_chain(trophies: list[dict]) -> list[int]:
    """Indices of the story-progression trophies, in order: Chapter 1..N, then the finale."""
    chapters: list[tuple[int, int]] = []
    finale: list[int] = []
    for i, trophy in enumerate(trophies):
        desc = trophy.get("description") or ""
        m = re.search(r"Complete Chapter (\d+)", desc)
        if m:
            chapters.append((int(m.group(1)), i))
        elif re.search(r"^Complete the final mission\.?$", desc.strip()):
            finale.append(i)
    chapters.sort()
    return [i for _, i in chapters] + finale


# --- The forecast --------------------------------------------------------------------------


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    levels = case["levels"]
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target_t = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def u_of(when: datetime) -> float:
        hours = (when - release).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    reads = distinct_reads(case)
    if not reads:
        raise ValueError("case has no usable history")

    times = [parse_time(r["slot"]) for r in reads]
    u = [u_of(t) for t in times]
    u_last = u[-1]
    u_target = u_of(target_t)
    span_u = max(0.0, u_target - u_last)
    t_last_h = (times[-1] - release).total_seconds() / 3600.0

    def slope_points(idxs: list[int]) -> list[int]:
        """Indices whose reads fall in the slope window; all of them if too few do."""
        recent = [
            j for j in idxs
            if (times[-1] - times[j]).total_seconds() / 3600.0 <= SLOPE_WINDOW_H
        ]
        return recent if len(recent) >= 3 else idxs

    def weight_of(j: int) -> float:
        age_h = (times[-1] - times[j]).total_seconds() / 3600.0
        return 0.5 ** (age_h / HALF_LIFE_H)

    # -- Pool: fit ln(players) on u over the slope window, shrink the slope to the launch
    # prior when data is thin, damp a fitted slope, and read from the last observed level.
    all_idx = list(range(len(reads)))
    pool_idx = slope_points(all_idx)
    pool_pts = [(u[j], math.log(reads[j]["players_tracked"]), weight_of(j)) for j in pool_idx]
    pool_slope_raw, pool_se_raw = wls_slope(pool_pts)
    if pool_slope_raw is not None:
        clipped = min(max(pool_slope_raw, POOL_SLOPE_CLIP[0]), POOL_SLOPE_CLIP[1])
        pool_slope = POOL_DAMP * shrink(clipped, POOL_SLOPE_PRIOR, float(len(pool_idx)))
    else:
        pool_slope = POOL_SLOPE_PRIOR
    pool_se = pool_se_raw if math.isfinite(pool_se_raw) else DEFAULT_POOL_SE
    ln_pool_target = math.log(reads[-1]["players_tracked"]) + pool_slope * span_u
    pool_target = math.exp(ln_pool_target)

    # -- Earners per trophy: fit ln(count + 0.5) on u over the slope window; collect the
    # fitted slopes for the cross-trophy median used to shrink them.
    n_tr = len(trophies)
    last_rate: list[float | None] = [None] * n_tr
    slopes: list[float | None] = [None] * n_tr
    slope_ses: list[float] = [DEFAULT_FIT_SE] * n_tr
    n_points: list[int] = [0] * n_tr
    for i in range(n_tr):
        obs: list[tuple[int, float]] = []
        for j, read in enumerate(reads):
            rate = read["rates"][i] if i < len(read["rates"]) else None
            if rate is None:
                continue
            obs.append((j, rate * read["players_tracked"] / 100.0))
            last_rate[i] = rate
        n_points[i] = len(obs)
        if not obs:
            continue
        anchor_j, anchor_count = obs[-1]
        _ANCHOR_U[i] = u[anchor_j]
        _ANCHOR_LN[i] = math.log(anchor_count + 0.5)
        fit_obs = [
            (j, c) for j, c in obs
            if (times[-1] - times[j]).total_seconds() / 3600.0 <= SLOPE_WINDOW_H
        ]
        if len(fit_obs) < 3:
            fit_obs = obs
        pts = [(u[j], math.log(c + 0.5), weight_of(j)) for j, c in fit_obs]
        if len(pts) >= 2:
            s_raw, se_raw = wls_slope(pts)
            if s_raw is not None:
                slopes[i] = min(max(s_raw, EARNER_SLOPE_CLIP[0]), EARNER_SLOPE_CLIP[1])
                if math.isfinite(se_raw):
                    slope_ses[i] = se_raw

    fitted = [s for s in slopes if s is not None]
    if fitted:
        ordered = sorted(fitted)
        mid = len(ordered) // 2
        ref_slope = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    else:
        ref_slope = EARNER_SLOPE_PRIOR

    counts_target: list[float] = [0.0] * n_tr
    fit_se_i: list[float] = [DEFAULT_FIT_SE] * n_tr
    pool_bound: list[bool] = [False] * n_tr
    for i in range(n_tr):
        if _ANCHOR_LN[i] is None:
            counts_target[i] = 0.25  # never seen earned: a token count, spread will be wide
            continue
        if last_rate[i] is not None and last_rate[i] >= 99.9:
            # Earned by effectively every tracked player: its earners ARE the pool, so
            # they grow with it and the rate holds, instead of diluting like the others.
            counts_target[i] = pool_target * last_rate[i] / 100.0
            pool_bound[i] = True
            continue
        base_slope = slopes[i] if slopes[i] is not None else ref_slope
        slope = shrink(base_slope, ref_slope, float(n_points[i]))
        if fitted:
            slope *= EARNER_DAMP
        counts_target[i] = math.exp(_ANCHOR_LN[i] + slope * (u_target - _ANCHOR_U[i])) - 0.5
        counts_target[i] = max(0.02, counts_target[i])
        fit_se_i[i] = slope_ses[i]

    # -- A generous cap keeps any single median from running away from its last rate.
    for i in range(n_tr):
        if last_rate[i] is not None and not pool_bound[i]:
            cap_rate = min(97.0, last_rate[i] * MEDIAN_CAP_RISE + MEDIAN_CAP_ADD)
            counts_target[i] = min(counts_target[i], pool_target * cap_rate / 100.0)

    # -- Structure the game itself imposes.
    chain = story_chain(trophies)
    for a, b in zip(chain, chain[1:]):
        if counts_target[b] > counts_target[a]:
            counts_target[b] = counts_target[a]
    plat = [i for i, t in enumerate(trophies) if (t.get("tier") or "") == "platinum"]
    if plat:
        others_min = min(counts_target[i] for i in range(n_tr) if i != plat[0])
        counts_target[plat[0]] = min(counts_target[plat[0]], others_min)

    # -- Medians and spreads.
    rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    pool_sd = min(pool_se * span_u, 0.8) if span_u > 0 else DEFAULT_POOL_SE
    rows = []
    for i, trophy in enumerate(trophies):
        med = clamp_rate(100.0 * counts_target[i] / pool_target)
        if pool_bound[i]:
            # The rate is pinned at the boundary; only boundary noise is in play.
            sd = 0.5
            centre = logit(med)
            qs = []
            for tau in levels:
                q = 100.0 / (1.0 + math.exp(-(centre + Z.inv_cdf(tau) * sd)))
                qs.append(round4(min(max(q, 0.0001), 99.9999)))
            rows.append({"id": trophy["id"], "q": sorted(qs)})
            continue
        fit_sd = min(fit_se_i[i] * span_u, 1.2) if span_u > 0 else DEFAULT_FIT_SE
        count_sd = min(0.7 / math.sqrt(max(counts_target[i], 0.25)), 0.7)
        var = rw * rw + fit_sd * fit_sd + (pool_sd * (1.0 - med / 100.0)) ** 2 + count_sd**2
        sd = min(max(math.sqrt(var), SPREAD_FLOOR), SPREAD_CAP)
        centre = logit(med)
        qs = []
        for tau in levels:
            q = 100.0 / (1.0 + math.exp(-(centre + Z.inv_cdf(tau) * sd)))
            qs.append(round4(min(max(q, 0.0001), 99.9999)))
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


# Per-case side channels for anchors (reset per case; the program handles one case at a time
# in the grader, and main() finishes each case before starting the next).
_ANCHOR_U: list[float] = []
_ANCHOR_LN: list[float | None] = []


def forecast(case: dict) -> dict:
    global _ANCHOR_U, _ANCHOR_LN
    n = len(case["trophies"])
    _ANCHOR_U = [0.0] * n
    _ANCHOR_LN = [None] * n
    return forecast_case(case)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
