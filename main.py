#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster.

make.fun competition gta6-trophy-rarity-forecast-01: read one case (the
house's recorded reads of Exophase's GTA VI PS5 trophy page before a
checkpoint) and write, for every trophy, five quantiles of its earn rate
at the case's target.

Model: a rate is earners / tracked pool, and in launch week the two move
very differently -- the pool grows by dilution as new players sync, while
each trophy's earner count grows and bends as it saturates.  So the two
are extrapolated separately on a ln(1 + hours since release) axis and
divided only at the end:

  * consecutive duplicate reads are collapsed (Exophase refreshes its
    counts in steps, so stale repeats are a single observation);
  * the pool's ln is fitted against ln(1 + hours) over the last 36 h of
    distinct reads, its slope shrunk toward a launch prior and damped;
  * each trophy's ln(count + 0.5) is fitted the same way, its slope
    shrunk toward the cross-trophy median slope and damped harder,
    because cumulative counts bend as they saturate;
  * structural facts are then enforced on the forecast counts: the
    story chain (Chapter 1 >= 2 >= 3 >= 4 >= final mission) is
    non-increasing, the platinum cannot exceed the rarest other trophy,
    and a trophy every tracked player holds tracks the pool;
  * the spread on the logit scale combines a random-walk term, the two
    fits' slope uncertainties over the extrapolation distance, and
    counting noise -- independent sources, combined in quadrature --
    floored and capped.

Standard library only, deterministic, no network.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

POOL_PRIOR = 1.25      # launch-week pool slope on the ln(1 + hours) axis
POOL_PRIOR_SD = 0.55
POOL_DAMP = 0.80
EARN_PRIOR_SD = 0.50
EARN_DAMP = 0.62       # cumulative earner counts bend as they saturate
EARN_POOL_FRACTION = 0.65  # no-fit fallback: earners trail the pool's growth
WINDOW_HOURS = 36.0
SPREAD_FLOOR = 0.45
SPREAD_CAP = 2.40
RW_COEF = 0.40         # random-walk spread per sqrt(day)
LOGIT_CAP = 3.0        # a median stays within this logit distance of the last rate
POOL_SD_FLOOR = 0.05   # allowance for Exophase's step refreshes


def parse_time(text):
    """Parse an ISO 8601 timestamp (or a bare date) to an aware datetime."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s + 'T00:00:00+00:00')
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def logit_pct(x):
    p = min(max(float(x), 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        p = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        p = e / (1.0 + e)
    return 100.0 * p


def wls(xs, ys, ws):
    """Weighted least squares fit y ~ a + b*x.

    Returns (a, b, se_b), or None when the fit is unusable (fewer than
    two points, no weight, or no spread in x).  se_b is None when there
    are no residual degrees of freedom; the caller substitutes a
    noise-based default.  An unusable fit is reported as None, never as
    a slope of 0.0: a finite zero would silently route a one-point
    series down a "measured flat" path it never measured.
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0.0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    se = None
    if n > 2:
        rss = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
        if rss > 0.0:
            se = math.sqrt((rss / (n - 2.0)) / sxx)
    return (a, b, se)


def shrink(slope, se, prior, prior_sd, se_default):
    """Precision-weighted pull of a fitted slope toward a prior slope.

    Returns (posterior mean, posterior sd).  With no fit at all, the
    prior stands on its own.
    """
    if slope is None:
        return prior, prior_sd
    s = se if (se is not None and se > 0.0) else se_default
    s = max(s, 0.02)
    w_fit = 1.0 / (s * s)
    w_prior = 1.0 / (prior_sd * prior_sd)
    mean = (slope * w_fit + prior * w_prior) / (w_fit + w_prior)
    sd = math.sqrt(1.0 / (w_fit + w_prior))
    return mean, sd


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def spread_rows(centers, sds):
    """Turn per-trophy median rates and logit-scale spreads into rows."""
    rows = []
    for center, sd in zip(centers, sds):
        lg = logit_pct(center)
        qs = []
        for z in Z:
            q = inv_logit_pct(lg + z * sd)
            q = round(min(max(q, 0.0), 100.0), 4)
            if qs and q < qs[-1]:
                q = qs[-1]
            qs.append(q)
        rows.append(qs)
    return rows


def fallback_rows(trophies, last_rates, days):
    """Persistence-style rows, used when there is nothing to fit."""
    sd = 0.5 * math.sqrt(max(days, 0.0))
    centers = []
    sds = []
    for j in range(len(trophies)):
        r = last_rates[j]
        centers.append(r if (r is not None and r > 0.0) else 0.1)
        sds.append(sd)
    return spread_rows(centers, sds)


def forecast_case(case):
    """Return the list of 5-quantile rows, in the case's trophy order."""
    trophies = case.get('trophies') or []
    n_tr = len(trophies)
    if n_tr == 0:
        return []
    history = case.get('history') or []
    at_dt = parse_time(case.get('at'))
    target_dt = parse_time(case.get('target'))
    horizon = case.get('horizon_minutes')
    if horizon is not None:
        days = max(0.0, float(horizon)) / 1440.0
    elif at_dt is not None and target_dt is not None:
        days = max(0.0, (target_dt - at_dt).total_seconds()) / 86400.0
    else:
        days = 0.0
    game = case.get('game') or {}
    release_dt = parse_time(game.get('release_date')) or at_dt or target_dt
    if release_dt is None:
        return fallback_rows(trophies, [None] * n_tr, days)

    def hours(dt):
        return (dt - release_dt).total_seconds() / 3600.0

    # Collect the reads: (hours since release, players tracked, rates).
    reads = []
    for r in history:
        rt = parse_time(r.get('at'))
        if rt is None:
            continue
        rates = list(r.get('rates') or [])
        if len(rates) < n_tr:
            rates = rates + [None] * (n_tr - len(rates))
        reads.append((hours(rt), r.get('players_tracked'), rates))

    # Collapse consecutive duplicate reads: Exophase refreshes in steps,
    # so a run of identical reads is one observation of the same state,
    # timestamped at the first read of the run -- the level has been in
    # force since then, and growth after it is real but unobserved.
    distinct = []
    for item in reads:
        key = (item[1], tuple(item[2]))
        if distinct and distinct[-1][0] == key:
            continue
        distinct.append((key, item))
    distinct = [item for _, item in distinct]

    # The anchor is the last distinct observation: its level, read at its
    # own (first-seen) time.  The last rate seen for each trophy is the
    # same value the final duplicate run carries.
    last_rates = [None] * n_tr
    for _, _, rates in reversed(distinct):
        for j in range(n_tr):
            if last_rates[j] is None and rates[j] is not None:
                last_rates[j] = rates[j]
        if all(v is not None for v in last_rates):
            break
    anchor = None
    for item in reversed(distinct):
        if item[1] is not None and item[1] > 0:
            anchor = item
            break
    if anchor is None:
        return fallback_rows(trophies, last_rates, days)
    t_last, p_last, _ = anchor
    if target_dt is not None:
        t_target = hours(target_dt)
    else:
        t_target = t_last + days * 24.0

    def x_of(t):
        return math.log(max(1.0 + t, 1e-6))

    dx = max(0.0, x_of(t_target) - x_of(t_last))

    window = [item for item in distinct
              if item[0] >= t_last - WINDOW_HOURS]

    # --- the tracked pool -------------------------------------------------
    px, py, pw = [], [], []
    for t, players, _ in window:
        if players is not None and players > 0 and 1.0 + t > 0.0:
            px.append(x_of(t))
            py.append(math.log(players))
            pw.append(1.0)
    pool_fit = wls(px, py, pw)
    if pool_fit is not None:
        pool_post, pool_sd = shrink(pool_fit[1], pool_fit[2],
                                    POOL_PRIOR, POOL_PRIOR_SD, 0.35)
    else:
        pool_post, pool_sd = POOL_PRIOR, POOL_PRIOR_SD
    pool_slope = POOL_DAMP * pool_post
    ln_pool = math.log(p_last) + pool_slope * dx
    pool_hat = math.exp(min(ln_pool, math.log(1e12)))
    pool_term = POOL_DAMP * pool_sd * dx + POOL_SD_FLOOR

    # --- each trophy's earner count ---------------------------------------
    # Last observed count per trophy (rate paired with its own read's pool).
    last_count = [None] * n_tr
    for t, players, rates in reversed(distinct):
        if players is None or players <= 0:
            continue
        for j in range(n_tr):
            if last_count[j] is None and rates[j] is not None:
                last_count[j] = rates[j] * players / 100.0
        if all(v is not None for v in last_count):
            break

    fits = []
    slopes = []
    for j in range(n_tr):
        xs, ys, ws = [], [], []
        for t, players, rates in window:
            if players is None or players <= 0 or rates[j] is None:
                continue
            if 1.0 + t <= 0.0:
                continue
            c = rates[j] * players / 100.0
            xs.append(x_of(t))
            ys.append(math.log(c + 0.5))
            ws.append(c + 0.5)
        fit = wls(xs, ys, ws)
        fits.append(fit)
        if fit is not None:
            slopes.append(fit[1])
    med_slope = median(slopes)

    earn_hat = [None] * n_tr
    slope_sds = [0.0] * n_tr
    for j in range(n_tr):
        if last_count[j] is None:
            continue
        if last_rates[j] is not None and last_rates[j] >= 99.9:
            # Earned by (nearly) every tracked player: it tracks the pool.
            earn_hat[j] = pool_hat * last_rates[j] / 100.0
            slope_sds[j] = 0.0
            continue
        fit = fits[j]
        if med_slope is not None:
            prior = med_slope
        else:
            # No trophy has a usable fit: the cross-trophy prior falls
            # back to a fraction of the pool's slope, since in launch
            # week earner counts trail the tracked pool's growth.
            prior = EARN_POOL_FRACTION * pool_post
        if fit is not None:
            post, post_sd = shrink(fit[1], fit[2], prior, EARN_PRIOR_SD, 0.5)
        else:
            post, post_sd = prior, EARN_PRIOR_SD
        ln_e = math.log(last_count[j] + 0.5) + EARN_DAMP * post * dx
        e = math.exp(min(ln_e, math.log(1e12))) - 0.5
        earn_hat[j] = max(e, 0.0)
        slope_sds[j] = EARN_DAMP * post_sd * dx

    # No earner count may exceed the pool it is a share of.
    for j in range(n_tr):
        if earn_hat[j] is not None and earn_hat[j] > pool_hat:
            earn_hat[j] = pool_hat

    # --- structural constraints on the counts ------------------------------
    chain_descs = ('Complete Chapter 1.', 'Complete Chapter 2.',
                   'Complete Chapter 3.', 'Complete Chapter 4.',
                   'Complete the final mission.')
    by_desc = {}
    for j, t in enumerate(trophies):
        by_desc.setdefault(str(t.get('description') or '').strip(), j)
    chain = [by_desc[d] for d in chain_descs if d in by_desc]
    for prev, cur in zip(chain, chain[1:]):
        if earn_hat[prev] is not None and earn_hat[cur] is not None:
            if earn_hat[cur] > earn_hat[prev]:
                earn_hat[cur] = earn_hat[prev]
    plat = None
    for j, t in enumerate(trophies):
        if str(t.get('tier') or '').strip().lower() == 'platinum':
            plat = j
            break
    if plat is not None and earn_hat[plat] is not None:
        others = [earn_hat[j] for j in range(n_tr)
                  if j != plat and earn_hat[j] is not None]
        if others:
            earn_hat[plat] = min(earn_hat[plat], min(others))

    # --- medians and spreads ------------------------------------------------
    rw = RW_COEF * math.sqrt(days)
    centers = []
    sds = []
    for j in range(n_tr):
        if earn_hat[j] is None or last_count[j] is None:
            centers.append(0.1)
            sds.append(1.8)
            continue
        rate = 100.0 * earn_hat[j] / pool_hat if pool_hat > 0 else 0.0
        if last_rates[j] is not None:
            lo = logit_pct(last_rates[j]) - LOGIT_CAP
            hi = logit_pct(last_rates[j]) + LOGIT_CAP
            lg = min(max(logit_pct(rate), lo), hi)
            rate = inv_logit_pct(lg)
        centers.append(min(max(rate, 0.01), 99.99))
        count_term = min(math.sqrt(1.0 / (last_count[j] + 0.5)), 0.6)
        sd = math.sqrt(rw * rw + slope_sds[j] ** 2 + pool_term ** 2
                       + count_term * count_term)
        sds.append(min(max(sd, SPREAD_FLOOR), SPREAD_CAP))
    return spread_rows(centers, sds)


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith('.in'))
    for name in names:
        with open(os.path.join(in_dir, name), 'r', encoding='utf-8') as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        trophies = case.get('trophies') or []
        out = {'trophies': [{'id': t.get('id'), 'q': q}
                            for t, q in zip(trophies, rows)]}
        stem = name[:-3] if name.endswith('.in') else name
        with open(os.path.join(out_dir, stem + '.out'), 'w',
                  encoding='utf-8') as fh:
            fh.write(json.dumps(out, separators=(',', ':')))


if __name__ == '__main__':
    main()
