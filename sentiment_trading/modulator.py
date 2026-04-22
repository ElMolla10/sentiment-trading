"""
Multiplicative sentiment modulation with sign-aware short handling (paper III-E, III-F).
Temporal decay (paper III-G) and graceful degradation (paper III-H).

Formula:
    if s == 0 or p == 0:
        final = p
    elif sign(p) == sign(s):          # aligned → amplify
        final = p * (1 + |s|)
    else:                              # opposed → attenuate
        final = p * (1 - |s|)

For shorts (p < 0), the sign test automatically inverts the effect:
  positive news (s > 0) → opposed → attenuates the short conviction.
  negative news (s < 0) → aligned → amplifies the short conviction.
"""
import math


def apply_decay(s: float, dt_minutes: float, tau: float = 90.0) -> float:
    """
    Exponential decay of sentiment magnitude over time (paper III-G, refs [13][14]).
    tau is the half-life equivalent time constant in minutes (60-120 minute range).
    """
    if dt_minutes <= 0:
        return s
    return s * math.exp(-dt_minutes / tau)


def modulate(p: float, s: float) -> float:
    """
    Core multiplicative modulation formula (paper III-E).

    p : base price-based prediction in (-inf, +inf).
        sign(p) encodes direction: positive = long, negative = short.
    s : sentiment score in [-1, +1] from the selected model.
        Positive = bullish news, negative = bearish news, 0 = neutral.

    Returns the modulated prediction.
    """
    if s == 0.0 or p == 0.0:
        return p

    aligned = (p > 0) == (s > 0)  # sign(p) == sign(s)
    abs_s = abs(s)

    if aligned:
        return p * (1.0 + abs_s)
    else:
        return p * (1.0 - abs_s)


def modulate_with_decay(
    p: float,
    s: float,
    dt_minutes: float = 0.0,
    tau: float = 90.0,
    s_available: bool = True,
) -> float:
    """
    Full modulation pipeline with temporal decay and graceful degradation (paper III-H).

    dt_minutes : minutes since the news item was published.
    s_available: False when the news pipeline is unavailable; forces s=0 so
                 final_prediction == p (pass-through).
    """
    if not s_available:
        return p  # graceful degradation: no sentiment → unchanged prediction

    s_decayed = apply_decay(s, dt_minutes, tau)
    return modulate(p, s_decayed)
